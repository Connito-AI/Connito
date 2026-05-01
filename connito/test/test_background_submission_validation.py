"""Tests for the background submission validation lifecycle.

Mirrors the (0)..(4) lifecycle described in
`_specs/background-submission-validation.md`. Uses mocks for bittensor
and HF; everything else is exercised against the real classes.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# If the heavy datasets/pandas chain cannot load (e.g. due to a local
# numpy/pandas binary mismatch), stub the modules that pull it in so
# unit tests can still exercise the lifecycle classes. When the real
# modules import cleanly, leave them alone so other test files in the
# same pytest run continue to see the real APIs (e.g. PhaseNames).
import connito.shared as _connito_shared  # noqa: E402


def _install_stub_if_unavailable(mod_path: str, attrs: dict) -> None:
    real_mod_name = mod_path.split(".")[-1]
    try:
        __import__(mod_path)
        return  # real module loaded fine — keep it
    except Exception:
        stub = types.ModuleType(mod_path)
        for k, v in attrs.items():
            setattr(stub, k, v)
        sys.modules[mod_path] = stub
        setattr(_connito_shared, real_mod_name, stub)


_install_stub_if_unavailable(
    "connito.shared.dataloader",
    {"get_dataloader": lambda **kwargs: None},
)
_install_stub_if_unavailable(
    "connito.shared.evaluate",
    {"evaluate_model": lambda *a, **kw: {"val_loss": 100.0}},
)
_install_stub_if_unavailable(
    "connito.shared.cycle",
    {
        "get_combined_validator_seed": lambda config, subtensor: "deadbeef",
        "get_validator_miner_assignment": lambda config, subtensor: {},
        # PhaseNames is referenced by other test modules' shared imports;
        # provide a minimal placeholder so collection of those tests does
        # not fail when this test ran first and installed the stub.
        "PhaseNames": types.SimpleNamespace(
            distribute="Distribute", train="Train",
            miner_commit_1="MinerCommit1", miner_commit_2="MinerCommit2",
            submission="Submission", validate="Validate", merge="Merge",
            validator_commit_1="ValidatorCommit1",
            validator_commit_2="ValidatorCommit2",
        ),
    },
)

from connito.validator.aggregator import MinerScoreAggregator  # noqa: E402
from connito.validator.round import Round, RosterEntry, RoundRef  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(val: float = 0.1) -> nn.Module:
    m = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(val)
    return m


def _make_metagraph(hotkey_to_incentive: dict[str, float]) -> SimpleNamespace:
    hotkeys = list(hotkey_to_incentive.keys())
    incentive = torch.tensor([hotkey_to_incentive[hk] for hk in hotkeys])
    return SimpleNamespace(hotkeys=hotkeys, incentive=incentive)


def _fake_subtensor(metagraph, block: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        block=block,
        metagraph=lambda netuid=None: metagraph,
        network="mock",
    )


def _fake_validator_config(my_hotkey: str = "vhk", group_id: int = 1, netuid: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        chain=SimpleNamespace(hotkey_ss58=my_hotkey, netuid=netuid, network="mock"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=group_id)),
    )


def _freeze_round(
    *,
    config,
    metagraph,
    assignment: dict[str, list[str]],
    miners_with_checkpoint: list[str] | None = None,
    seed: str = "deadbeef",
    round_id: int = 100,
    global_model: nn.Module | None = None,
    chain_checkpoints_by_hotkey: dict[str, "object"] | None = None,
    last_evaluated: dict[int, datetime] | None = None,
    prior_avg_scores: dict[int, float] | None = None,
) -> Round:
    """Build a Round bypassing the chain helpers.

    `chain_checkpoints_by_hotkey` pairs each hotkey with a stub object
    exposing `hf_repo_id` / `hf_revision`. Hotkeys absent from the map
    end up in `freeze_zero_uids`. Default = empty (all miners are
    treated as freeze-zero), which mirrors how the test suite has
    historically run.
    """
    if global_model is None:
        global_model = _make_model()
    subtensor = _fake_subtensor(metagraph, block=round_id)

    if miners_with_checkpoint is None:
        # Default: every miner across all validators' assignments has a
        # checkpoint, ranked by metagraph incentive desc.
        union = set()
        for ms in assignment.values():
            union.update(ms)
        miners_with_checkpoint = sorted(
            union,
            key=lambda hk: (-metagraph.incentive[metagraph.hotkeys.index(hk)].item(), hk),
        )
    assignment_result = SimpleNamespace(
        assignment=assignment,
        miners_with_checkpoint=miners_with_checkpoint,
        chain_checkpoints_by_hotkey=chain_checkpoints_by_hotkey or {},
    )
    with patch("connito.shared.chain.get_chain_commits", return_value=[]), \
         patch("connito.shared.cycle.get_combined_validator_seed", return_value=seed), \
         patch("connito.shared.cycle.get_validator_miner_assignment", return_value=assignment_result):
        return Round.freeze(
            config=config,
            subtensor=subtensor,
            metagraph=metagraph,
            global_model=global_model,
            round_id=round_id,
            last_evaluated=last_evaluated,
            prior_avg_scores=prior_avg_scores,
        )


def _valid_chain_checkpoints(hotkeys: list[str]) -> dict[str, "object"]:
    """Stub `ChainCheckpoint`-shaped objects with `hf_repo_id` and
    `hf_revision` set so `Round.freeze` keeps these hotkeys *out* of
    `freeze_zero_uids`. Use when a test needs miners to be treated as
    "had a valid commit at freeze time".
    """
    return {
        hk: SimpleNamespace(hf_repo_id=f"repo/{hk}", hf_revision="rev123")
        for hk in hotkeys
    }


# ---------------------------------------------------------------------------
# (0) Lock and prioritize
# ---------------------------------------------------------------------------

class TestRoundFreeze:
    def test_roster_ordered_by_incentive_desc(self) -> None:
        # Foreground == this validator's assignment (whole slice). Background
        # is the rest of the roster — for a single-validator universe that
        # set is empty.
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5, "hk_d": 0.3})
        assignment = {"vhk": ["hk_a", "hk_b", "hk_c", "hk_d"]}

        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        assert [e.hotkey for e in rnd.roster] == ["hk_b", "hk_c", "hk_d", "hk_a"]
        assert rnd.foreground_uids == (1, 2, 3, 0)  # all four, incentive desc
        assert rnd.background_uids == ()
        assert rnd.assigned_uids == rnd.foreground_uids

    def test_foreground_and_background_disjoint_and_cover_roster(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5})
        assignment = {"vhk": ["hk_a", "hk_b", "hk_c"]}

        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        all_uids = set(rnd.foreground_uids) | set(rnd.background_uids)
        assert set(rnd.foreground_uids).isdisjoint(rnd.background_uids)
        assert all_uids == {e.uid for e in rnd.roster}

    def test_late_registrant_excluded(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4})
        # Assignment was computed with these two; a third hotkey appearing
        # later (registered after freeze) is not in the assignment, so it's
        # excluded.
        assignment = {"vhk": ["hk_a", "hk_b"]}

        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        assert {e.hotkey for e in rnd.roster} == {"hk_a", "hk_b"}

    def test_other_validators_assignment_lands_in_background(self) -> None:
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4})
        # Other validators' miners are still in the roster — they end up
        # in background_uids so bg-download can fetch them — but they are
        # excluded from foreground and from the penalty pass (assigned_uids).
        assignment = {"vhk": ["hk_a"], "other_validator": ["hk_b"]}

        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        assert {e.hotkey for e in rnd.roster} == {"hk_a", "hk_b"}
        uid_a = next(e.uid for e in rnd.roster if e.hotkey == "hk_a")
        uid_b = next(e.uid for e in rnd.roster if e.hotkey == "hk_b")
        assert rnd.foreground_uids == (uid_a,)
        assert rnd.background_uids == (uid_b,)
        assert rnd.assigned_uids == (uid_a,)


# ---------------------------------------------------------------------------
# Background queue prepend: top-N prior-avg-scored miners come first so
# `bg-download` / `bg-eval` re-check the current leaders before falling
# through to the staleness rotation.
# ---------------------------------------------------------------------------

class TestBackgroundPriorScorePrepend:
    def test_top_five_prior_scored_come_first(self) -> None:
        # Eight background miners; five of them have positive prior-avg
        # scores. The top-5 by prior avg should lead the queue (in score-
        # desc order), then the rest by staleness.
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({
            f"hk_{i}": 0.5 - 0.01 * i for i in range(8)
        })
        # No miner is mine — push everyone to background.
        assignment = {"other_validator": [f"hk_{i}" for i in range(8)]}
        prior = {
            0: 0.10, 1: 2.50, 2: 1.00, 3: 0.00, 4: 0.50,
            5: 1.50, 6: 0.00, 7: 3.00,
        }
        # Stagger last_evaluated so the tail order is observable.
        ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        last_eval = {i: ts.replace(minute=i) for i in range(8)}

        rnd = _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            prior_avg_scores=prior, last_evaluated=last_eval,
        )

        # uid 7 (3.00), uid 1 (2.50), uid 5 (1.50), uid 2 (1.00), uid 4 (0.50).
        assert rnd.background_uids[:5] == (7, 1, 5, 2, 4)
        # Tail: uids 0, 3, 6 — sorted by staleness (minute asc).
        assert rnd.background_uids[5:] == (0, 3, 6)

    def test_prepend_capped_by_class_constant(self) -> None:
        # Even with seven positively-scored miners, only the top
        # `BG_TOP_SCORED_PREPEND_COUNT` come first. The rest fall back
        # into the staleness-sorted tail.
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({f"hk_{i}": 0.5 - 0.01 * i for i in range(7)})
        assignment = {"other_validator": [f"hk_{i}" for i in range(7)]}
        # All seven have positive scores, descending.
        prior = {i: 1.0 + (7 - i) for i in range(7)}
        ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        last_eval = {i: ts.replace(minute=i) for i in range(7)}

        rnd = _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            prior_avg_scores=prior, last_evaluated=last_eval,
        )

        assert Round.BG_TOP_SCORED_PREPEND_COUNT == 5
        # uid 0 (8.0), uid 1 (7.0), uid 2 (6.0), uid 3 (5.0), uid 4 (4.0).
        assert rnd.background_uids[:5] == (0, 1, 2, 3, 4)
        # Remaining two (uids 5, 6) fall through to the staleness tail.
        assert rnd.background_uids[5:] == (5, 6)

    def test_zero_prior_score_excluded_from_prepend(self) -> None:
        # Miners with prior avg <= 0 are NOT prepended — they fall into
        # the staleness tail. This means a brand-new validator (empty
        # aggregator) sees no prepend at all and the queue is identical
        # to the staleness-sorted version.
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({f"hk_{i}": 0.5 - 0.01 * i for i in range(4)})
        assignment = {"other_validator": [f"hk_{i}" for i in range(4)]}
        ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        last_eval = {0: ts.replace(minute=3), 1: ts, 2: ts.replace(minute=1), 3: ts.replace(minute=2)}

        rnd = _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            prior_avg_scores={i: 0.0 for i in range(4)},  # all zeros
            last_evaluated=last_eval,
        )

        # Pure staleness order: uid 1 (oldest), then 2, 3, 0.
        assert rnd.background_uids == (1, 2, 3, 0)

    def test_foreground_unaffected_by_prepend(self) -> None:
        # Foreground stays in incentive order. Prior avg scores must not
        # leak into the foreground ordering.
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5})
        assignment = {"vhk": ["hk_a", "hk_b", "hk_c"]}
        # Make hk_a the highest-scored historically — it must still
        # come *after* hk_b in foreground because foreground uses
        # incentive order.
        rnd = _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            prior_avg_scores={0: 99.0, 1: 0.0, 2: 0.0},
        )
        # Incentive desc: hk_b (uid 1), hk_c (uid 2), hk_a (uid 0).
        assert rnd.foreground_uids == (1, 2, 0)


# ---------------------------------------------------------------------------
# (3) Snapshot isolation — mutating global_model after freeze must not
# leak into round.model_snapshot_cpu
# ---------------------------------------------------------------------------

class TestSnapshotIsolation:
    def test_mutating_global_model_after_freeze_does_not_change_snapshot(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4})
        assignment = {"vhk": ["hk_a", "hk_b"]}

        global_model = _make_model(val=0.1)
        rnd = _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            global_model=global_model,
        )

        # Mutate the live model.
        with torch.no_grad():
            for p in global_model.parameters():
                p.fill_(99.0)

        for k, v in rnd.model_snapshot_cpu.items():
            assert torch.equal(v, torch.full_like(v, 0.1)), f"snapshot for {k} drifted"


# ---------------------------------------------------------------------------
# (1) Background queue ordering and dedup
# ---------------------------------------------------------------------------

class TestBackgroundQueue:
    def test_next_for_download_yields_foreground_first_then_background(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5, "hk_d": 0.3})
        # Only hk_b is mine; the rest belong to other validators (background).
        assignment = {"vhk": ["hk_b"], "other_validator": ["hk_a", "hk_c", "hk_d"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        order = [e.hotkey for e in rnd.next_for_download()]
        # Foreground (hk_b) first, then background in per-(validator, round)
        # shuffled order (PR #55).
        assert order[: len(rnd.foreground_uids)] == ["hk_b"]
        assert set(order[len(rnd.foreground_uids):]) == {"hk_a", "hk_c", "hk_d"}

    def test_foreground_claim_removes_uid_from_download_queue(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5})
        # hk_b is mine; hk_a and hk_c are someone else's so they go background.
        assignment = {"vhk": ["hk_b"], "other_validator": ["hk_a", "hk_c"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        # Roster covers foreground (hk_b) + background (hk_a, hk_c).
        assert {e.hotkey for e in rnd.next_for_download()} == {"hk_a", "hk_b", "hk_c"}

        # Claiming a UID via foreground removes it from the download queue.
        uid_c = next(e.uid for e in rnd.roster if e.hotkey == "hk_c")
        assert rnd.claim_for_foreground(uid_c) is True
        assert {e.hotkey for e in rnd.next_for_download()} == {"hk_a", "hk_b"}

    def test_publish_download_then_pop_round_trip(self, tmp_path: Path) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.6})
        # hk_a is mine; hk_b belongs to another validator (background).
        assignment = {"vhk": ["hk_a"], "other_validator": ["hk_b"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        bg_uid = rnd.background_uids[0]
        fake_path = tmp_path / "ckpt.pt"
        fake_path.write_bytes(b"x")

        assert rnd.publish_download(bg_uid, fake_path) is True
        assert rnd.has_downloaded(bg_uid) is True
        # next_for_eval should yield it now.
        assert [e.uid for e in rnd.next_for_eval()] == [bg_uid]
        popped = rnd.pop_downloaded(bg_uid)
        assert popped == fake_path
        # After pop, no longer in pool.
        assert rnd.pop_downloaded(bg_uid) is None


# ---------------------------------------------------------------------------
# Round claim semantics — round-level dedup across pause/resume
# ---------------------------------------------------------------------------

class TestRoundClaims:
    def test_claim_for_eval_then_mark_scored_excludes_uid(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5})
        assignment = {"vhk": ["hk_a"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)
        uid = rnd.foreground_uids[0]

        assert rnd.claim_for_foreground(uid) is True
        # Re-claim must fail until released or scored.
        assert rnd.claim_for_foreground(uid) is False
        rnd.mark_scored(uid)
        assert uid in rnd.scored_uids
        # Even after re-attempting a claim post-scoring, claim returns False.
        assert rnd.claim_for_foreground(uid) is False

    def test_unscored_roster_uids(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4})
        assignment = {"vhk": ["hk_a", "hk_b"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        # Score one; the other remains unscored.
        rnd.mark_scored(rnd.roster[0].uid)
        unscored = rnd.unscored_roster_uids()
        assert {e.uid for e in unscored} == {rnd.roster[1].uid}

    def test_unscored_roster_uids_scoped_to_assigned(self) -> None:
        # Other validators' miners are in the roster (so bg-download can
        # reach them) but must not appear in the missed-submission penalty
        # pass — that's only for miners *this* validator is responsible for.
        config = _fake_validator_config(my_hotkey="vhk")
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4})
        assignment = {"vhk": ["hk_a"], "other_validator": ["hk_b"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        unscored = rnd.unscored_roster_uids()
        assert {e.hotkey for e in unscored} == {"hk_a"}


# ---------------------------------------------------------------------------
# Aggregator: schema v1 + v2 round-trip, atomic persist, drop_round
# ---------------------------------------------------------------------------

class TestAggregatorSchema:
    def test_v2_roundtrip_preserves_round_id(self) -> None:
        agg = MinerScoreAggregator(max_points=8)
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
        agg.add_score(uid=1, hotkey="hk1", score=0.5, ts=ts, round_id=42)
        agg.add_score(uid=1, hotkey="hk1", score=0.7, ts=ts.replace(minute=1), round_id=43)

        encoded = agg.to_json()
        payload = json.loads(encoded)
        assert payload["schema_version"] == 2
        assert payload["miners"]["1"]["points"][0][2] == 42

        restored = MinerScoreAggregator.from_json(encoded, max_points=8)
        # The restored aggregator should contain both points with their ids.
        history_v2 = restored._miners[1].series.points  # internal access OK in test
        assert [p[2] for p in history_v2] == [42, 43]

    def test_v1_legacy_format_loads_with_none_round_id(self) -> None:
        # Hand-build a v1 payload (no envelope, no round_id).
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
        v1 = {"5": {"hotkey": "hk5", "points": [[ts.isoformat(), 0.42]]}}
        agg = MinerScoreAggregator.from_json(json.dumps(v1), max_points=8)
        pts = agg._miners[5].series.points
        assert len(pts) == 1
        assert pts[0][1] == pytest.approx(0.42)
        assert pts[0][2] is None

    def test_drop_round_removes_only_targeted_round(self) -> None:
        agg = MinerScoreAggregator(max_points=8)
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
        agg.add_score(uid=1, hotkey="hk1", score=0.5, ts=ts, round_id=10)
        agg.add_score(uid=1, hotkey="hk1", score=0.7, ts=ts.replace(minute=1), round_id=11)
        agg.add_score(uid=2, hotkey="hk2", score=0.3, ts=ts.replace(minute=2), round_id=10)

        dropped = agg.drop_round(10)
        assert dropped == 2
        # Round 11 still there for uid=1.
        assert [p[2] for p in agg._miners[1].series.points] == [11]
        assert agg._miners[2].series.points == []

    def test_persist_atomic_writes_full_payload(self, tmp_path: Path) -> None:
        agg = MinerScoreAggregator(max_points=8)
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
        agg.add_score(uid=1, hotkey="hk1", score=0.5, ts=ts, round_id=42)
        target = tmp_path / "score_aggregator.json"
        agg.persist_atomic(target)

        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded["schema_version"] == 2
        assert "1" in loaded["miners"]

        # No leftover .tmp files in the directory.
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_concurrent_add_and_persist_does_not_corrupt(self, tmp_path: Path) -> None:
        agg = MinerScoreAggregator(max_points=64)
        target = tmp_path / "score_aggregator.json"
        ts0 = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)

        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                agg.add_score(uid=1, hotkey="hk1", score=float(i % 5),
                              ts=ts0.replace(microsecond=i % 1_000_000),
                              round_id=i % 7)
                i += 1
                if i > 200:
                    break

        def persister():
            for _ in range(20):
                agg.persist_atomic(target)

        threads = [threading.Thread(target=writer) for _ in range(2)] + [
            threading.Thread(target=persister)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        stop.set()

        # Any persisted snapshot must be valid JSON with a v2 envelope.
        loaded = json.loads(target.read_text())
        assert loaded["schema_version"] == 2
        assert "miners" in loaded


# ---------------------------------------------------------------------------
# RoundRef swap behavior
# ---------------------------------------------------------------------------

class TestRoundRefSwap:
    def test_swap_moves_current_to_previous(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5})
        assignment = {"vhk": ["hk_a"]}

        ref = RoundRef()
        r1 = _freeze_round(config=config, metagraph=metagraph, assignment=assignment, round_id=100)
        ref.swap(new_current=r1)
        assert ref.current is r1
        assert ref.previous is None

        r2 = _freeze_round(config=config, metagraph=metagraph, assignment=assignment, round_id=200)
        ref.swap(new_current=r2)
        assert ref.current is r2
        assert ref.previous is r1


# ---------------------------------------------------------------------------
# (3) BackgroundEvalWorker GPU-lock yielding invariant + round transition
# ---------------------------------------------------------------------------

class TestBackgroundEvalWorker:
    """Construct the worker, drive a round transition, assert the
    gpu_eval_lock is yielded and the snapshot is loaded once per round.
    Heavy ops (evaluate_one_miner, dataloader, evaluate_model) are mocked.
    """

    def _build_worker(self, *, round_ref: RoundRef):
        from connito.validator.background_eval_worker import BackgroundEvalWorker

        # Minimal config surface needed by the worker.
        cfg = SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=5),
            dataloader=SimpleNamespace(world_size=1),
        )

        gpu_lock = threading.Lock()
        merge_active = threading.Event()
        eval_window = threading.Event()
        stop = threading.Event()

        worker = BackgroundEvalWorker(
            config=cfg,
            round_ref=round_ref,
            device=torch.device("cpu"),
            tokenizer=MagicMock(),
            merge_phase_active=merge_active,
            eval_window_active=eval_window,
            gpu_eval_lock=gpu_lock,
            expert_group_assignment={},
            stop_event=stop,
            poll_interval_sec=0.05,
        )
        worker.set_eval_base_model(_make_model())
        return worker, gpu_lock, merge_active, eval_window, stop

    def test_lock_unheld_at_iteration_boundary(self, tmp_path: Path) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5})
        assignment = {"vhk": ["hk_a"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment, round_id=100)

        ref = RoundRef()
        ref.swap(new_current=rnd)
        worker, gpu_lock, merge_active, eval_window, stop = self._build_worker(
            round_ref=ref,
        )
        # Ensure both gates start in the paused state so the worker idles
        # immediately and never enters the eval branch.
        merge_active.set()  # paused

        worker.start()
        try:
            time.sleep(0.3)
            # The worker should not be holding the lock while parked.
            acquired = gpu_lock.acquire(blocking=False)
            assert acquired, "BackgroundEvalWorker held gpu_eval_lock while paused"
            gpu_lock.release()
        finally:
            stop.set()
            merge_active.clear()
            eval_window.set()
            worker.join(timeout=5)
            assert not worker.is_alive(), "Worker did not stop"


# ---------------------------------------------------------------------------
# (4) Penalty pass + delayed weight submission flow at top of cycle
# ---------------------------------------------------------------------------

class TestDelayedSubmission:
    def test_penalty_recorded_under_round_id_for_unscored_uids(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4, "hk_c": 0.3})
        assignment = {"vhk": ["hk_a", "hk_b", "hk_c"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment, round_id=999)

        agg = MinerScoreAggregator(max_points=8)
        # Score one miner during the cycle.
        agg.add_score(uid=rnd.roster[0].uid, hotkey=rnd.roster[0].hotkey,
                      score=0.4, round_id=rnd.round_id)
        rnd.mark_scored(rnd.roster[0].uid)

        # Drive the penalty pass: every unscored roster UID gets 0.0
        # under round.round_id.
        for entry in rnd.unscored_roster_uids():
            agg.add_score(uid=entry.uid, hotkey=entry.hotkey, score=0.0, round_id=rnd.round_id)

        # Both unscored miners should now have a 0.0 entry under round 999.
        for entry in rnd.roster[1:]:
            pts = agg._miners[entry.uid].series.points
            zero_under_round = [p for p in pts if p[1] == 0.0 and p[2] == rnd.round_id]
            assert len(zero_under_round) == 1, (
                f"missed-submission penalty not recorded for {entry.hotkey}"
            )

        rnd.weights_submitted = True
        assert rnd.weights_submitted is True

    def test_weights_submitted_flag_prevents_double_submit(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5})
        assignment = {"vhk": ["hk_a"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        rnd.weights_submitted = True
        # Mirroring run.py top-of-loop: only submit if not weights_submitted.
        # This is just a flag-level test.
        if not rnd.weights_submitted:
            pytest.fail("weights_submitted should already be set")


# ---------------------------------------------------------------------------
# Rank-based scoring: finalize_round_scores writes 3/2/1/0 to the aggregator
# using `round.scores` (delta**1.2) as the ranking signal.
# ---------------------------------------------------------------------------

class TestFinalizeRoundScores:
    @staticmethod
    def _round_with_five_miners(*, valid_checkpoints: bool = False) -> Round:
        """Build a 5-miner round.

        `valid_checkpoints=True` populates chain checkpoints for every
        miner so they are NOT in `freeze_zero_uids` — required when the
        test wants to verify "unreached" / "operational failure" paths,
        which are only meaningful for miners that had a valid commit at
        freeze time.
        """
        config = _fake_validator_config()
        hotkeys = ["hk_a", "hk_b", "hk_c", "hk_d", "hk_e"]
        metagraph = _make_metagraph({
            "hk_a": 0.5, "hk_b": 0.4, "hk_c": 0.3, "hk_d": 0.2, "hk_e": 0.1,
        })
        assignment = {"vhk": hotkeys}
        chain_ckpts = _valid_chain_checkpoints(hotkeys) if valid_checkpoints else None
        return _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            round_id=500, chain_checkpoints_by_hotkey=chain_ckpts,
        )

    @staticmethod
    def _scores_for_uid(agg: MinerScoreAggregator, uid: int, round_id: int) -> list[float]:
        # Returns the list of score values this UID has under `round_id`.
        # `[]` means either: aggregator has no record of the UID at all,
        # or it has records but none tagged with this round.
        state = agg._miners.get(uid)
        if state is None:
            return []
        return [v for _, v, rid in state.series.points if rid == round_id]

    def test_top_three_get_geometric_rank_scores(self) -> None:
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        # mark_scored stores delta**1.2; pick distinct values so ranking
        # is unambiguous.
        rnd.mark_scored(rnd.roster[0].uid, score=0.10)  # rank 4
        rnd.mark_scored(rnd.roster[1].uid, score=0.40)  # rank 1 → 2.25
        rnd.mark_scored(rnd.roster[2].uid, score=0.30)  # rank 2 → 1.5
        rnd.mark_scored(rnd.roster[3].uid, score=0.20)  # rank 3 → 1.0
        rnd.mark_scored(rnd.roster[4].uid, score=0.05)  # rank 5

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        # Rank 1 → 2.25, rank 2 → 1.5, rank 3 → 1.0, rest → 0.0.
        assert self._scores_for_uid(agg, rnd.roster[1].uid, rnd.round_id) == [2.25]
        assert self._scores_for_uid(agg, rnd.roster[2].uid, rnd.round_id) == [1.5]
        assert self._scores_for_uid(agg, rnd.roster[3].uid, rnd.round_id) == [1.0]
        assert self._scores_for_uid(agg, rnd.roster[0].uid, rnd.round_id) == [0.0]
        assert self._scores_for_uid(agg, rnd.roster[4].uid, rnd.round_id) == [0.0]

    def test_tied_scores_get_zero_and_skip_rank_slot(self) -> None:
        # A tied val_loss between two miners is evidence of a duplicated
        # submission, not legitimate parallel improvement. Both tied
        # miners receive score 0; the next non-tied miner becomes rank 1.
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        # Two miners tied at the highest score, then three distinct
        # values below.
        rnd.mark_scored(rnd.roster[0].uid, score=0.50)  # tied → 0
        rnd.mark_scored(rnd.roster[1].uid, score=0.50)  # tied → 0
        rnd.mark_scored(rnd.roster[2].uid, score=0.40)  # rank 1 → 2.25
        rnd.mark_scored(rnd.roster[3].uid, score=0.30)  # rank 2 → 1.5
        rnd.mark_scored(rnd.roster[4].uid, score=0.20)  # rank 3 → 1.0

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        assert self._scores_for_uid(agg, rnd.roster[0].uid, rnd.round_id) == [0.0]
        assert self._scores_for_uid(agg, rnd.roster[1].uid, rnd.round_id) == [0.0]
        assert self._scores_for_uid(agg, rnd.roster[2].uid, rnd.round_id) == [2.25]
        assert self._scores_for_uid(agg, rnd.roster[3].uid, rnd.round_id) == [1.5]
        assert self._scores_for_uid(agg, rnd.roster[4].uid, rnd.round_id) == [1.0]

    def test_three_way_tie_all_get_zero(self) -> None:
        # All three would have placed top-3 by score; instead all three
        # get 0 because they share the same val_loss.
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        rnd.mark_scored(rnd.roster[0].uid, score=0.40)  # tied
        rnd.mark_scored(rnd.roster[1].uid, score=0.40)  # tied
        rnd.mark_scored(rnd.roster[2].uid, score=0.40)  # tied
        rnd.mark_scored(rnd.roster[3].uid, score=0.20)  # rank 1 → 2.25

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        for tied_idx in (0, 1, 2):
            assert self._scores_for_uid(
                agg, rnd.roster[tied_idx].uid, rnd.round_id,
            ) == [0.0]
        assert self._scores_for_uid(agg, rnd.roster[3].uid, rnd.round_id) == [2.25]

    def test_zero_delta_excluded_from_top_three(self) -> None:
        # If only one miner has delta > 0, only that miner gets the
        # rank-1 reward (2.25). The other "scored but delta==0" miners
        # receive 0.0 — they did not actually improve over baseline so
        # they should not collect reward weight just for showing up.
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        rnd.mark_scored(rnd.roster[0].uid, score=0.0)
        rnd.mark_scored(rnd.roster[1].uid, score=0.5)
        rnd.mark_scored(rnd.roster[2].uid, score=0.0)
        rnd.mark_scored(rnd.roster[3].uid, score=0.0)

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        assert self._scores_for_uid(agg, rnd.roster[1].uid, rnd.round_id) == [2.25]
        for entry in (rnd.roster[0], rnd.roster[2], rnd.roster[3]):
            assert self._scores_for_uid(agg, entry.uid, rnd.round_id) == [0.0]

    def test_validation_failed_uids_get_zero(self) -> None:
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        rnd.mark_scored(rnd.roster[0].uid, score=0.50)
        rnd.mark_validation_failed(rnd.roster[1].uid)
        rnd.mark_validation_failed(rnd.roster[2].uid)

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        assert self._scores_for_uid(agg, rnd.roster[0].uid, rnd.round_id) == [2.25]
        assert self._scores_for_uid(agg, rnd.roster[1].uid, rnd.round_id) == [0.0]
        assert self._scores_for_uid(agg, rnd.roster[2].uid, rnd.round_id) == [0.0]

    def test_operational_failure_preserves_prior_ema(self) -> None:
        # `mark_failed` (without `mark_validation_failed`) marks the UID
        # as failed for operational reasons — download timeout, eval
        # timeout, OOM, unexpected exception. Finalize must NOT write a
        # score=0 for these so the miner's prior EMA is preserved.
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners(valid_checkpoints=True)
        op_uid = rnd.roster[1].uid
        op_hotkey = rnd.roster[1].hotkey

        rnd.mark_scored(rnd.roster[0].uid, score=0.50)
        rnd.mark_failed(op_uid)  # operational failure (timeout etc.)

        agg = MinerScoreAggregator(max_points=8)
        # Preload a positive history for the operational-fail UID so we
        # can prove finalize did not overwrite it with a 0.
        prior_ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        agg.add_score(
            uid=op_uid, hotkey=op_hotkey, score=1.5, ts=prior_ts, round_id=42,
        )

        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        # No new entry under THIS round's id for the operational-fail UID.
        assert self._scores_for_uid(agg, op_uid, rnd.round_id) == []
        # Prior history untouched.
        all_pts = agg._miners[op_uid].series.points
        assert any(p[1] == 1.5 and p[2] == 42 for p in all_pts)

    def test_unreached_miners_get_no_entry(self) -> None:
        # Miners we never reached — submission never landed on disk, or
        # bg-eval ran out of time before claiming them — are absent from
        # `scored_uids`, `validation_failed_uids`, and `freeze_zero_uids`
        # (because they had a valid checkpoint at freeze). They must
        # receive no aggregator entry so their prior EMA is preserved.
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners(valid_checkpoints=True)
        unreached_uid = rnd.roster[2].uid

        rnd.mark_scored(rnd.roster[0].uid, score=0.50)
        rnd.mark_scored(rnd.roster[1].uid, score=0.30)
        # roster[2] is intentionally not touched.

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        assert self._scores_for_uid(agg, unreached_uid, rnd.round_id) == []

    def test_freeze_zero_uids_get_zero_at_finalize(self) -> None:
        # Miners absent from `miners_with_checkpoint` at freeze time end up
        # in `freeze_zero_uids` and should still receive a 0.0 entry from
        # finalize.
        from connito.validator.evaluator import finalize_round_scores

        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5, "hk_b": 0.4, "hk_c": 0.3})
        assignment = {"vhk": ["hk_a", "hk_b", "hk_c"]}
        # hk_b has no checkpoint this round.
        rnd = _freeze_round(
            config=config, metagraph=metagraph, assignment=assignment,
            miners_with_checkpoint=["hk_a", "hk_c"],
            round_id=600,
        )
        uid_b = metagraph.hotkeys.index("hk_b")
        assert uid_b in rnd.freeze_zero_uids

        rnd.mark_scored(metagraph.hotkeys.index("hk_a"), score=0.20)
        rnd.mark_scored(metagraph.hotkeys.index("hk_c"), score=0.30)

        agg = MinerScoreAggregator(max_points=8)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        assert self._scores_for_uid(agg, uid_b, rnd.round_id) == [0.0]

    def test_drop_round_clears_stale_entries_before_rank_writes(self) -> None:
        # Defensive: if the aggregator already holds entries tagged with
        # this round_id (e.g. from a partial run that crashed and was
        # restored from disk), finalize must drop them before writing the
        # rank-based entries — otherwise the avg would mix stale signal
        # with rank.
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        rnd.mark_scored(rnd.roster[0].uid, score=0.40)
        rnd.mark_scored(rnd.roster[1].uid, score=0.30)

        agg = MinerScoreAggregator(max_points=8)
        # Pre-load a stale partial entry tagged with this round_id.
        agg.add_score(uid=rnd.roster[0].uid, hotkey=rnd.roster[0].hotkey,
                      score=42.0, round_id=rnd.round_id)
        finalize_round_scores(round_obj=rnd, score_aggregator=agg)

        assert self._scores_for_uid(agg, rnd.roster[0].uid, rnd.round_id) == [2.25]
        assert self._scores_for_uid(agg, rnd.roster[1].uid, rnd.round_id) == [1.5]

    def test_persists_score_path_when_provided(self, tmp_path: Path) -> None:
        from connito.validator.evaluator import finalize_round_scores

        rnd = self._round_with_five_miners()
        rnd.mark_scored(rnd.roster[0].uid, score=0.40)
        rnd.mark_scored(rnd.roster[1].uid, score=0.30)

        agg = MinerScoreAggregator(max_points=8)
        score_path = tmp_path / "score_aggregator.json"
        finalize_round_scores(
            round_obj=rnd, score_aggregator=agg, score_path=score_path,
        )

        assert score_path.exists()
        restored = MinerScoreAggregator.from_json(
            score_path.read_text(encoding="utf-8"), max_points=8,
        )
        assert TestFinalizeRoundScores._scores_for_uid(
            restored, rnd.roster[0].uid, rnd.round_id,
        ) == [2.25]

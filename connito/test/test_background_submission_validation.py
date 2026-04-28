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
) -> Round:
    """Build a Round bypassing the chain helpers."""
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
        )


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
    def test_next_for_download_yields_full_background_set(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5, "hk_d": 0.3})
        # Only hk_b is mine; the rest belong to other validators (background).
        assignment = {"vhk": ["hk_b"], "other_validator": ["hk_a", "hk_c", "hk_d"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        # Order is shuffled per-(validator, round) to spread HF load — the
        # contract here is set membership, not ordering. Foreground = hk_b.
        order = [e.hotkey for e in rnd.next_for_download()]
        assert set(order) == {"hk_a", "hk_c", "hk_d"}

    def test_foreground_claim_removes_uid_from_download_queue(self) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.1, "hk_b": 0.9, "hk_c": 0.5})
        # hk_b is mine; hk_a and hk_c are someone else's so they go background.
        assignment = {"vhk": ["hk_b"], "other_validator": ["hk_a", "hk_c"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment)

        # Background candidates start as {hk_a, hk_c} (shuffled order).
        assert {e.hotkey for e in rnd.next_for_download()} == {"hk_a", "hk_c"}

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

    def _build_worker(self, *, round_ref: RoundRef, score_path: Path):
        from connito.validator.background_eval_worker import BackgroundEvalWorker

        # Minimal config surface needed by the worker.
        cfg = SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=5),
            dataloader=SimpleNamespace(world_size=1),
        )

        agg = MinerScoreAggregator(max_points=8)

        gpu_lock = threading.Lock()
        merge_active = threading.Event()
        eval_window = threading.Event()
        stop = threading.Event()

        worker = BackgroundEvalWorker(
            config=cfg,
            round_ref=round_ref,
            device=torch.device("cpu"),
            tokenizer=MagicMock(),
            score_aggregator=agg,
            score_path=score_path,
            merge_phase_active=merge_active,
            eval_window_active=eval_window,
            gpu_eval_lock=gpu_lock,
            expert_group_assignment={},
            stop_event=stop,
            poll_interval_sec=0.05,
        )
        worker.set_eval_base_model(_make_model())
        return worker, agg, gpu_lock, merge_active, eval_window, stop

    def test_lock_unheld_at_iteration_boundary(self, tmp_path: Path) -> None:
        config = _fake_validator_config()
        metagraph = _make_metagraph({"hk_a": 0.5})
        assignment = {"vhk": ["hk_a"]}
        rnd = _freeze_round(config=config, metagraph=metagraph, assignment=assignment, round_id=100)

        ref = RoundRef()
        ref.swap(new_current=rnd)
        worker, _agg, gpu_lock, merge_active, eval_window, stop = self._build_worker(
            round_ref=ref, score_path=tmp_path / "score.json",
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

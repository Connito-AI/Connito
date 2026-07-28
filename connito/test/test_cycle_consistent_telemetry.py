"""Tests for the cycle-consistent miner-telemetry contract.

Covers:
  - `set_miner_evaluated_commit` eviction invariant (≤ 1 labelset per uid,
    restart-simulation safe);
  - RoundJournal v2 round-trip + v1 backward compatibility;
  - `finalize_round_scores` emitting last_scored_round_id / score snapshots
    for every verdict uid, round deltas for scored uids, and exactly one
    evaluated-commit labelset per checkpointed uid — both on a live-shaped
    round and again via the journal-recovery path;
  - `evict_round_series_before` removing tracked per-round labelsets.

The prometheus registry is process-global, so every test uses uids in a
dedicated 9xxx range to avoid colliding with other test modules.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from connito.shared import telemetry as T
from connito.validator import round_journal as RJ
from connito.validator.evaluator import finalize_round_scores


def _samples_for(gauge, family_name: str) -> list:
    for metric in gauge.collect():
        return [s for s in metric.samples if s.name == family_name]
    return []


def _labelsets_for_uid(uid: int) -> list:
    return [
        s
        for s in _samples_for(
            T.VALIDATOR_MINER_EVALUATED_COMMIT_INFO,
            "validator_miner_evaluated_commit_info",
        )
        if s.labels["miner_uid"] == str(uid)
    ]


# ---------------------------------------------------------------------------
# Commit-info eviction invariant
# ---------------------------------------------------------------------------

def test_evaluated_commit_evicts_previous_labelset():
    uid = 9001
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-a", 100)
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-b", 200)
    rows = _labelsets_for_uid(uid)
    assert len(rows) == 1
    assert rows[0].labels["hf_revision"] == "rev-b"
    assert rows[0].value == 200.0


def test_evaluated_commit_same_labels_updates_value_in_place():
    uid = 9002
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-x", 100)
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-x", 300)
    rows = _labelsets_for_uid(uid)
    assert len(rows) == 1
    assert rows[0].value == 300.0


def test_evaluated_commit_restart_simulation_is_safe():
    uid = 9003
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-a", 100)
    # Simulate a restart of the tracking dict only (registry keeps series in
    # a real restart neither survives; here the tracking dict being empty
    # while the registry still has the old row exercises the KeyError-free
    # first write, and the follow-up change exercises eviction rebuilt from
    # scratch).
    with T._COMMIT_INFO_LOCK:
        T._COMMIT_INFO_LABELS.clear()
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-b", 200)
    # Old row may linger (tracking was lost) but the invariant re-establishes
    # on the NEXT change:
    T.set_miner_evaluated_commit(uid, "acme/repo", "rev-c", 300)
    rows = _labelsets_for_uid(uid)
    revs = {r.labels["hf_revision"] for r in rows}
    assert "rev-c" in revs
    assert "rev-b" not in revs  # evicted by the rev-b -> rev-c change


def test_evaluated_commit_skips_empty_repo_or_revision():
    uid = 9004
    T.set_miner_evaluated_commit(uid, "", "rev-a", 100)
    T.set_miner_evaluated_commit(uid, "acme/repo", None, 100)  # type: ignore[arg-type]
    assert _labelsets_for_uid(uid) == []


# ---------------------------------------------------------------------------
# Journal v2 / v1 compatibility
# ---------------------------------------------------------------------------

def test_journal_v2_roundtrip(tmp_path):
    j = RJ.RoundJournal(
        round_id=123,
        uid_to_hotkey={1: "hk1"},
        scores={1: 2.5},
        scored_uids=(1,),
        uid_to_commit={1: ("acme/repo", "deadbeef")},
        finalized=True,
    )
    p = tmp_path / "round_123.json"
    RJ.write_atomic(p, j)
    loaded = RJ.load(p)
    assert loaded is not None
    assert loaded.schema_version == RJ.SCHEMA_VERSION
    assert loaded.uid_to_commit == {1: ("acme/repo", "deadbeef")}
    assert loaded.scores == {1: 2.5}


def test_journal_v1_loads_with_empty_commit_map(tmp_path):
    v1 = {
        "round_id": 77,
        "uid_to_hotkey": {"5": "hk5"},
        "scores": {"5": 1.25},
        "scored_uids": [5],
        "failed_uids": [],
        "validation_failed_uids": [],
        "freeze_zero_uids": [],
        "freeze_zero_hotkeys": {},
        "finalized": False,
        "schema_version": 1,
    }
    p = tmp_path / "round_77.json"
    p.write_text(json.dumps(v1), encoding="utf-8")
    loaded = RJ.load(p)
    assert loaded is not None
    assert loaded.uid_to_commit == {}
    assert loaded.scores == {5: 1.25}


def test_journal_future_version_rejected(tmp_path):
    p = tmp_path / "round_88.json"
    p.write_text(
        json.dumps({"round_id": 88, "schema_version": RJ.SCHEMA_VERSION + 1}),
        encoding="utf-8",
    )
    try:
        RJ.load(p)
    except ValueError:
        return
    raise AssertionError("future schema_version must raise")


def test_commit_map_from_checkpoints_skips_incomplete():
    m = RJ.commit_map_from_checkpoints({
        1: SimpleNamespace(hf_repo_id="a/r", hf_revision="v1"),
        2: SimpleNamespace(hf_repo_id=None, hf_revision="v2"),
        3: SimpleNamespace(hf_repo_id="a/r3", hf_revision=""),
    })
    assert m == {1: ("a/r", "v1")}


# ---------------------------------------------------------------------------
# finalize_round_scores emission (live-shaped round + recovery path)
# ---------------------------------------------------------------------------


class _FakeAggregator:
    def __init__(self):
        self.rows: dict[int, list[float]] = {}

    def drop_round(self, round_id):
        pass

    def add_score(self, *, uid, hotkey, score, round_id):
        self.rows.setdefault(int(uid), []).append(float(score))

    def uid_score_pairs(self, how="avg"):
        if how == "latest":
            return {u: v[-1] for u, v in self.rows.items()}
        return {u: sum(v) / len(v) for u, v in self.rows.items()}

    def record_count(self, uid):
        return len(self.rows.get(int(uid), []))


def _make_round(round_id: int, journal_path=None):
    return RJ._RecoveryRound(
        round_id=round_id,
        scores={9101: 2.5, 9102: 1.5, 9103: 0.0},
        scored_uids={9101, 9102, 9103},
        failed_uids=set(),
        validation_failed_uids={9104},
        freeze_zero_uids={9105},
        freeze_zero_hotkeys={9105: "hk9105"},
        uid_to_hotkey={u: f"hk{u}" for u in (9101, 9102, 9103, 9104)},
        journal_path=journal_path,
        uid_to_chain_checkpoint={
            9101: SimpleNamespace(hf_repo_id="miner/one", hf_revision="aaa1"),
            9102: SimpleNamespace(hf_repo_id="miner/two", hf_revision="bbb2"),
        },
    )


def _gauge_value(gauge, family_name, uid):
    for s in _samples_for(gauge, family_name):
        if s.labels.get("miner_uid") == str(uid):
            return s.value
    return None


def test_finalize_emits_cycle_consistent_series(tmp_path):
    rid = 524_000
    round_obj = _make_round(rid, journal_path=tmp_path / f"round_{rid}.json")
    written = finalize_round_scores(
        round_obj=round_obj, score_aggregator=_FakeAggregator(), score_path=None,
    )
    # Verdict uids: 3 scored + 1 validation_failed + 1 freeze_zero.
    assert set(written) == {9101, 9102, 9103, 9104, 9105}
    for uid in written:
        assert _gauge_value(
            T.VALIDATOR_MINER_LAST_SCORED_ROUND_ID,
            "validator_miner_last_scored_round_id", uid,
        ) == float(rid)
        assert _gauge_value(
            T.VALIDATOR_MINER_SCORE_LATEST, "validator_miner_score_latest", uid,
        ) is not None
    # Deltas only for evaluated uids, raw values preserved.
    assert _gauge_value(
        T.VALIDATOR_MINER_ROUND_DELTA, "validator_miner_round_delta", 9101
    ) == 2.5
    assert _gauge_value(
        T.VALIDATOR_MINER_ROUND_DELTA, "validator_miner_round_delta", 9103
    ) == 0.0
    assert _gauge_value(
        T.VALIDATOR_MINER_ROUND_DELTA, "validator_miner_round_delta", 9105
    ) is None
    # Exactly one commit labelset per checkpointed uid.
    assert len(_labelsets_for_uid(9101)) == 1
    assert _labelsets_for_uid(9101)[0].value == float(rid)
    assert len(_labelsets_for_uid(9102)) == 1


def test_finalize_recovery_path_reemits_commits(tmp_path):
    rid = 524_524
    # Write a v2 journal, hydrate a recovery round from it, finalize.
    live = _make_round(rid, journal_path=tmp_path / f"round_{rid}.json")
    RJ.write_atomic(
        live.journal_path,
        RJ.RoundJournal(
            round_id=rid,
            uid_to_hotkey=dict(live.uid_to_hotkey),
            scores=dict(live.scores),
            scored_uids=tuple(sorted(live.scored_uids)),
            validation_failed_uids=tuple(sorted(live.validation_failed_uids)),
            freeze_zero_uids=tuple(sorted(live.freeze_zero_uids)),
            freeze_zero_hotkeys=dict(live.freeze_zero_hotkeys),
            uid_to_commit=RJ.commit_map_from_checkpoints(live.uid_to_chain_checkpoint),
        ),
    )
    journal = RJ.load(live.journal_path)
    assert journal is not None
    recovered = RJ._RecoveryRound.from_journal(journal, live.journal_path)
    finalize_round_scores(
        round_obj=recovered, score_aggregator=_FakeAggregator(), score_path=None,
    )
    rows = _labelsets_for_uid(9101)
    assert len(rows) == 1
    assert rows[0].value == float(rid)
    assert _gauge_value(
        T.VALIDATOR_MINER_LAST_SCORED_ROUND_ID,
        "validator_miner_last_scored_round_id", 9105,
    ) == float(rid)


# ---------------------------------------------------------------------------
# Per-round baseline loss
# ---------------------------------------------------------------------------

def test_set_baseline_loss_sets_both_gauges():
    rid = 920_100
    T.set_baseline_loss(rid, 1.8342)
    # Unlabeled gauge carries the latest value (backward compat).
    unlabeled = _samples_for(T.VALIDATOR_BASELINE_LOSS, "validator_baseline_loss")
    assert unlabeled and unlabeled[0].value == 1.8342
    # Labeled family carries the same value under the round's id.
    labeled = [
        s
        for s in _samples_for(
            T.VALIDATOR_BASELINE_LOSS_BY_ROUND, "validator_baseline_loss_by_round"
        )
        if s.labels["round_id"] == str(rid)
    ]
    assert len(labeled) == 1
    assert labeled[0].value == 1.8342
    # round_id registered for eviction.
    assert rid in T._EMITTED_ROUND_IDS


def test_set_baseline_loss_labeled_is_stable_per_round():
    # A second round's baseline does not disturb the first round's labeled
    # value — the whole point of the per-round label vs the overwritten
    # unlabeled gauge.
    T.set_baseline_loss(920_200, 2.0)
    T.set_baseline_loss(920_724, 3.0)
    by_round = {
        s.labels["round_id"]: s.value
        for s in _samples_for(
            T.VALIDATOR_BASELINE_LOSS_BY_ROUND, "validator_baseline_loss_by_round"
        )
    }
    assert by_round["920200"] == 2.0
    assert by_round["920724"] == 3.0


# ---------------------------------------------------------------------------
# Per-round series eviction
# ---------------------------------------------------------------------------

def test_evict_round_series_before():
    old_rid, new_rid = 910_000, 910_524
    for rid in (old_rid, new_rid):
        T.note_round_series(rid)
        T.VALIDATOR_ROUND_MINERS_SCORED.labels(round_id=str(rid)).set(5)
        T.VALIDATOR_BASELINE_LOSS_BY_ROUND.labels(round_id=str(rid)).set(1.8)
    removed = T.evict_round_series_before(new_rid)
    assert removed >= 1
    rids = {
        s.labels["round_id"]
        for s in _samples_for(
            T.VALIDATOR_ROUND_MINERS_SCORED, "validator_round_miners_scored"
        )
    }
    assert str(old_rid) not in rids
    assert str(new_rid) in rids
    # The baseline-by-round family is evicted on the same cutoff.
    baseline_rids = {
        s.labels["round_id"]
        for s in _samples_for(
            T.VALIDATOR_BASELINE_LOSS_BY_ROUND, "validator_baseline_loss_by_round"
        )
    }
    assert str(old_rid) not in baseline_rids
    assert str(new_rid) in baseline_rids
    # Idempotent / KeyError-safe on repeat.
    assert T.evict_round_series_before(new_rid) == 0

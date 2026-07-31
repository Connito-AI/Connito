"""Tests for the metrics-only telemetry republish on startup recovery.

Startup recovery restores telemetry by *replaying* `finalize_round_scores`,
which marks the journal finalized — and the replay loop skips finalized
journals. So recovery only ever restored telemetry ONCE: a second restart
found nothing to replay and emitted nothing, leaving the dashboard blank
for the entire last completed cycle (observed 2026-07-31, two Watchtower
restarts 25 minutes apart).

`republish_telemetry_from_journal` closes that. It must be metrics-only:
re-running finalize would keep the aggregator's point *set* correct
(`drop_round` runs first) but re-stamp those points with fresh timestamps,
reshuffling the "last N by timestamp" rolling average that drives weight
submission.

The prometheus registry is process-global, so these tests use uids in a
dedicated 94xx range and round ids in a 94xxxx range.
"""
from __future__ import annotations

import json

from connito.shared import telemetry as T
from connito.validator import round_journal as RJ


def _sample(gauge, family: str, **labels) -> float | None:
    for metric in gauge.collect():
        for s in metric.samples:
            if s.name == family and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return None


def _journal(round_id: int, **overrides) -> RJ.RoundJournal:
    kwargs = dict(
        round_id=round_id,
        uid_to_hotkey={9401: "hk1", 9402: "hk2", 9403: "hk3", 9404: "hk4"},
        scores={9401: 2.5, 9402: 1.5, 9403: 0.0},
        scored_uids=(9401, 9402, 9403),
        failed_uids=(9404,),          # operational failure — no verdict
        validation_failed_uids=(),
        freeze_zero_uids=(9405,),
        freeze_zero_hotkeys={9405: "hk5"},
        uid_to_commit={9401: ("miner/one", "aaa1")},
        uid_to_val_loss={9401: 1.25, 9402: 1.75},
        roster_size=10,
        lifecycle_step=3,
        finalized=True,
    )
    kwargs.update(overrides)
    return RJ.RoundJournal(**kwargs)


class _RecordingAggregator:
    """Read-only probe: records every mutating call so a test can assert
    the republish pass made none."""

    def __init__(self):
        self.mutations: list[str] = []

    # --- read side (what republish is allowed to use) ---
    def uid_score_pairs(self, how="avg"):
        return {9401: 2.25 if how == "latest" else 1.1, 9402: 1.5}

    def record_count(self, uid):
        return 4

    # --- write side (must never be called) ---
    def add_score(self, **kwargs):
        self.mutations.append("add_score")

    def drop_round(self, round_id):
        self.mutations.append("drop_round")

    def persist_atomic(self, path):
        self.mutations.append("persist_atomic")

    def prune_before_round(self, min_round_id):
        self.mutations.append("prune_before_round")


# ---------------------------------------------------------------------------
# The guard the backend specifically asked for
# ---------------------------------------------------------------------------

def test_republish_does_not_mutate_the_aggregator():
    agg = _RecordingAggregator()
    RJ.republish_telemetry_from_journal(_journal(940_001), score_aggregator=agg)
    assert agg.mutations == []


def test_republish_works_without_an_aggregator():
    # Snapshot gauges are skipped, everything else still emits.
    n = RJ.republish_telemetry_from_journal(_journal(940_002), score_aggregator=None)
    assert n == 4  # 3 scored + 1 freeze_zero (the failed uid gets no verdict)


# ---------------------------------------------------------------------------
# What gets emitted
# ---------------------------------------------------------------------------

def test_republish_emits_last_scored_round_for_verdict_uids():
    rid = 940_010
    RJ.republish_telemetry_from_journal(_journal(rid), score_aggregator=_RecordingAggregator())
    for uid in (9401, 9402, 9403, 9405):
        assert _sample(
            T.VALIDATOR_MINER_LAST_SCORED_ROUND_ID,
            "validator_miner_last_scored_round_id",
            miner_uid=str(uid),
        ) == float(rid)


def test_republish_excludes_operational_failures():
    """`failed_uids` minus validation failures get NO aggregator entry at
    finalize, so telemetry must not imply they were judged."""
    j = _journal(940_011, uid_to_hotkey={9410: "hk"}, scored_uids=(), scores={},
                 failed_uids=(9410,), freeze_zero_uids=(), freeze_zero_hotkeys={},
                 uid_to_commit={}, uid_to_val_loss={})
    assert RJ.verdict_uids(j) == set()
    assert RJ.republish_telemetry_from_journal(j) == 0


def test_republish_emits_val_loss():
    rid = 940_020
    RJ.republish_telemetry_from_journal(_journal(rid))
    assert _sample(
        T.VALIDATOR_MINER_VAL_LOSS, "validator_miner_val_loss", miner_uid="9401"
    ) == 1.25
    assert _sample(
        T.VALIDATOR_MINER_VAL_LOSS, "validator_miner_val_loss", miner_uid="9402"
    ) == 1.75


def test_republish_emits_round_counters_and_lifecycle():
    rid = 940_030
    RJ.republish_telemetry_from_journal(_journal(rid))
    label = {"round_id": str(rid)}
    assert _sample(T.VALIDATOR_ROUND_MINERS_SCORED, "validator_round_miners_scored", **label) == 3.0
    assert _sample(T.VALIDATOR_ROUND_MINERS_FAILED, "validator_round_miners_failed", **label) == 1.0
    # roster 10 - 3 scored - 1 failed
    assert _sample(T.VALIDATOR_ROUND_MINERS_PENDING, "validator_round_miners_pending", **label) == 6.0
    assert _sample(
        T.VALIDATOR_ROUND_LIFECYCLE_STEP, "validator_round_lifecycle_step", **label
    ) == 3.0


def test_republish_emits_round_delta_and_commit():
    rid = 940_040
    RJ.republish_telemetry_from_journal(_journal(rid))
    assert _sample(
        T.VALIDATOR_MINER_ROUND_DELTA, "validator_miner_round_delta", miner_uid="9401"
    ) == 2.5
    assert _sample(
        T.VALIDATOR_MINER_EVALUATED_COMMIT_INFO,
        "validator_miner_evaluated_commit_info",
        miner_uid="9401", hf_repo_id="miner/one", hf_revision="aaa1",
    ) == float(rid)


def test_republish_pre_v3_journal_clamps_pending_and_skips_val_loss():
    """A v2 journal has no roster_size and no losses — republish must not
    invent a denominator or crash."""
    rid = 940_050
    j = _journal(rid, roster_size=0, lifecycle_step=0, uid_to_val_loss={})
    assert RJ.republish_telemetry_from_journal(j) == 4
    assert _sample(
        T.VALIDATOR_ROUND_MINERS_PENDING, "validator_round_miners_pending",
        round_id=str(rid),
    ) == 0.0


def test_republish_is_idempotent():
    rid = 940_060
    a = RJ.republish_telemetry_from_journal(_journal(rid))
    b = RJ.republish_telemetry_from_journal(_journal(rid))
    assert a == b
    assert _sample(
        T.VALIDATOR_ROUND_MINERS_SCORED, "validator_round_miners_scored",
        round_id=str(rid),
    ) == 3.0


def test_republish_registers_round_for_eviction():
    rid = 940_070
    RJ.republish_telemetry_from_journal(_journal(rid))
    assert rid in T._EMITTED_ROUND_IDS


def test_republish_never_raises_on_malformed_journal():
    class Broken:
        round_id = "not-an-int"
    assert RJ.republish_telemetry_from_journal(Broken()) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# val_loss journalling (Gap 1)
# ---------------------------------------------------------------------------

def test_journal_v3_roundtrips_val_loss(tmp_path):
    j = _journal(940_080)
    p = tmp_path / "round.json"
    RJ.write_atomic(p, j)
    loaded = RJ.load(p)
    assert loaded is not None
    assert loaded.uid_to_val_loss == {9401: 1.25, 9402: 1.75}


def test_journal_v2_file_loads_with_empty_val_loss(tmp_path):
    v2 = {"round_id": 42, "schema_version": 2, "scored_uids": [1], "finalized": True}
    p = tmp_path / "round_42.json"
    p.write_text(json.dumps(v2), encoding="utf-8")
    loaded = RJ.load(p)
    assert loaded is not None
    assert loaded.uid_to_val_loss == {}
    assert loaded.roster_size == 0


def test_recovery_round_carries_val_losses(tmp_path):
    stub = RJ._RecoveryRound.from_journal(_journal(940_090), tmp_path / "r.json")
    assert stub.val_losses == {9401: 1.25, 9402: 1.75}

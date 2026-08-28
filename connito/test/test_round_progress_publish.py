"""Tests for round progress-counter publishing.

`validator_round_miners_{scored,failed,pending}` used to be written only by
`BackgroundEvalWorker._record_metrics`, whose loop starts when the eval
window opens at Merge — so foreground evaluations (which run during
Submission) accumulated in `scored_uids` with nothing publishing them, and
the dashboard's "Evaluated N of M" panel sat on the previous round's final
value for the first ~14 minutes of every round.

The publish now lives in `telemetry.set_round_progress` / `Round.publish_progress`
and is called from freeze, the foreground path, and the background worker.

The prometheus registry is process-global, so these tests use round ids in a
dedicated 93xxxx range to avoid colliding with other test modules.
"""
from __future__ import annotations

import threading

from connito.shared import telemetry as T
from connito.validator.round import Round


def _round_gauge(gauge, family_name: str, round_id: int) -> float | None:
    for metric in gauge.collect():
        for s in metric.samples:
            if s.name == family_name and s.labels.get("round_id") == str(round_id):
                return s.value
    return None


def _progress(round_id: int) -> tuple[float | None, float | None, float | None]:
    return (
        _round_gauge(T.VALIDATOR_ROUND_MINERS_SCORED, "validator_round_miners_scored", round_id),
        _round_gauge(T.VALIDATOR_ROUND_MINERS_FAILED, "validator_round_miners_failed", round_id),
        _round_gauge(T.VALIDATOR_ROUND_MINERS_PENDING, "validator_round_miners_pending", round_id),
    )


def _make_round(round_id: int, *, foreground=(1, 2), background=(3, 4, 5)) -> Round:
    # Built directly rather than via `Round.freeze` (which needs chain +
    # a model); `journal_path` / `score_aggregator` stay None so the
    # `mark_*` helpers skip their persistence side effects.
    return Round(
        round_id=round_id,
        seed="0" * 64,
        validator_miner_assignment={},
        background_uids=tuple([*foreground, *background]),
        uid_to_hotkey={u: f"hk{u}" for u in (*foreground, *background)},
        model_snapshot_cpu={},
    )


# ---------------------------------------------------------------------------
# set_round_progress
# ---------------------------------------------------------------------------

def test_set_round_progress_sets_all_three_gauges():
    rid = 930_001
    T.set_round_progress(rid, scored=7, failed=2, pending=11)
    assert _progress(rid) == (7.0, 2.0, 11.0)


def test_set_round_progress_registers_round_for_eviction():
    rid = 930_002
    T.set_round_progress(rid, scored=1, failed=0, pending=3)
    assert rid in T._EMITTED_ROUND_IDS
    # And the normal cutoff removes the labelsets it just created.
    T.evict_round_series_before(rid + 1)
    assert _progress(rid) == (None, None, None)


def test_set_round_progress_never_raises_on_bad_input():
    # Telemetry must not be able to break scoring.
    T.set_round_progress(None, scored="x", failed=None, pending=object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round.publish_progress
# ---------------------------------------------------------------------------

def test_publish_progress_at_freeze_seeds_zero_scored_full_pending():
    rid = 930_010
    r = _make_round(rid)  # roster = 5
    r.publish_progress()
    assert _progress(rid) == (0.0, 0.0, 5.0)


def test_publish_progress_tracks_foreground_marks():
    """The regression this change exists for: a foreground `mark_scored`
    followed by a publish must move the counter, with no background worker
    involved."""
    rid = 930_011
    r = _make_round(rid)  # roster = 5
    r.publish_progress()
    assert _progress(rid) == (0.0, 0.0, 5.0)

    r.mark_scored(1, 2.5)
    r.publish_progress()
    assert _progress(rid) == (1.0, 0.0, 4.0)

    r.mark_scored(2, 1.5)
    r.publish_progress()
    assert _progress(rid) == (2.0, 0.0, 3.0)


def test_publish_progress_counts_failures_and_validation_failures():
    rid = 930_012
    r = _make_round(rid)  # roster = 5
    r.mark_failed(3)
    r.publish_progress()
    assert _progress(rid) == (0.0, 1.0, 4.0)

    # `mark_validation_failed` lands in `failed_uids` too, so pending drops.
    r.mark_validation_failed(4)
    r.publish_progress()
    assert _progress(rid) == (0.0, 2.0, 3.0)


def test_publish_progress_matches_stats():
    rid = 930_013
    r = _make_round(rid)
    r.mark_scored(1, 1.0)
    r.mark_failed(3)
    r.publish_progress()
    stats = r.stats()
    assert _progress(rid) == (
        float(stats["scored"]), float(stats["failed"]), float(stats["pending"]),
    )


def test_publish_progress_does_not_hold_round_lock():
    """`stats()` acquires `Round._lock`, which is a plain (non-reentrant)
    Lock — so `publish_progress` must be callable without the caller
    holding it, and must leave it released."""
    rid = 930_014
    r = _make_round(rid)
    r.publish_progress()
    assert r._lock.acquire(blocking=False) is True
    r._lock.release()


def test_publish_progress_is_isolated_per_round():
    rid_a, rid_b = 930_020, 930_021
    ra = _make_round(rid_a, foreground=(1,), background=(2,))
    rb = _make_round(rid_b, foreground=(1,), background=(2, 3))
    ra.mark_scored(1, 1.0)
    ra.publish_progress()
    rb.publish_progress()
    assert _progress(rid_a) == (1.0, 0.0, 1.0)
    assert _progress(rid_b) == (0.0, 0.0, 3.0)

"""Integration tests for the bg-eval score-persistence path.

These tests exercise the full chain — `Round.mark_*` writes the journal
to disk + (for `mark_scored`) updates the aggregator with the raw
in-cycle score; `finalize_round_scores` flips the journal to
`finalized=True` and replaces the raw aggregator entries with the
rank-based ones; the startup-recovery pass replays an unfinalized
journal so the aggregator ends up identical to what a clean run would
have produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from connito.validator.aggregator import MinerScoreAggregator
from connito.validator import round_journal as rj
from connito.validator.evaluator import finalize_round_scores
from connito.validator.round import Round


def _make_round(*, round_id: int, score_path: Path, journal_path: Path,
                aggregator: MinerScoreAggregator,
                uid_to_hotkey: dict[int, str],
                freeze_zero_uids: set[int] | None = None,
                freeze_zero_hotkeys: dict[int, str] | None = None) -> Round:
    """Build a `Round` directly without going through `Round.freeze`.

    Used so the tests don't need a metagraph / chain stub. The journal
    + aggregator paths are wired up so `mark_*` exercises the full
    persistence path.
    """
    return Round(
        round_id=round_id,
        seed="test-seed",
        validator_miner_assignment={},
        foreground_uids=tuple(uid_to_hotkey.keys()),
        background_uids=(),
        uid_to_hotkey=dict(uid_to_hotkey),
        model_snapshot_cpu={},
        freeze_zero_uids=set(freeze_zero_uids or set()),
        freeze_zero_hotkeys=dict(freeze_zero_hotkeys or {}),
        journal_path=journal_path,
        score_aggregator=aggregator,
        score_path=score_path,
    )


def test_mark_scored_writes_journal_and_aggregator(tmp_path: Path) -> None:
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 1000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=1000, score_path=score_path, journal_path=journal_path,
        aggregator=agg, uid_to_hotkey={1: "hk1", 2: "hk2"},
    )
    round_obj.mark_scored(1, score=0.018)

    # Journal landed on disk with the score.
    journal = rj.load(journal_path)
    assert journal is not None
    assert journal.scored_uids == (1,)
    assert journal.scores == {1: 0.018}
    assert journal.finalized is False

    # Aggregator on disk has the raw entry tagged with this round_id.
    reloaded_agg = MinerScoreAggregator.from_json(score_path.read_text())
    assert reloaded_agg.record_count(1) == 1
    points = reloaded_agg._miners[1].series.points
    assert any(rid == 1000 and v == pytest.approx(0.018) for _, v, rid in points)


def test_mark_failed_and_validation_failed_journal(tmp_path: Path) -> None:
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 2000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=2000, score_path=score_path, journal_path=journal_path,
        aggregator=agg, uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3"},
    )
    round_obj.mark_failed(1)
    round_obj.mark_validation_failed(2)

    journal = rj.load(journal_path)
    assert journal is not None
    assert set(journal.failed_uids) == {1, 2}
    assert journal.validation_failed_uids == (2,)
    assert journal.finalized is False


def test_finalize_flips_journal_to_finalized_and_replaces_raw_entries(tmp_path: Path) -> None:
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 3000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=3000, score_path=score_path, journal_path=journal_path,
        aggregator=agg, uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3"},
    )
    round_obj.mark_scored(1, score=0.5)
    round_obj.mark_scored(2, score=1.0)
    round_obj.mark_scored(3, score=0.25)

    # Pre-finalize: aggregator has raw deltas (0.5, 1.0, 0.25).
    pre = MinerScoreAggregator.from_json(score_path.read_text())
    assert pre.record_count(1) == 1

    finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, score_path=score_path,
    )

    # Post-finalize: journal flipped to finalized.
    journal = rj.load(journal_path)
    assert journal is not None
    assert journal.finalized is True

    # Aggregator now has rank-based scores: top-1 (uid 2) = 2.25,
    # top-2 (uid 1) = 1.5, top-3 (uid 3) = 1.0. Raw deltas dropped.
    post = MinerScoreAggregator.from_json(score_path.read_text())
    assert post._miners[2].series.points[-1][1] == pytest.approx(2.25)
    assert post._miners[1].series.points[-1][1] == pytest.approx(1.5)
    assert post._miners[3].series.points[-1][1] == pytest.approx(1.0)
    # No leftover raw entries.
    for state in post._miners.values():
        assert len(state.series.points) == 1


def test_kill_before_finalize_recovers_via_startup_pass(tmp_path: Path) -> None:
    """Simulate: round runs, mark_scored 3 miners, validator killed,
    no finalize. On startup the journal is replayed through
    finalize_round_scores. Result: aggregator on disk should be
    identical to a clean (no-kill) run.
    """
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 4000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=4000, score_path=score_path, journal_path=journal_path,
        aggregator=agg, uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3"},
    )
    round_obj.mark_scored(1, score=0.5)
    round_obj.mark_scored(2, score=1.0)
    round_obj.mark_scored(3, score=0.25)
    # No finalize — simulate kill by dropping `round_obj` and the
    # in-memory aggregator. We will reload from disk.
    del round_obj

    # Startup: load aggregator from disk + run recovery against the
    # leftover journal.
    recovered_agg = MinerScoreAggregator.from_json(score_path.read_text())

    journal = rj.load(journal_path)
    assert journal is not None
    assert journal.finalized is False
    stub = rj._RecoveryRound.from_journal(journal, journal_path)
    finalize_round_scores(
        round_obj=stub, score_aggregator=recovered_agg, score_path=score_path,
    )

    # Journal flipped to finalized.
    after = rj.load(journal_path)
    assert after is not None
    assert after.finalized is True

    # Aggregator on disk now has the rank-based scores — same as the
    # no-kill case in the previous test.
    final = MinerScoreAggregator.from_json(score_path.read_text())
    assert final._miners[2].series.points[-1][1] == pytest.approx(2.25)
    assert final._miners[1].series.points[-1][1] == pytest.approx(1.5)
    assert final._miners[3].series.points[-1][1] == pytest.approx(1.0)


def test_journal_persists_after_finalize_for_audit(tmp_path: Path) -> None:
    """The journal file must NOT be unlinked at finalize — only flipped.
    Audit consumers can read it later; only `prune_before_round`
    removes journals (by age)."""
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 5000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=5000, score_path=score_path, journal_path=journal_path,
        aggregator=agg, uid_to_hotkey={1: "hk1"},
    )
    round_obj.mark_scored(1, score=0.5)
    finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, score_path=score_path,
    )
    assert journal_path.exists()
    assert rj.load(journal_path).finalized is True

    # `prune_before_round` with cutoff above this round_id removes it.
    rj.prune_before_round(tmp_path, min_round_id=6000)
    assert not journal_path.exists()


def test_mark_methods_are_no_op_when_round_has_no_journal_path(tmp_path: Path) -> None:
    """Legacy code paths that build `Round` without journal wiring must
    not crash on `mark_*`."""
    round_obj = Round(
        round_id=9999,
        seed="x",
        validator_miner_assignment={},
        foreground_uids=(1,),
        background_uids=(),
        uid_to_hotkey={1: "hk1"},
        model_snapshot_cpu={},
        # journal_path / score_aggregator / score_path all None.
    )
    round_obj.mark_scored(1, score=1.0)
    round_obj.mark_failed(1)
    round_obj.mark_validation_failed(1)
    # No file written, no exceptions raised.
    assert round_obj.scores == {1: 1.0}

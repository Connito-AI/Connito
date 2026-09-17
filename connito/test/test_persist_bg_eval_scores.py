"""Integration tests for the bg-eval score-persistence path.

These tests exercise the full chain — `Round.mark_*` writes the journal to
disk and nothing else; `finalize_round_scores` is the aggregator's sole
writer and flips the journal to `finalized=True`; the startup-recovery pass
replays an unfinalized journal so a killed round ends up with the same
rank-based scores a clean run would have produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from connito.validator.aggregator import MinerScoreAggregator
from connito.validator import round_journal as rj
from connito.validator.evaluator import finalize_round_scores
from connito.validator.round import Round


def _make_round(*, round_id: int, journal_path: Path,
                uid_to_hotkey: dict[int, str],
                freeze_zero_uids: set[int] | None = None,
                freeze_zero_hotkeys: dict[int, str] | None = None) -> Round:
    """Build a `Round` directly without going through `Round.freeze`.

    Used so the tests don't need a metagraph / chain stub. `journal_path` is
    wired up so `mark_*` exercises the full persistence path.
    """
    return Round(
        round_id=round_id,
        seed="test-seed",
        validator_miner_assignment={},
        background_uids=tuple(uid_to_hotkey.keys()),
        uid_to_hotkey=dict(uid_to_hotkey),
        base_shard=Path("pretrained/model_expgroup_1.safetensors"),
        freeze_zero_uids=set(freeze_zero_uids or set()),
        freeze_zero_hotkeys=dict(freeze_zero_hotkeys or {}),
        journal_path=journal_path,
    )


def test_mark_scored_writes_the_journal_and_nothing_else(tmp_path: Path) -> None:
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 1000)

    round_obj = _make_round(
        round_id=1000, journal_path=journal_path,
        uid_to_hotkey={1: "hk1", 2: "hk2"},
    )
    round_obj.mark_scored(1, score=0.018)

    # Journal landed on disk with the score.
    journal = rj.load(journal_path)
    assert journal is not None
    assert journal.scored_uids == (1,)
    assert journal.scores == {1: 0.018}
    assert journal.finalized is False

    # The raw `delta ** 1.2` never reaches the aggregator: only
    # `finalize_round_scores` writes it, and only rank values.
    assert not score_path.exists()


def test_mark_failed_and_validation_failed_journal(tmp_path: Path) -> None:
    journal_path = rj.journal_path_for(tmp_path, 2000)

    round_obj = _make_round(
        round_id=2000, journal_path=journal_path,
        uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3"},
    )
    round_obj.mark_failed(1)
    round_obj.mark_validation_failed(2)

    journal = rj.load(journal_path)
    assert journal is not None
    assert set(journal.failed_uids) == {1, 2}
    assert journal.validation_failed_uids == (2,)
    assert journal.finalized is False


def test_finalize_flips_journal_to_finalized_and_writes_ranks(tmp_path: Path) -> None:
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 3000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=3000, journal_path=journal_path,
        uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3"},
    )
    round_obj.mark_scored(1, score=0.5)
    round_obj.mark_scored(2, score=1.0)
    round_obj.mark_scored(3, score=0.25)

    finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, score_path=score_path,
    )

    # Post-finalize: journal flipped to finalized.
    journal = rj.load(journal_path)
    assert journal is not None
    assert journal.finalized is True

    # Aggregator now has rank-based scores: top-1 (uid 2) = 2.25,
    # top-2 (uid 1) = 1.5, top-3 (uid 3) = 1.0.
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

    round_obj = _make_round(
        round_id=4000, journal_path=journal_path,
        uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3"},
    )
    round_obj.mark_scored(1, score=0.5)
    round_obj.mark_scored(2, score=1.0)
    round_obj.mark_scored(3, score=0.25)
    # No finalize — simulate the kill by dropping the round.
    del round_obj

    # Startup: the round contributed nothing to the aggregator, so recovery
    # starts from an empty one and rebuilds the round from its journal alone.
    assert not score_path.exists()
    recovered_agg = MinerScoreAggregator(max_points=8, max_history_points=64)

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
        round_id=5000, journal_path=journal_path,
        uid_to_hotkey={1: "hk1"},
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
        background_uids=(1,),
        uid_to_hotkey={1: "hk1"},
        base_shard=Path("pretrained/model_expgroup_1.safetensors"),
        # journal_path is None.
    )
    round_obj.mark_scored(1, score=1.0)
    round_obj.mark_failed(1)
    round_obj.mark_validation_failed(1)
    # No file written, no exceptions raised.
    assert round_obj.scores == {1: 1.0}


def test_late_result_after_finalize_is_dropped(tmp_path: Path) -> None:
    """The bg worker consults the eval-window flag only when claiming a new
    miner, so the one already on the GPU calls `mark_*` after finalize has
    read the scores and written the ranks. Recording it then would leave a
    raw `delta ** 1.2` in the aggregator that no `drop_round` ever removes,
    and would rewrite the journal with `finalized=False` — making startup
    recovery replay an already-finalized round.
    """
    score_path = tmp_path / "score_aggregator.json"
    journal_path = rj.journal_path_for(tmp_path, 7000)
    agg = MinerScoreAggregator(max_points=8, max_history_points=64)

    round_obj = _make_round(
        round_id=7000, journal_path=journal_path,
        uid_to_hotkey={1: "hk1", 2: "hk2", 3: "hk3", 4: "hk4", 5: "hk5"},
    )
    round_obj.mark_scored(1, score=0.5)
    round_obj.mark_scored(2, score=1.0)
    round_obj.mark_scored(3, score=0.25)
    finalize_round_scores(
        round_obj=round_obj, score_aggregator=agg, score_path=score_path,
    )

    # Two evaluations land after the round closed.
    round_obj.mark_scored(4, score=0.9)
    round_obj.mark_failed(5)

    # Neither reaches the round's own state.
    assert round_obj.scored_uids == {1, 2, 3}
    assert round_obj.scores == {1: 0.5, 2: 1.0, 3: 0.25}
    assert round_obj.failed_uids == set()

    # The aggregator holds rank values only — uid 4 has no point at all.
    final = MinerScoreAggregator.from_json(score_path.read_text())
    assert 4 not in final._miners
    for state in final._miners.values():
        for _, value, round_id in state.series.points:
            assert round_id == 7000
            assert value in (2.25, 1.5, 1.0, 0.0)

    # The journal stays finalized, so startup recovery will not replay it.
    assert rj.load(journal_path).finalized is True

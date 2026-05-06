"""Unit tests for the per-round journal — round-trip serialization,
schema-version handling, atomic-write safety, and age-based pruning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from connito.validator import round_journal as rj


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    journal = rj.RoundJournal(
        round_id=8122866,
        uid_to_hotkey={1: "hk1", 2: "hk2", 7: "hk7"},
        scores={1: 0.018, 7: 0.05},
        scored_uids=(1, 7),
        failed_uids=(2,),
        validation_failed_uids=(2,),
        freeze_zero_uids=(0, 99),
        freeze_zero_hotkeys={0: "vme", 99: "stale"},
        finalized=False,
    )
    path = tmp_path / "round_journal" / "round_8122866.json"
    rj.write_atomic(path, journal)

    loaded = rj.load(path)
    assert loaded is not None
    assert loaded.round_id == 8122866
    assert loaded.uid_to_hotkey == {1: "hk1", 2: "hk2", 7: "hk7"}
    assert loaded.scores == {1: 0.018, 7: 0.05}
    assert loaded.scored_uids == (1, 7)
    assert loaded.failed_uids == (2,)
    assert loaded.validation_failed_uids == (2,)
    assert loaded.freeze_zero_uids == (0, 99)
    assert loaded.freeze_zero_hotkeys == {0: "vme", 99: "stale"}
    assert loaded.finalized is False


def test_load_returns_none_on_missing_file(tmp_path: Path) -> None:
    assert rj.load(tmp_path / "round_journal" / "round_999.json") is None


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "round_journal" / "round_1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 999,
        "round_id": 1,
        "uid_to_hotkey": {},
        "scores": {},
        "scored_uids": [],
        "failed_uids": [],
        "validation_failed_uids": [],
        "freeze_zero_uids": [],
        "freeze_zero_hotkeys": {},
        "finalized": False,
    }))
    with pytest.raises(ValueError, match="schema_version"):
        rj.load(path)


def test_atomic_write_no_partial_file(tmp_path: Path) -> None:
    """`write_atomic` must produce no partial files: only the final
    `round_<rid>.json` and (transiently) a `.<name>.tmp` that has been
    `os.replace`d into place. After the call returns, only the final
    file should exist.
    """
    journal = rj.RoundJournal(round_id=42)
    path = tmp_path / "round_journal" / "round_42.json"
    rj.write_atomic(path, journal)

    siblings = list(path.parent.iterdir())
    assert len(siblings) == 1
    assert siblings[0] == path


def test_scan_returns_journals_sorted_by_round_id(tmp_path: Path) -> None:
    for rid in (300, 100, 200):
        rj.write_atomic(rj.journal_path_for(tmp_path, rid), rj.RoundJournal(round_id=rid))
    # Plus a non-journal file that must be ignored.
    (tmp_path / "round_journal" / "garbage.txt").write_text("nope")

    found = rj.scan(tmp_path)
    assert [p.name for p in found] == [
        "round_100.json", "round_200.json", "round_300.json",
    ]


def test_scan_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    assert rj.scan(tmp_path / "nonexistent") == []


def test_prune_before_round_drops_old_journals_only(tmp_path: Path) -> None:
    for rid in (10, 50, 100, 500):
        rj.write_atomic(rj.journal_path_for(tmp_path, rid), rj.RoundJournal(round_id=rid))

    dropped = rj.prune_before_round(tmp_path, min_round_id=100)
    assert dropped == 2  # 10 and 50

    remaining = sorted(p.name for p in rj.scan(tmp_path))
    assert remaining == ["round_100.json", "round_500.json"]


def test_recovery_round_from_journal_carries_finalize_inputs() -> None:
    journal = rj.RoundJournal(
        round_id=7,
        uid_to_hotkey={1: "hk1"},
        scores={1: 0.5},
        scored_uids=(1,),
        validation_failed_uids=(2,),
        freeze_zero_uids=(0,),
        freeze_zero_hotkeys={0: "vme"},
    )
    stub = rj._RecoveryRound.from_journal(journal, Path("/tmp/round_7.json"))
    assert stub.round_id == 7
    assert stub.scores == {1: 0.5}
    assert stub.scored_uids == {1}
    assert stub.validation_failed_uids == {2}
    assert stub.freeze_zero_uids == {0}
    assert stub.freeze_zero_hotkeys == {0: "vme"}
    assert stub.uid_to_hotkey == {1: "hk1"}

    scored, failed = stub.processed_uids_snapshot()
    assert scored == {1}
    assert failed == set()

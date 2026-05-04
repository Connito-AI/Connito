"""Tests for CohortState serialization and atomic persistence."""

from __future__ import annotations

import json

import pytest

from connito.validator.cohort_state import (
    CohortState,
    SCHEMA_VERSION,
    load,
    persist_atomic,
)


def _example_state(epoch: int = 8, expert_group: str = "g1") -> CohortState:
    return CohortState(
        cohort_epoch=epoch,
        expert_group=expert_group,
        weight_group_1=(1, 2, 3),
        weight_group_2=(4, 5, 6, 7, 8),
        validation_group_a=(1, 2, 3),
        validation_group_b=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        validation_group_c=(20, 21, 22, 23, 24, 25, 26),
        last_election_round_id=12345,
        highest_seen_cycle_index=11,
    )


def test_to_json_round_trip_preserves_all_fields():
    original = _example_state()
    restored = CohortState.from_json(original.to_json())
    assert restored == original


def test_to_json_emits_schema_version():
    state = _example_state()
    raw = json.loads(state.to_json())
    assert raw["schema_version"] == SCHEMA_VERSION


def test_from_json_normalizes_lists_to_tuples():
    state = _example_state()
    restored = CohortState.from_json(state.to_json())
    # JSON serializes tuples as lists; from_json must convert back
    # so downstream `set(...)` comparisons stay stable.
    assert isinstance(restored.weight_group_1, tuple)
    assert isinstance(restored.validation_group_b, tuple)
    assert isinstance(restored.validation_group_c, tuple)


def test_from_json_rejects_unknown_schema_version():
    raw = json.loads(_example_state().to_json())
    raw["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        CohortState.from_json(json.dumps(raw))


def test_from_json_handles_missing_optional_fields():
    minimal = json.dumps({
        "cohort_epoch": 0,
        "expert_group": "g1",
        "schema_version": SCHEMA_VERSION,
    })
    restored = CohortState.from_json(minimal)
    assert restored.cohort_epoch == 0
    assert restored.weight_group_1 == ()
    assert restored.validation_group_b == ()
    assert restored.last_election_round_id is None


def test_persist_and_load_round_trip(tmp_path):
    path = tmp_path / "cohort_state.json"
    original = _example_state()
    persist_atomic(path, original)
    assert path.exists()
    loaded = load(path, expected_expert_group="g1")
    assert loaded == original


def test_load_returns_none_when_file_missing(tmp_path):
    assert load(tmp_path / "nope.json", expected_expert_group="g1") is None


def test_load_rejects_expert_group_mismatch(tmp_path):
    path = tmp_path / "cohort_state.json"
    persist_atomic(path, _example_state(expert_group="g1"))
    with pytest.raises(ValueError, match="expert_group"):
        load(path, expected_expert_group="g_other")


def test_persist_is_atomic_no_partial_file(tmp_path):
    """A successful persist leaves only the final file, no `.tmp`."""
    path = tmp_path / "cohort_state.json"
    persist_atomic(path, _example_state())
    leftovers = [p for p in path.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_persist_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "cohort_state.json"
    persist_atomic(path, _example_state())
    assert path.exists()

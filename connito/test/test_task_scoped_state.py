"""What a task switch must carry over, and what it must drop.

Score history is miner history: the roster is unchanged by a switch, so it
survives one. Cohort state was drawn from the old assignment, so it does not.
Both follow from where the file lives relative to the group-scoped path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from connito.shared.config import ValidatorConfig
from connito.validator import cohort_state
from connito.validator.aggregator import resolve_score_path

SHIPPED = "exp_nemotron_c4"
TARGET = "exp_switch_target"


def _task_dir(root: Path, name: str, group_id: int) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))
    (d / "expert_assignment.json").write_text(json.dumps({"0": [[0, group_id * 10]]}))


@pytest.fixture
def config(tmp_path: Path) -> ValidatorConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4)
    _task_dir(tmp_path, TARGET, group_id=7)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    return ValidatorConfig.from_path(cfg_path, active_task=None, auto_update_config=True)


# --- score history survives ---------------------------------------------------

def test_the_score_path_survives_a_switch(config) -> None:
    """The path `run` captures once must carry no expert group."""
    before = resolve_score_path(config.ckpt.checkpoint_path)
    group_scoped_before = config.ckpt.checkpoint_path

    config.switch_active_task(TARGET)

    assert resolve_score_path(config.ckpt.checkpoint_path) == before
    # Not vacuous: the group-scoped path really did move underneath it.
    assert config.ckpt.checkpoint_path != group_scoped_before
    # And it is not the shared model cache, which the miner path also reads.
    assert config.ckpt.validator_checkpoint_path not in before.parents


def test_legacy_history_is_carried_across_the_move(config) -> None:
    legacy = Path(config.ckpt.checkpoint_path) / "score_aggregator.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"7": [1.0, 2.0]}')

    dest = resolve_score_path(config.ckpt.checkpoint_path)

    assert dest.read_text() == '{"7": [1.0, 2.0]}'
    assert not legacy.exists()


def test_an_existing_destination_wins(config) -> None:
    """A validator already on the new layout must not be rolled back."""
    legacy = Path(config.ckpt.checkpoint_path) / "score_aggregator.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"stale": true}')
    dest = Path(config.ckpt.checkpoint_path).parent / "score_aggregator.json"
    dest.write_text('{"current": true}')

    assert resolve_score_path(config.ckpt.checkpoint_path).read_text() == '{"current": true}'


def test_migrating_twice_is_a_no_op(config) -> None:
    legacy_dir = Path(config.ckpt.checkpoint_path)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "score_aggregator.json").write_text('{"a": 1}')

    resolve_score_path(config.ckpt.checkpoint_path)
    dest = resolve_score_path(config.ckpt.checkpoint_path)

    assert dest.read_text() == '{"a": 1}'


# --- cohort state is dropped --------------------------------------------------

def test_cohort_state_is_dropped_not_migrated_across_a_switch(config) -> None:
    """The cohort was drawn from the old assignment, so `run` starts a fresh
    one after a switch. No code moves it: the file sits under the group-scoped
    path, the switch moves the path, and the old state is never found."""
    filename = config.evaluation.cohort_state_filename
    old = Path(config.ckpt.checkpoint_path) / filename
    cohort_state.persist_atomic(old, cohort_state.CohortState(cohort_epoch=8, expert_group="4"))

    config.switch_active_task(TARGET)
    new = Path(config.ckpt.checkpoint_path) / filename

    assert new != old
    assert cohort_state.load(new, expected_expert_group="7") is None
    assert old.exists()

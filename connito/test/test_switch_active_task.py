"""`WorkerConfig.switch_active_task` — re-point a live config at a new task.

`task.path`, the group-scoped `ckpt.checkpoint_path` and `task.exp` all derive
from `task.expert_group_name`, so a switch has to move them together and create
the new directories. The rollback is the case that earns the method — see its
docstring for why a partial switch is silent rather than loud.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from connito.shared.config import MinerConfig

# The locked default. The task the config is constructed on must use this
# name or `check_and_prompt_locked` resets it out from under the fixture.
SHIPPED = "exp_nemotron_c4"
OTHER = "exp_switch_target"


def _task_dir(root: Path, name: str, group_id: int) -> None:
    """A task folder shaped like `expert_groups/<name>/`."""
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))


@pytest.fixture
def config(tmp_path: Path) -> MinerConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4)
    _task_dir(tmp_path, OTHER, group_id=7)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        # Pre-filled so from_path skips the chain lookup.
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    return MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)


def test_every_task_scoped_value_follows_the_switch(config: MinerConfig) -> None:
    assert config.task.exp.group_id == 4

    config.switch_active_task(OTHER)

    assert config.task.expert_group_name == OTHER
    assert config.task.path is not None and config.task.path.name == OTHER
    assert config.ckpt.checkpoint_path is not None
    assert config.ckpt.checkpoint_path.name == OTHER
    # The payload, not just the paths — this is what `_update_by_task` reloads.
    assert config.task.exp.group_id == 7


def test_the_new_groups_directories_are_created(config: MinerConfig) -> None:
    """Both carry the group name, so neither exists after a switch."""
    config.switch_active_task(OTHER)

    assert config.task.path.is_dir()
    assert config.ckpt.checkpoint_path.is_dir()


def test_a_failed_switch_leaves_the_config_on_the_previous_task(
    config: MinerConfig,
) -> None:
    """The rollback. Without it the name moves and `task.exp` does not."""
    before = (
        config.task.expert_group_name,
        config.task.path,
        config.ckpt.checkpoint_path,
        config.task.exp.group_id,
    )

    with pytest.raises(Exception):
        config.switch_active_task("exp_this_node_does_not_have")

    assert (
        config.task.expert_group_name,
        config.task.path,
        config.ckpt.checkpoint_path,
        config.task.exp.group_id,
    ) == before


def test_a_failed_switch_still_loads_the_previous_task_from_disk(
    config: MinerConfig,
) -> None:
    """The rollback must re-derive, not just restore: leaving the fields
    matching keeps `task.path` correct by luck, and the next
    `_update_by_task` reads the wrong directory."""
    with pytest.raises(Exception):
        config.switch_active_task("exp_this_node_does_not_have")

    config._update_by_task()
    assert config.task.exp.group_id == 4


def test_switching_to_the_current_task_is_a_no_op(config: MinerConfig) -> None:
    config.switch_active_task(SHIPPED)

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 4

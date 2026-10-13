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

from connito.shared.config import ExpertCfg, MinerConfig, ResidualCfg

# The locked default. The task the config is constructed on must use this
# name or `check_and_prompt_locked` resets it out from under the fixture.
SHIPPED = "exp_pmc_residual"
OTHER = "exp_switch_target"


def _task_dir(root: Path, name: str, group_id: int, residual: bool = True) -> None:
    """A task folder shaped like `expert_groups/<name>/`."""
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    if not residual:
        del body["residual"]
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


def test_a_task_without_residual_settings_is_refused_and_rolled_back(
    config: MinerConfig, tmp_path: Path,
) -> None:
    """K, A and τ are protocol values; a task that does not serve them cannot be trained or scored."""
    _task_dir(tmp_path, "exp_no_residual", group_id=8, residual=False)
    before = (config.task.expert_group_name, config.task.path, config.ckpt.checkpoint_path, config.task.exp)

    with pytest.raises(ValueError, match="residual"):
        config.switch_active_task("exp_no_residual")

    assert (config.task.expert_group_name, config.task.path, config.ckpt.checkpoint_path, config.task.exp) == before


def test_the_shipped_task_serves_its_protocol_values() -> None:
    exp = ExpertCfg.from_path(Path(f"expert_groups/{SHIPPED}/config.yaml"))

    assert exp.residual == ResidualCfg(k=1, threshold=0.8, temperature=0.05)
    assert {s.path: s.replay for s in exp.data.dataset_sources} == {
        "TomTBT/pmc_open_access_section": False,
        "allenai/c4": True,
    }


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5])
def test_a_threshold_outside_the_gate_range_is_refused(threshold: float) -> None:
    """A is compared with a sigmoid; at 0 or 1 the gate is always open or always shut."""
    with pytest.raises(ValueError):
        ResidualCfg(k=1, threshold=threshold, temperature=0.05)


def test_a_gate_that_fp32_could_shut_for_good_is_refused() -> None:
    """Past A/τ ≈ 88 a token with r near 0 gets q = 0 exactly in fp32, and an honest down_proj can stay all zero."""
    with pytest.raises(ValueError, match="below 80"):
        ResidualCfg(k=1, threshold=0.8, temperature=0.01)

    assert ResidualCfg(k=1, threshold=0.79, temperature=0.01).threshold == 0.79

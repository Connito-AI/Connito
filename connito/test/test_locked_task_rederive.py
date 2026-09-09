"""Activation-order regression: when the active task changes during load,
the derived `task.path` / `task.exp` must be re-derived in the same load.

Observed live on the pioneer validator (2026-07-11 11:49 UTC) during the
exp_legal activation: a config YAML still saying `exp_math` was reset to
the then-locked default `exp_legal` and persisted to disk — but the
process kept RUNNING exp_math (`task_path=/app/expert_groups/exp_math`,
chain commits `group_id: 0`) because `task.path`/`task.exp` had been
derived at construction, before the reset ran. Every fleet validator
would have needed a second restart to actually switch groups.

The source of truth for the name has since moved from a locked class
default to the owner API, so these exercise the same failure through the
new channel: the API names a task the YAML disagrees with, and the whole
derived chain must follow in one load.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from connito.shared import task_sync
from connito.shared.config import MinerConfig
from connito.shared.task_sync import ActiveTask


def _write_cfg(tmp_path: Path, expert_group_name: str) -> Path:
    cfg = {
        "task": {"expert_group_name": expert_group_name},
        # Pre-filled wallet identifiers so from_path skips the chain lookup.
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def owner_api(monkeypatch):
    """Stub the owner API. `None` stands for an unreachable one."""
    def _serve(name: str | None, group_id: int = 4):
        served = None if name is None else ActiveTask(
            name=name, group_id=group_id, bundle_sha256="0" * 64
        )
        # `_apply_owner_task` imports this at call time (the module-scope import
        # would be circular), so patching the attribute here is enough.
        monkeypatch.setattr(task_sync, "get_active_task", lambda config: served)
    return _serve


def test_owner_task_change_rederives_task_path(tmp_path: Path, owner_api) -> None:
    # Run from the repo root so relative expert_groups/<name> resolves.
    assert Path("expert_groups/exp_nemotron_c4/config.yaml").exists(), (
        f"run from repo root (cwd={os.getcwd()})"
    )
    # An operator YAML left behind on a previous task.
    cfg_path = _write_cfg(tmp_path, "exp_legal")
    owner_api("exp_nemotron_c4")

    config = MinerConfig.from_path(cfg_path, auto_update_config=True)

    # The API flipped the name...
    assert config.task.expert_group_name == "exp_nemotron_c4"
    # ...and the DERIVED task state must have followed in the same load:
    assert config.task.path is not None and config.task.path.name == "exp_nemotron_c4"
    assert config.task.exp.group_id == 4, (
        f"stale task.exp — still group_id={config.task.exp.group_id} "
        "(exp_math=0, exp_legal=3): the name changed without re-deriving "
        "task.path/task.exp"
    )
    # ...including the checkpoint path, which is group-scoped so the switch
    # writes/resumes from a fresh dir instead of the prior group's checkpoints.
    assert config.ckpt.checkpoint_path is not None
    assert config.ckpt.checkpoint_path.name == "exp_nemotron_c4", (
        f"checkpoint_path leaf must track the effective group, got "
        f"{config.ckpt.checkpoint_path}"
    )
    # And the persisted YAML matches what the process actually runs, so an API
    # outage on the next boot resumes this task and not the stale one.
    persisted = yaml.safe_load(cfg_path.read_text())
    assert persisted["task"]["expert_group_name"] == "exp_nemotron_c4"


def test_no_change_no_rederive_noise(tmp_path: Path, owner_api) -> None:
    # A config already on the served task loads once and stays put.
    cfg_path = _write_cfg(tmp_path, "exp_nemotron_c4")
    owner_api("exp_nemotron_c4")
    config = MinerConfig.from_path(cfg_path, auto_update_config=True)
    assert config.task.expert_group_name == "exp_nemotron_c4"
    assert config.task.exp.group_id == 4


def test_unreachable_api_keeps_the_configured_task(tmp_path: Path, owner_api) -> None:
    # The agreed outage behaviour: keep mining the current task, never halt.
    cfg_path = _write_cfg(tmp_path, "exp_legal")
    owner_api(None)
    config = MinerConfig.from_path(cfg_path, auto_update_config=True)
    assert config.task.expert_group_name == "exp_legal"
    assert config.task.exp.group_id == 3


def test_unknown_task_is_refused_rather_than_crashing_the_load(tmp_path: Path, owner_api) -> None:
    # Materializing an absent task is a later PR; adopting a name with no
    # files on disk would fail the load outright, which is worse than staying.
    cfg_path = _write_cfg(tmp_path, "exp_nemotron_c4")
    owner_api("exp_not_shipped_yet", group_id=9)
    config = MinerConfig.from_path(cfg_path, auto_update_config=True)
    assert config.task.expert_group_name == "exp_nemotron_c4"
    assert config.task.exp.group_id == 4


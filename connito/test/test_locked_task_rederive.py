"""Activation-order regression: when the locked `task.expert_group_name`
is auto-reset on load (the exp_legal activation path), the derived
`task.path` / `task.exp` must be re-derived in the same load.

Observed live on the pioneer validator (2026-07-11 11:49 UTC): a config
YAML still saying `exp_math` was reset to the locked default `exp_legal`
and persisted to disk — but the process kept RUNNING exp_math
(`task_path=/app/expert_groups/exp_math`, chain commits `group_id: 0`)
because `task.path`/`task.exp` had been derived at construction, before
`check_and_prompt_locked` ran. Every fleet validator would have needed a
second restart to actually switch groups.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from connito.shared.config import MinerConfig


def _write_cfg(tmp_path: Path, expert_group_name: str) -> Path:
    cfg = {
        "task": {"expert_group_name": expert_group_name},
        # Pre-filled wallet identifiers so from_path skips the chain lookup.
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_locked_reset_rederives_task_path(tmp_path: Path) -> None:
    # Run from the repo root so relative expert_groups/<name> resolves.
    assert Path("expert_groups/exp_legal/config.yaml").exists(), (
        f"run from repo root (cwd={os.getcwd()})"
    )
    cfg_path = _write_cfg(tmp_path, "exp_math")

    config = MinerConfig.from_path(cfg_path, auto_update_config=True)

    # The locked default flipped the name...
    assert config.task.expert_group_name == "exp_legal"
    # ...and the DERIVED task state must have followed in the same load:
    assert config.task.path is not None and config.task.path.name == "exp_legal"
    assert config.task.exp.group_id == 3, (
        f"stale task.exp — still group_id={config.task.exp.group_id} "
        "(exp_math=0): check_and_prompt_locked reset the name without "
        "re-deriving task.path/task.exp"
    )
    # And the persisted YAML matches what the process actually runs.
    persisted = yaml.safe_load(cfg_path.read_text())
    assert persisted["task"]["expert_group_name"] == "exp_legal"


def test_no_reset_no_rederive_noise(tmp_path: Path) -> None:
    # A config already on the locked default loads once, stays exp_legal.
    cfg_path = _write_cfg(tmp_path, "exp_legal")
    config = MinerConfig.from_path(cfg_path, auto_update_config=True)
    assert config.task.expert_group_name == "exp_legal"
    assert config.task.exp.group_id == 3

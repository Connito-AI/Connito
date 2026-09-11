"""Activation-order regression: the active task must be applied before
anything is derived from it.

Observed live on the pioneer validator (2026-07-11 11:49 UTC) during the
exp_legal activation: a config YAML still saying `exp_math` was reset to
the then-locked default `exp_legal` and persisted to disk — but the
process kept RUNNING exp_math (`task_path=/app/expert_groups/exp_math`,
chain commits `group_id: 0`), because `task.path`/`task.exp` had been
derived at construction, before the reset ran. Every fleet validator
would have needed a second restart to actually switch groups.

The source of truth for the name has since moved to the owner API, and
`from_path` now applies it *before* constructing rather than correcting
afterwards — so the whole derived chain is right the first time. These
pin that, plus the fallback that keeps a node alive when it is handed a
task it does not have.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
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


def test_active_task_is_applied_before_anything_is_derived(tmp_path: Path) -> None:
    # Run from the repo root so relative expert_groups/<name> resolves.
    assert Path("expert_groups/exp_nemotron_c4/config.yaml").exists(), (
        f"run from repo root (cwd={os.getcwd()})"
    )
    # An operator YAML left behind on a previous task; the API says otherwise.
    cfg_path = _write_cfg(tmp_path, "exp_legal")

    config = MinerConfig.from_path(cfg_path, active_task="exp_nemotron_c4")

    assert config.task.expert_group_name == "exp_nemotron_c4"
    # The DERIVED state must match, in this same load:
    assert config.task.path is not None and config.task.path.name == "exp_nemotron_c4"
    assert config.task.exp.group_id == 4, (
        f"stale task.exp — still group_id={config.task.exp.group_id} "
        "(exp_math=0, exp_legal=3): the name was applied after task.path/task.exp "
        "had already been derived"
    )
    # ...including the checkpoint path, which is group-scoped so the switch
    # writes/resumes from a fresh dir instead of the prior group's checkpoints.
    assert config.ckpt.checkpoint_path is not None
    assert config.ckpt.checkpoint_path.name == "exp_nemotron_c4", (
        f"checkpoint_path leaf must track the effective group, got "
        f"{config.ckpt.checkpoint_path}"
    )


def test_no_active_task_falls_back_to_the_config(tmp_path: Path) -> None:
    # What an unreachable owner API looks like from here: the resolver returns
    # None and the node starts on the task its config already names.
    cfg_path = _write_cfg(tmp_path, "exp_legal")
    config = MinerConfig.from_path(cfg_path, active_task=None)
    assert config.task.expert_group_name == "exp_legal"
    assert config.task.exp.group_id == 3


def test_a_task_we_do_not_have_degrades_instead_of_crashing(tmp_path: Path) -> None:
    # Fetching an absent task is PR 4's job. Until then this must not raise:
    # the failure would happen during construction, before anything could ask
    # the API or download it, which is an unrecoverable crash loop.
    cfg_path = _write_cfg(tmp_path, "exp_nemotron_c4")
    config = MinerConfig.from_path(cfg_path, active_task="exp_not_shipped_yet")
    assert config.task.expert_group_name == "exp_nemotron_c4"
    assert config.task.exp.group_id == 4
    assert config.ckpt.checkpoint_path.name == "exp_nemotron_c4"


def test_active_task_is_required_so_a_new_entrypoint_cannot_skip_it() -> None:
    # Keyword-only with no default: forgetting to resolve the task would
    # silently disable the whole mechanism, so it has to be a TypeError.
    with pytest.raises(TypeError):
        MinerConfig.from_path("unused.yaml")

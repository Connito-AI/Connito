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

from pathlib import Path

import pytest
import yaml

from connito.shared.config import MinerConfig

SHIPPED = "exp_pmc_residual"  # the fallback default
PREVIOUS = "exp_previous"


def _task_dir(root: Path, name: str, group_id: int | None, residual: bool = True) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    if group_id is None:
        del body["group_id"]
    if not residual:
        del body["residual"]
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))


def _write_cfg(tmp_path: Path, expert_group_name: str) -> Path:
    _task_dir(tmp_path, SHIPPED, group_id=9)
    _task_dir(tmp_path, PREVIOUS, group_id=3)
    cfg = {
        "run": {"root_path": str(tmp_path)},
        "task": {"expert_group_name": expert_group_name},
        # Pre-filled wallet identifiers so from_path skips the chain lookup.
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_active_task_is_applied_before_anything_is_derived(tmp_path: Path) -> None:
    # An operator YAML left behind on a previous task; the API says otherwise.
    cfg_path = _write_cfg(tmp_path, PREVIOUS)

    config = MinerConfig.from_path(cfg_path, active_task=SHIPPED)

    assert config.task.expert_group_name == SHIPPED
    # The DERIVED state must match, in this same load:
    assert config.task.path is not None and config.task.path.name == SHIPPED
    assert config.task.exp.group_id == 9, (
        f"stale task.exp — still group_id={config.task.exp.group_id}: the name was "
        "applied after task.path/task.exp had already been derived"
    )
    # ...including the checkpoint path, which is group-scoped so the switch
    # writes/resumes from a fresh dir instead of the prior group's checkpoints.
    assert config.ckpt.checkpoint_path is not None
    assert config.ckpt.checkpoint_path.name == SHIPPED, (
        f"checkpoint_path leaf must track the effective group, got "
        f"{config.ckpt.checkpoint_path}"
    )


def test_no_active_task_falls_back_to_the_config(tmp_path: Path) -> None:
    # What an unreachable owner API looks like from here: the resolver returns
    # None and the node starts on the task its config already names.
    cfg_path = _write_cfg(tmp_path, PREVIOUS)
    config = MinerConfig.from_path(cfg_path, active_task=None)
    assert config.task.expert_group_name == PREVIOUS
    assert config.task.exp.group_id == 3


def test_a_task_we_do_not_have_degrades_instead_of_crashing(tmp_path: Path) -> None:
    # Raising here would happen during construction, before anything could ask
    # the API or download the task, which is an unrecoverable crash loop.
    cfg_path = _write_cfg(tmp_path, SHIPPED)
    config = MinerConfig.from_path(cfg_path, active_task="exp_not_shipped_yet")
    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 9
    assert config.ckpt.checkpoint_path.name == SHIPPED


def test_a_task_materialized_for_an_older_release_degrades_instead_of_crashing(tmp_path: Path) -> None:
    # A task folder from before the residual protocol survives an upgrade on disk.
    cfg_path = _write_cfg(tmp_path, SHIPPED)
    _task_dir(tmp_path, "exp_pmc_c4", group_id=8, residual=False)

    config = MinerConfig.from_path(cfg_path, active_task="exp_pmc_c4")

    assert config.task.expert_group_name == SHIPPED
    assert config.task.path.name == SHIPPED
    assert config.task.exp.group_id == 9
    assert config.ckpt.checkpoint_path.name == SHIPPED


@pytest.mark.parametrize("group_id", [-1, None], ids=["sentinel", "missing"])
def test_a_task_without_a_real_group_id_falls_back_at_boot(tmp_path: Path, group_id: int | None) -> None:
    cfg_path = _write_cfg(tmp_path, SHIPPED)
    _task_dir(tmp_path, "exp_no_group", group_id=group_id)

    config = MinerConfig.from_path(cfg_path, active_task="exp_no_group")

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 9


def test_the_shipped_default_itself_unloadable_still_raises(tmp_path: Path, caplog) -> None:
    # Nothing to fall back to: fail loudly, without claiming a fallback first.
    cfg_path = _write_cfg(tmp_path, SHIPPED)
    _task_dir(tmp_path, SHIPPED, group_id=9, residual=False)
    with pytest.raises(ValueError, match="residual"):
        MinerConfig.from_path(cfg_path, active_task=None)
    assert "falling back" not in caplog.text


def test_active_task_is_required_so_a_new_entrypoint_cannot_skip_it() -> None:
    # Keyword-only with no default: forgetting to resolve the task would
    # silently disable the whole mechanism, so it has to be a TypeError.
    with pytest.raises(TypeError):
        MinerConfig.from_path("unused.yaml")


@pytest.mark.parametrize(
    "corrupt",
    [b"group_id: 8\n  sources: [unclosed\n", b"group_id: 8\n\xff\xfe\x00", b"<html>502 Bad Gateway</html>\n"],
    ids=["yaml_syntax", "not_utf8", "not_a_mapping"],
)
def test_a_corrupt_task_file_degrades_instead_of_crashing(tmp_path: Path, caplog, capsys, corrupt: bytes) -> None:
    # A truncated or hand-edited config.yaml on disk must not crash-loop the node either.
    cfg_path = _write_cfg(tmp_path, SHIPPED)
    _task_dir(tmp_path, "exp_corrupt", group_id=8)
    (tmp_path / "expert_groups" / "exp_corrupt" / "config.yaml").write_bytes(corrupt)

    config = MinerConfig.from_path(cfg_path, active_task="exp_corrupt")

    assert config.task.expert_group_name == SHIPPED
    assert config.task.exp.group_id == 9
    assert "falling back to the shipped default" in caplog.text + capsys.readouterr().out

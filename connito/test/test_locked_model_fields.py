"""An operator YAML written by an older release carries stale fields; loading must reset or migrate them."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from structlog.testing import capture_logs

from connito.shared.config import MinerConfig

SHIPPED = "exp_pmc_residual"  # the locked default task


def test_stale_model_fields_are_reset_on_load(tmp_path: Path) -> None:
    task_dir = tmp_path / "expert_groups" / SHIPPED
    task_dir.mkdir(parents=True)
    (task_dir / "config.yaml").write_text(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        # Pre-filled so from_path skips the chain lookup.
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
        # Fields since removed (`model_path`, `precision`, `torch_compile`) must still load and drop out on rewrite.
        "model": {
            "precision": "fp16-mixed", "model_path": "deepseek-ai/DeepSeek-V2-Lite", "torch_compile": True,
            "family": "deepseek_v2", "quantization_validator": "all",
        },
    }))

    config = MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)

    assert (config.model.family, config.model.quantization_validator) == ("qwen3_5_moe", "experts")
    written = yaml.safe_load(cfg_path.read_text())["model"]
    assert (written["family"], written["quantization_validator"]) == ("qwen3_5_moe", "experts")
    assert not {"model_path", "precision", "torch_compile"} & written.keys()


@pytest.mark.parametrize("old_release, expected", [(True, (2, 2)), (False, (1, 8))])
def test_accumulation_from_before_micro_batch_takes_the_new_default(tmp_path: Path, monkeypatch, old_release,
                                                                    expected) -> None:
    """A full dump from before micro_batch_size counted 1-row micro-batches: its accumulation of 4 is not kept."""
    task_dir = tmp_path / "expert_groups" / SHIPPED
    task_dir.mkdir(parents=True)
    (task_dir / "config.yaml").write_text(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    monkeypatch.chdir(tmp_path)  # the rewrite stores run.root_path as ".", and paths derived from it are created
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)  # now a full dump, as nodes write
    dump = yaml.safe_load(cfg_path.read_text())
    if old_release:
        del dump["local_par"]["micro_batch_size"]
        dump["local_par"]["gradient_accumulation_steps"] = 4
    else:
        dump["local_par"] |= {"micro_batch_size": 1, "gradient_accumulation_steps": 8}
    cfg_path.write_text(yaml.safe_dump(dump))

    with capture_logs() as logs:
        config = MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)

    assert (config.local_par.micro_batch_size, config.local_par.gradient_accumulation_steps) == expected
    written = yaml.safe_load(cfg_path.read_text())["local_par"]
    assert (written["micro_batch_size"], written["gradient_accumulation_steps"]) == expected
    # Without auto_update or a TTY nothing is rewritten, so this warning, naming the remedy, is all an operator sees.
    warned = [e for e in logs if e["log_level"] == "warning" and "add local_par.micro_batch_size" in e["event"]]
    assert len(warned) == old_release

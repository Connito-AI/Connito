"""An operator YAML written before the family registry carries stale model fields; the locks must reset them."""

from __future__ import annotations

from pathlib import Path

import yaml

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

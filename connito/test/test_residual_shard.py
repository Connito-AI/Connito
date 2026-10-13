"""The residual shard: its spec for each family and K, how it is written, and what a check refuses.

CPU only, tiny models, nothing downloaded: `python -m pytest connito/test/test_residual_shard.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from connito.shared import model as shared_model
from connito.shared.config import ResidualCfg
from connito.shared.modeling.residual import residual_shapes

# The tiny checkpoints' MoE layers (DeepSeek-V2's layer 0 is dense), hidden size and expert width.
MOE_LAYERS = {"qwen3_5_moe": [0, 1, 2, 3], "deepseek_v2": [1, 2]}
HIDDEN, WIDTH = 64, 32


def _config(name: str, k: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        role="miner",
        model=SimpleNamespace(family=name, device="cpu", quantization="off", quantization_miner="off"),
        task=SimpleNamespace(
            exp=SimpleNamespace(group_id=9, residual=ResidualCfg(k=k, threshold=0.5, temperature=0.5)),
        ),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )


def _model(config) -> torch.nn.Module:
    model, _ = shared_model.get_model_from_checkpoint(rank=0, config=config, load_global_checkpoint=False)
    return model


def _expected(name: str, k: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    spec = {}
    for layer in MOE_LAYERS[name]:
        prefix = f"model.layers.{layer}.mlp.residual_"
        for j in range(k):
            spec[f"{prefix}experts.{j}.gate_proj.weight"] = ((WIDTH, HIDDEN), torch.bfloat16)
            spec[f"{prefix}experts.{j}.up_proj.weight"] = ((WIDTH, HIDDEN), torch.bfloat16)
            spec[f"{prefix}experts.{j}.down_proj.weight"] = ((HIDDEN, WIDTH), torch.bfloat16)
        spec[f"{prefix}router.weight"] = ((k, HIDDEN), torch.float32)
    return spec


@pytest.mark.parametrize("k", [1, 2])
def test_residual_shapes_are_the_familys_moe_layers_at_k(family_name, k):
    assert residual_shapes(_config(family_name, k)) == _expected(family_name, k)


def test_residual_shapes_match_the_installed_residual(family_name):
    config = _config(family_name, k=2)

    live = {n: (tuple(p.shape), p.dtype) for n, p in _model(config).named_parameters() if ".residual_" in n}

    assert residual_shapes(config) == live

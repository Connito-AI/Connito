# python3 -m pytest connito/test/test_custom_deepseek_v2_lite.py
"""Integration tests for the partial DeepSeek-V2-Lite build.

Downloads the real ~16 GB checkpoint; gated behind RUN_DEEPSEEK_V2_LITE_TEST=1.

These used to build the full model and call `convert_full_to_partial_model`.
That function is gone: routed experts are now individual `DeepseekV2MLP` modules
named exactly as the checkpoint names them, so a partial model is built directly
by `from_pretrained` and the subset is chosen by which experts were declared.

`test_partial_from_pretrained_matches_checkpoint_weights` is the regression test
for the failure that motivated the change: with the previous fused/stacked
storage, `from_pretrained` matched no routed-expert key at all, `strict=False`
swallowed both the unexpected and the missing keys, and 14.39 B of 15.71 B
parameters silently kept their `_init_weights` random values — measured as an
eval loss of 9.956 against the stock model's 1.449.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.utils import cached_file

from connito.shared.expert_manager import get_layer_expert_id


MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
GROUP_ID = 0
EXPERTS_PER_LAYER = 2
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

pytestmark = pytest.mark.integration

requires_checkpoint = pytest.mark.skipif(
    os.getenv("RUN_DEEPSEEK_V2_LITE_TEST") != "1",
    reason="Set RUN_DEEPSEEK_V2_LITE_TEST=1 to run this heavy integration test.",
)


def _partial_config(experts_per_layer: int = EXPERTS_PER_LAYER) -> Any:
    """Config for a partial build: the first K experts of every MoE layer."""
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    total_experts = int(getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts")))
    if not hasattr(cfg, "num_experts"):
        cfg.num_experts = total_experts

    first_moe_layer = int(getattr(cfg, "first_k_dense_replace", 0))
    selected = min(experts_per_layer, total_experts)
    layer_map = {
        layer_id: [(i, i) for i in range(selected)]
        for layer_id in range(first_moe_layer, int(cfg.num_hidden_layers))
    }

    cfg.expert_group_assignment = {GROUP_ID: layer_map}
    cfg.group_ids_trainable = [GROUP_ID]
    cfg.group_ids_helper = None
    cfg.full = False
    return cfg


def _checkpoint_state_dict() -> dict[str, torch.Tensor]:
    """The stock checkpoint's tensors, keyed as the checkpoint keys them."""
    import json

    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

    index_path = cached_file(MODEL_ID, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path) as handle:
        shards = sorted(set(json.load(handle)["weight_map"].values()))

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(cached_file(MODEL_ID, shard)))
    return state


@requires_checkpoint
def test_partial_from_pretrained_matches_checkpoint_weights() -> None:
    """Every materialised parameter must equal the checkpoint's, exactly.

    No `torch.allclose` here — the load is a name match and a copy, so anything
    short of equality means a tensor was transformed or left at its random init.
    """
    deepseek_mod = pytest.importorskip(
        "connito.shared.modeling.custom_deepseek_v2_lite",
        reason="custom DeepSeek-V2-lite modeling is unavailable in this environment",
    )
    cfg = _partial_config()

    model = deepseek_mod.CustomDeekSeekMoE.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).eval()

    checkpoint = _checkpoint_state_dict()
    model_state = model.state_dict()

    # The partial model declares a strict subset of the checkpoint's tensors.
    assert set(model_state) <= set(checkpoint), sorted(set(model_state) - set(checkpoint))[:10]

    expert_params = 0
    for name, tensor in model_state.items():
        expected = checkpoint[name].to(tensor.dtype)
        assert torch.equal(tensor, expected), name
        if get_layer_expert_id(name)[1] is not None:
            expert_params += 1

    # Guard the guard: an empty expert set would make the loop above vacuous,
    # which is exactly how the original bug went unnoticed.
    assert expert_params > 0, "no routed-expert parameters were materialised"


@requires_checkpoint
def test_partial_declares_only_the_assigned_experts() -> None:
    deepseek_mod = pytest.importorskip(
        "connito.shared.modeling.custom_deepseek_v2_lite",
        reason="custom DeepSeek-V2-lite modeling is unavailable in this environment",
    )
    cfg = _partial_config()
    model = deepseek_mod.CustomDeekSeekMoE(deepcopy(cfg))

    seen: set[int] = set()
    for name in model.state_dict():
        _, expert_id = get_layer_expert_id(name)
        if expert_id is not None:
            seen.add(expert_id)
    assert seen == set(range(EXPERTS_PER_LAYER))


@requires_checkpoint
def test_partial_state_dict_round_trips(tmp_path: Path) -> None:
    deepseek_mod = pytest.importorskip(
        "connito.shared.modeling.custom_deepseek_v2_lite",
        reason="custom DeepSeek-V2-lite modeling is unavailable in this environment",
    )
    cfg = _partial_config()
    model = deepseek_mod.CustomDeekSeekMoE.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).eval()

    checkpoint_path = tmp_path / "deepseek_v2_lite_partial_state_dict.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    reloaded = deepseek_mod.CustomDeekSeekMoE(deepcopy(cfg))
    saved = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = reloaded.load_state_dict(
        saved["model_state_dict"], strict=False,
    )

    assert not missing_keys
    assert not unexpected_keys
    assert len(reloaded.state_dict()) == len(model.state_dict())


@requires_checkpoint
def test_partial_model_forward() -> None:
    deepseek_mod = pytest.importorskip(
        "connito.shared.modeling.custom_deepseek_v2_lite",
        reason="custom DeepSeek-V2-lite modeling is unavailable in this environment",
    )
    cfg = _partial_config()
    model = deepseek_mod.CustomDeekSeekMoE.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).eval()

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, int(cfg.vocab_size), (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert outputs.logits is not None
    assert outputs.logits.shape == (batch_size, seq_len, int(cfg.vocab_size))

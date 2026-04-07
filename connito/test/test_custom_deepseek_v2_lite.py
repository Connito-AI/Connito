# python3 -m pytest connito/test/test_custom_deepseek_v2_lite.py
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoConfig

from connito.shared.expert_manager import get_layer_expert_id


MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
GROUP_ID = 0
EXPERTS_PER_LAYER = 2
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


pytestmark = pytest.mark.integration


def _find_moe_layers(full_model: Any) -> list[int]:
    layers: set[int] = set()
    for name in full_model.state_dict().keys():
        layer_id, expert_id = get_layer_expert_id(name)
        if "experts" in name:
            layers.add(layer_id)
    return sorted(layers)


def _build_dummy_expert_group_assignment(
    full_model: Any,
    group_id: int = GROUP_ID,
    experts_per_layer: int = EXPERTS_PER_LAYER,
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    cfg = full_model.config
    total_experts = int(getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts")))
    selected = min(experts_per_layer, total_experts)
    layers = _find_moe_layers(full_model)

    if not layers:
        raise RuntimeError("No MoE layers detected in full model state_dict.")

    layer_map: dict[int, list[tuple[int, int]]] = {}
    for layer_id in layers:
        layer_map[layer_id] = [(i, i) for i in range(selected)]

    return {group_id: layer_map}


@pytest.mark.skipif(
    os.getenv("RUN_DEEPSEEK_V2_LITE_TEST") != "1",
    reason="Set RUN_DEEPSEEK_V2_LITE_TEST=1 to run this heavy integration test.",
)
def test_full_to_partial_save_and_reload(tmp_path: Path) -> None:
    deepseek_mod = pytest.importorskip(
        "connito.shared.modeling.custom_deepseek_v2_lite",
        reason="custom DeepSeek-V2-lite modeling is unavailable in this environment",
    )
    CustomDeekSeekMoE = deepseek_mod.CustomDeekSeekMoE
    convert_full_to_partial_model = deepseek_mod.convert_full_to_partial_model

    # (1) Load full model.
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    if not hasattr(full_cfg, "num_experts"):
        full_cfg.num_experts = int(getattr(full_cfg, "n_routed_experts"))
    full_cfg.expert_group_assignment = None
    full_cfg.group_ids = None

    full_model = CustomDeekSeekMoE.from_pretrained(
        MODEL_ID,
        config=full_cfg,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    full_model.eval()
    assert full_model is not None
    assert len(full_model.state_dict()) > 0

    # (2) Convert to partial model with dummy assignment and save state_dict.
    expert_group_assignment = _build_dummy_expert_group_assignment(full_model)

    partial_cfg = deepcopy(full_model.config)
    partial_cfg.expert_group_assignment = expert_group_assignment
    partial_cfg.group_ids = [GROUP_ID]
    if not hasattr(partial_cfg, "num_experts"):
        partial_cfg.num_experts = int(getattr(partial_cfg, "n_routed_experts"))

    partial_model = CustomDeekSeekMoE(partial_cfg)
    partial_model = convert_full_to_partial_model(
        partial_model=partial_model,
        full_model=full_model,
        expert_group_assignment=expert_group_assignment,
        target_group=GROUP_ID,
    )
    partial_model.eval()

    checkpoint_path = tmp_path / "deepseek_v2_lite_partial_dummy_state_dict.pt"
    torch.save({"model_state_dict": partial_model.state_dict()}, checkpoint_path)
    assert checkpoint_path.exists()

    # (3) Reload partial model via load_state_dict.
    reloaded_partial = CustomDeekSeekMoE(deepcopy(partial_cfg))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = reloaded_partial.load_state_dict(checkpoint["model_state_dict"], strict=False)
    reloaded_partial.eval()

    assert reloaded_partial is not None
    assert not missing_keys
    assert not unexpected_keys
    assert len(reloaded_partial.state_dict()) == len(partial_model.state_dict())


@pytest.mark.skipif(
    os.getenv("RUN_DEEPSEEK_V2_LITE_TEST") != "1",
    reason="Set RUN_DEEPSEEK_V2_LITE_TEST=1 to run this heavy integration test.",
)
def test_partial_model_forward() -> None:
    deepseek_mod = pytest.importorskip(
        "connito.shared.modeling.custom_deepseek_v2_lite",
        reason="custom DeepSeek-V2-lite modeling is unavailable in this environment",
    )
    CustomDeekSeekMoE = deepseek_mod.CustomDeekSeekMoE
    convert_full_to_partial_model = deepseek_mod.convert_full_to_partial_model

    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    if not hasattr(full_cfg, "num_experts"):
        full_cfg.num_experts = int(getattr(full_cfg, "n_routed_experts"))
    full_cfg.expert_group_assignment = None
    full_cfg.group_ids = None

    full_model = CustomDeekSeekMoE.from_pretrained(
        MODEL_ID,
        config=full_cfg,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    full_model.eval()

    expert_group_assignment = _build_dummy_expert_group_assignment(full_model)
    
    print("expert_group_assignment", expert_group_assignment)

    partial_cfg = deepcopy(full_model.config)
    partial_cfg.expert_group_assignment = expert_group_assignment
    partial_cfg.group_ids = [GROUP_ID]
    if not hasattr(partial_cfg, "num_experts"):
        partial_cfg.num_experts = int(getattr(partial_cfg, "n_routed_experts"))

    partial_model = CustomDeekSeekMoE(partial_cfg)
    partial_model = convert_full_to_partial_model(
        partial_model=partial_model,
        full_model=full_model,
        expert_group_assignment=expert_group_assignment,
        target_group=GROUP_ID,
    )
    partial_model.eval()

    seq_len = 8
    batch_size = 1
    input_ids = torch.randint(
        low=0,
        high=int(partial_cfg.vocab_size),
        size=(batch_size, seq_len),
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = partial_model(input_ids=input_ids, attention_mask=attention_mask)

    assert hasattr(outputs, "logits")
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == batch_size
    assert outputs.logits.shape[1] == seq_len
    assert outputs.logits.shape[2] == int(partial_cfg.vocab_size)

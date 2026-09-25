"""The miner trains the full model by default, and saving it stays cheap.

`moe.miner_topology` replaces `moe.partial_moe`, which nothing read. The new
name is the point: miner YAMLs are written with every field, so existing ones
say `partial_moe: true`, and only a field they do not carry picks up the new
default.

Under full topology every frozen expert is an fp8 module, and
`model.state_dict()` dequantizes each one to a CPU copy. The group save must
not go through it when only the active group is kept.

CPU only, tiny model, nothing is downloaded.
Run with `python -m pytest connito/test/test_miner_full_topology.py`.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from safetensors import safe_open
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Config

from connito.shared.checkpoint_helper import save_checkpoint
from connito.shared.config import MoECfg
from connito.shared.expert_manager import get_layer_expert_id
from connito.shared.model import freeze_parameters
from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeekSeekMoE
from connito.shared.modeling.quantization import FP8Linear, quantize_

HIDDEN, INTER, NUM_EXPERTS, TOP_K, LAYER_ID = 32, 16, 8, 2, 1
GROUP = 0
ASSIGNED = [1, 5]
ASSIGNMENT = {GROUP: {LAYER_ID: [(0, 1), (1, 5)]}}


class StubExpertManager:
    expert_group_assignment = ASSIGNMENT


def test_default_topology_is_full():
    assert MoECfg().miner_topology == "full"


def test_config_written_with_partial_moe_gets_full():
    """What an existing miner YAML carries: the old field, not the new one."""
    assert MoECfg(partial_moe=True).miner_topology == "full"


def test_partial_is_still_selectable():
    assert MoECfg(miner_topology="partial").miner_topology == "partial"


def test_unknown_topology_is_rejected():
    with pytest.raises(ValidationError):
        MoECfg(miner_topology="half")


@pytest.fixture()
def full_fp8_model():
    """Full topology, active group trainable, everything else fp8 — the miner's layout."""
    cfg = DeepseekV2Config(
        hidden_size=HIDDEN, intermediate_size=4 * HIDDEN, moe_intermediate_size=INTER,
        num_hidden_layers=2, num_attention_heads=4, n_routed_experts=NUM_EXPERTS,
        n_shared_experts=1, num_experts_per_tok=TOP_K, first_k_dense_replace=1,
        topk_method="greedy", n_group=1, topk_group=1, vocab_size=64,
    )
    cfg.num_experts = NUM_EXPERTS
    cfg.full = True
    cfg.expert_group_assignment = ASSIGNMENT
    model = CustomDeekSeekMoE(cfg)
    freeze_parameters(model=model, expert_manager=StubExpertManager(), expert_group_id=GROUP)
    converted = quantize_(model, "experts", ASSIGNMENT[GROUP])
    assert converted, "the frozen experts should have been converted to fp8"
    assert any(isinstance(m, FP8Linear) for m in model.modules())
    return model


def test_group_save_skips_state_dict(full_fp8_model, tmp_path, monkeypatch):
    def _refuse():
        raise AssertionError("state_dict() dequantizes every fp8 module; the group save must not call it")

    monkeypatch.setattr(full_fp8_model, "state_dict", _refuse)
    save_checkpoint(
        checkpoint_path=tmp_path / "ckpt", model=full_fp8_model, rank=0, save_global_state=False,
        save_model_by_expert_group=True, expert_manager=StubExpertManager(), active_expert_group_id=GROUP,
    )

    with safe_open(str(tmp_path / "ckpt" / f"model_expgroup_{GROUP}.safetensors"), "pt") as f:
        saved = set(f.keys())
    expected = {
        name for name, _ in full_fp8_model.named_parameters()
        if get_layer_expert_id(name)[1] in ASSIGNED
    }
    assert expected and saved == expected


def test_trainable_state_holds_only_what_the_optimizer_changes(full_fp8_model):
    from connito.miner.train import _trainable_state

    state = _trainable_state(full_fp8_model)
    assert state
    assert all(get_layer_expert_id(name)[1] in ASSIGNED for name in state)
    assert all(t.requires_grad is False for t in state.values()), "detached, so hashing records no graph"

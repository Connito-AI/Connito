"""`quantize_` converts exactly the in-scope modules a role never trains.

The cases worth pinning are the two that fail silently: converting an expert the
assignment marks trainable (its weight becomes a buffer and drops out of the
optimizer with no error), and letting `.to(dtype=...)` upcast the fp8 buffer back
to a floating dtype so the memory saving disappears.

CPU only, tiny hidden size, nothing downloaded.

Run with `python -m pytest connito/test/test_fp8_quantization.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Config

from connito.shared.expert_manager import get_layer_expert_id
from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeekSeekMoE
from connito.shared.modeling.quantization import (
    FP8_DTYPE,
    FP8Linear,
    is_quantized,
    quantize_,
)

GROUP, HELPER_GROUP, LAYER_ID = 0, 2, 1
# (my_expert_id, org_expert_id) — selection must key off the global id.
ASSIGNMENT = {
    GROUP: {LAYER_ID: [(0, 1), (1, 5)]},
    HELPER_GROUP: {LAYER_ID: [(0, 3), (1, 6)]},
}
ASSIGNED, HELPER = [1, 5], [3, 6]

# Layer 0 is dense (first_k_dense_replace=1), layer 1 is MoE.
EXPECTED = {
    "off": 0,
    "attention": 8,           # q_a/q_b/kv_b/o on 2 layers; kv_a_proj_with_mqa denied
    "experts": 6,             # helper experts 3 and 6, three projections each
    "attention+experts": 14,
    "all": 20,                # + dense MLP (3) + shared_experts (3)
}


def _model() -> nn.Module:
    cfg = DeepseekV2Config(
        hidden_size=32, intermediate_size=128, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, n_routed_experts=8,
        n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1,
        topk_method="greedy", n_group=1, topk_group=1, vocab_size=64,
    )
    cfg.num_experts = 8
    cfg.full = False
    cfg.expert_group_assignment = ASSIGNMENT
    cfg.group_ids_trainable = [GROUP]
    cfg.group_ids_helper = [HELPER_GROUP]
    cfg.routing_mode = "masked_topk"
    return CustomDeekSeekMoE(cfg)


@pytest.fixture(scope="module")
def baseline() -> nn.Module:
    return _model()


@pytest.mark.parametrize("scope", list(EXPECTED))
def test_scope_converts_the_expected_count(scope):
    assert len(quantize_(_model(), scope, ASSIGNMENT[GROUP])) == EXPECTED[scope]


def _expert_ids(names: list[str]) -> set[int]:
    return {eid for eid in (get_layer_expert_id(n)[1] for n in names) if eid is not None}


@pytest.mark.parametrize("scope", [s for s in EXPECTED if s != "off"])
def test_assigned_experts_are_never_converted(scope):
    """The silent failure: a converted trainable expert leaves the optimizer."""
    leaked = _expert_ids(quantize_(_model(), scope, ASSIGNMENT[GROUP])) & set(ASSIGNED)
    assert not leaked, f"converted trainable experts: {sorted(leaked)}"


def test_helper_experts_are_converted_under_experts_scope():
    assert _expert_ids(quantize_(_model(), "experts", ASSIGNMENT[GROUP])) == set(HELPER)


@pytest.mark.parametrize("scope", [s for s in EXPECTED if s != "off"])
def test_denylist_survives_every_scope(scope):
    converted = quantize_(_model(), scope, ASSIGNMENT[GROUP])
    suffixes = {n.rsplit(".", 1)[-1] for n in converted}
    assert suffixes.isdisjoint({"lm_head", "gate", "kv_a_proj_with_mqa"})


def test_gate_proj_is_converted_but_gate_is_not():
    """`mlp.gate` matched as a substring would also drop `mlp.gate_proj`."""
    converted = quantize_(_model(), "all", ASSIGNMENT[GROUP])
    assert "model.layers.0.mlp.gate_proj" in converted
    assert "model.layers.1.mlp.gate" not in converted


def test_state_dict_and_named_parameters_are_unchanged(baseline):
    model = _model()
    quantize_(model, "all", ASSIGNMENT[GROUP])
    assert is_quantized(model)

    base_sd, sd = baseline.state_dict(), model.state_dict()
    assert set(sd) == set(base_sd)
    assert all(sd[k].shape == base_sd[k].shape for k in sd)
    assert {n for n, _ in model.named_parameters()} <= {n for n, _ in baseline.named_parameters()}


def test_cast_does_not_upcast_the_fp8_buffer():
    """fp8 is a floating dtype, so `.to()` silently erases the saving."""
    model = _model()
    quantize_(model, "all", ASSIGNMENT[GROUP])
    model.to(dtype=torch.bfloat16)
    buffers = [m.weight_fp8 for m in model.modules() if isinstance(m, FP8Linear)]
    assert buffers and all(b.dtype is FP8_DTYPE for b in buffers)


def test_round_trip_error_is_within_budget():
    linear = nn.Linear(64, 32, bias=False)
    x = torch.randn(8, 64)
    reference = linear(x)
    error = (FP8Linear(linear)(x) - reference).abs().mean() / reference.abs().mean()
    assert error < 0.05, f"relative error {error:.4f} exceeds the e4m3 budget"


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="quantization scope"):
        quantize_(_model(), "int8", ASSIGNMENT[GROUP])

"""`CustomDeepseekV2Moe` stores experts as individual modules, keyed by global id.

Two properties, and the whole refactor rests on them:

  - **Names match the checkpoint.** Every parameter is called exactly what stock
    `DeepseekV2MoE` calls it, so `from_pretrained` and `load_state_dict` fill a
    partial model with no key translation — which is what retired
    `convert_full_to_partial_model`, `_apply_pretrained_tensor_to_partial` and
    both streaming loaders. The previous layout stored a layer's experts in two
    fused 3D tensors (`experts.gate_up_proj`, `experts.down_proj`) that matched
    no checkpoint key at all.

  - **Numerics are unchanged.** `_fused_reference` below is the pre-refactor
    dispatch — one fused `[gate; up]` matmul per expert against a stacked
    tensor — and `moe_infer` must match it exactly. Only the dispatch changed,
    so that is all the reference re-implements; routing is called from the
    module itself.

Run with `python -m pytest connito/test/test_per_expert_moe.py`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Config,
    DeepseekV2MoE,
)

from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Moe

HIDDEN, INTER, NUM_EXPERTS, TOP_K, LAYER_ID = 32, 16, 8, 2, 1
TRAINABLE = [1, 5]
HELPER = [3, 6]
MATERIALISED = sorted(TRAINABLE + HELPER)
ASSIGNMENT = {
    0: {LAYER_ID: [(0, 1), (1, 5)]},
    2: {LAYER_ID: [(0, 3), (1, 6)]},
}


def _det(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Deterministic fill — no RNG, so it survives any torch version."""
    count = 1
    for dim in shape:
        count *= dim
    return (torch.linspace(-1.0, 1.0, steps=count, dtype=torch.float32) * (0.1 + 0.01 * seed)).reshape(shape)


def _config(routing_mode: str = "masked_topk") -> DeepseekV2Config:
    cfg = DeepseekV2Config(
        hidden_size=HIDDEN, intermediate_size=4 * HIDDEN, moe_intermediate_size=INTER,
        num_hidden_layers=2, num_attention_heads=4, n_routed_experts=NUM_EXPERTS,
        n_shared_experts=1, num_experts_per_tok=TOP_K, first_k_dense_replace=1,
        topk_method="greedy", n_group=1, topk_group=1, vocab_size=64,
    )
    cfg.num_experts = NUM_EXPERTS
    cfg.full = False
    cfg.expert_group_assignment = ASSIGNMENT
    cfg.group_ids_trainable = [0]
    cfg.group_ids_helper = [2]
    cfg.routing_mode = routing_mode
    return cfg


def _weights() -> dict[str, torch.Tensor]:
    """Per-expert weights in checkpoint form."""
    weights = {"gate.weight": _det((NUM_EXPERTS, HIDDEN), 99)}
    for position, expert_id in enumerate(MATERIALISED):
        weights[f"experts.{expert_id}.gate_proj.weight"] = _det((INTER, HIDDEN), 3 * position + 0)
        weights[f"experts.{expert_id}.up_proj.weight"] = _det((INTER, HIDDEN), 3 * position + 1)
        weights[f"experts.{expert_id}.down_proj.weight"] = _det((HIDDEN, INTER), 3 * position + 2)
    weights["shared_experts.gate_proj.weight"] = _det((INTER, HIDDEN), 70)
    weights["shared_experts.up_proj.weight"] = _det((INTER, HIDDEN), 71)
    weights["shared_experts.down_proj.weight"] = _det((HIDDEN, INTER), 72)
    return weights


# ── names match the checkpoint ───────────────────────────────────────────────
def test_parameter_names_are_a_subset_of_stock():
    """A partial model declares a strict subset of stock's parameters.

    This is the mechanism, not a nicety: the subset is chosen by *which experts
    are declared*, so name identity is the only thing making the load work.
    """
    cfg = _config()
    mine = set(CustomDeepseekV2Moe(cfg, layer_id=LAYER_ID).state_dict())
    stock = set(DeepseekV2MoE(cfg).state_dict())

    assert mine <= stock, sorted(mine - stock)
    assert mine  # a vacuous subset would pass the line above


def test_only_the_assigned_and_helper_experts_are_materialised():
    moe = CustomDeepseekV2Moe(_config(), layer_id=LAYER_ID)
    assert sorted(int(k) for k in moe.experts) == MATERIALISED
    assert moe.trainable_ids.tolist() == TRAINABLE
    assert moe.helper_ids.tolist() == HELPER


def test_expert_weights_load_by_name_with_no_translation():
    moe = CustomDeepseekV2Moe(_config(), layer_id=LAYER_ID)
    weights = _weights()

    missing, unexpected = moe.load_state_dict(weights, strict=False)

    assert not unexpected
    for name, expected in weights.items():
        assert torch.equal(moe.state_dict()[name], expected), name


def test_experts_outside_the_group_are_rejected_as_unexpected():
    """The other 48 experts of a real checkpoint land here — as unexpected keys
    that nothing claims, which is exactly how the subset gets selected."""
    moe = CustomDeepseekV2Moe(_config(), layer_id=LAYER_ID)
    foreign = "experts.7.gate_proj.weight"

    _, unexpected = moe.load_state_dict({foreign: _det((INTER, HIDDEN), 1)}, strict=False)

    assert unexpected == [foreign]


# ── numerics are unchanged ───────────────────────────────────────────────────
def _fused_reference(moe, hidden_states, top_k_index, top_k_weights, weights):
    """The pre-refactor dispatch, verbatim in shape: fuse gate and up into one
    `[2I, H]` tensor, one `F.linear` per expert, chunk the result."""
    final = torch.zeros_like(hidden_states)
    mask = F.one_hot(top_k_index, num_classes=NUM_EXPERTS).permute(2, 1, 0)

    for expert_id in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        expert_id = expert_id[0].item()
        if expert_id not in MATERIALISED:
            continue
        slot, token = torch.where(mask[expert_id])
        fused = torch.cat(
            [weights[f"experts.{expert_id}.gate_proj.weight"],
             weights[f"experts.{expert_id}.up_proj.weight"]], dim=0,
        )
        gate, up = F.linear(hidden_states[token], fused).chunk(2, dim=-1)
        out = F.linear(F.silu(gate) * up, weights[f"experts.{expert_id}.down_proj.weight"])
        final.index_add_(0, token, (out * top_k_weights[token, slot, None]).to(final.dtype))

    return final


@pytest.mark.parametrize("routing_mode", ["masked_topk", "natural_with_fallback"])
def test_dispatch_matches_the_fused_implementation_exactly(routing_mode):
    """`torch.equal`, not `allclose`: `cat([gate, up]) @ x` and two separate
    matmuls read the same rows in the same order, so the results are bit-identical
    and anything short of equality is a real change."""
    weights = _weights()
    moe = CustomDeepseekV2Moe(_config(routing_mode), layer_id=LAYER_ID).float().eval()
    _, unexpected = moe.load_state_dict(weights, strict=False)
    assert not unexpected

    hidden = _det((2, 5, HIDDEN), 7).view(-1, HIDDEN)
    with torch.no_grad():
        logits = F.linear(hidden.float(), moe.gate.weight.float()).view(2, 5, NUM_EXPERTS)
        index, routing_weights = moe.route_tokens_to_experts(logits)
        actual = moe.moe_infer(hidden, index, routing_weights)
        expected = _fused_reference(moe, hidden, index, routing_weights, weights)

    assert torch.equal(actual, expected)

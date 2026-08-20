"""`freeze_parameters` trains the assigned experts and freezes everything else.

The helper group is the case worth pinning. Experts used to share one stacked
tensor per layer and `requires_grad` is per-tensor, so a helper expert could not
be frozen independently of an assigned one — on the live topology the miner
optimized 419 experts where the assignment asks for 187. Per-expert modules make
the distinction expressible, and these tests assert it holds.

The freeze is decided by parameter names and the assignment, so the model is
built at a tiny hidden size. CPU only, nothing is downloaded.

Run with `python -m pytest connito/test/test_freeze_parameters.py`.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Config

from connito.shared.expert_manager import get_layer_expert_id
from connito.shared.model import freeze_parameters
from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeekSeekMoE

HIDDEN, INTER, NUM_EXPERTS, TOP_K, LAYER_ID = 32, 16, 8, 2, 1
GROUP, HELPER_GROUP = 0, 2
ASSIGNED = [1, 5]
HELPER = [3, 6]
# (my_expert_id, org_expert_id) — the freeze must key off the global id on the
# right, not the local slot on the left.
ASSIGNMENT = {
    GROUP: {LAYER_ID: [(0, 1), (1, 5)]},
    HELPER_GROUP: {LAYER_ID: [(0, 3), (1, 6)]},
}


class StubExpertManager:
    """`freeze_parameters` reads exactly one attribute off the manager."""

    expert_group_assignment = ASSIGNMENT


@pytest.fixture(scope="module")
def frozen_model():
    cfg = DeepseekV2Config(
        hidden_size=HIDDEN, intermediate_size=4 * HIDDEN, moe_intermediate_size=INTER,
        num_hidden_layers=2, num_attention_heads=4, n_routed_experts=NUM_EXPERTS,
        n_shared_experts=1, num_experts_per_tok=TOP_K, first_k_dense_replace=1,
        topk_method="greedy", n_group=1, topk_group=1, vocab_size=64,
    )
    cfg.num_experts = NUM_EXPERTS
    cfg.full = False
    cfg.expert_group_assignment = ASSIGNMENT
    cfg.group_ids_trainable = [GROUP]
    cfg.group_ids_helper = [HELPER_GROUP]
    cfg.routing_mode = "masked_topk"

    # The full model, not a bare `CustomDeepseekV2Moe`: `get_layer_expert_id`
    # needs the `model.layers.{L}.mlp.experts.{N}` prefix to read the ids.
    model = CustomDeekSeekMoE(cfg)
    freeze_parameters(model=model, expert_manager=StubExpertManager(), expert_group_id=GROUP)
    return model


def _expert_params(model) -> dict[int, list[tuple[str, torch.nn.Parameter]]]:
    """Routed-expert params grouped by global expert id."""
    by_id: dict[int, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, param in model.named_parameters():
        _, expert_id = get_layer_expert_id(name)
        if expert_id is not None:
            by_id.setdefault(expert_id, []).append((name, param))
    return by_id


def test_assigned_experts_are_trainable(frozen_model):
    by_id = _expert_params(frozen_model)
    for expert_id in ASSIGNED:
        assert expert_id in by_id, f"expert {expert_id} was not materialised"
        for name, param in by_id[expert_id]:
            assert param.requires_grad, f"{name} should be trainable"


def test_helper_experts_are_frozen(frozen_model):
    """The case the stacked layout could not express.

    A helper expert is a routed-expert parameter that is *not* in this group's
    assignment. It must come out frozen.
    """
    by_id = _expert_params(frozen_model)
    for expert_id in HELPER:
        assert expert_id in by_id, f"helper expert {expert_id} was not materialised"
        for name, param in by_id[expert_id]:
            assert not param.requires_grad, f"{name} is a helper expert and must be frozen"


def test_nothing_outside_the_assignment_is_trainable(frozen_model):
    """Backbone, gate and `shared_experts` all stay frozen."""
    leaked = []
    for name, param in frozen_model.named_parameters():
        if not param.requires_grad:
            continue
        _, expert_id = get_layer_expert_id(name)
        if expert_id is None or expert_id not in ASSIGNED:
            leaked.append(name)
    assert not leaked, f"trainable params outside the assignment: {leaked[:5]}"


def test_trainable_tensor_count_matches_the_assignment(frozen_model):
    """Three projections per assigned expert, and nothing else."""
    trainable = [n for n, p in frozen_model.named_parameters() if p.requires_grad]
    assert len(trainable) == 3 * len(ASSIGNED)


def test_freeze_keys_off_the_global_id_not_the_local_slot(frozen_model):
    """`my_expert_id` 0 and 1 map to global 1 and 5 here.

    A freeze that read the left half of the assignment pair would train experts
    0 and 1 instead — and expert 1 is assigned, so the mistake would be half
    invisible. Expert 0 is materialised only if some group claims it; assert on
    the ids that distinguish the two readings.
    """
    by_id = _expert_params(frozen_model)
    # Global 5 is assigned but is not a local slot index in this fixture.
    assert all(p.requires_grad for _, p in by_id[5])
    # Global 3 is local slot 0 of the helper group — frozen despite the 0.
    assert all(not p.requires_grad for _, p in by_id[3])

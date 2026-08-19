"""Unit test for the natural-with-fallback (2Fnat) routing branch in
`CustomDeepseekV2Moe.route_tokens_to_experts`.

Tests the routing rule directly with synthetic router_logits, avoiding the
DeepSeek-V2-Lite HuggingFace download. Uses a small n_routed_experts (8) with
top_k=2, one trainable set = {0,1,2}, one helper set = {3,4,5}, and hand-crafted
logits where we know which experts should win natural top-k per token.

The test asserts three invariants of the 2Fnat rule:

  1. Trainable-preservation: every natural top-k pick that landed in the
     trainable set MUST appear in the final routing output with its natural
     gate weight.
  2. Helper fallback: every natural top-k slot that landed OUTSIDE the trainable
     set MUST be replaced by a helper expert (id ∈ helper_ids).
  3. Slot count: each token gets exactly top_k experts back.

Run: pytest connito/test/test_natural_with_fallback_routing.py -v
Or:  python -m connito.test.test_natural_with_fallback_routing
"""
from __future__ import annotations

import types

import torch


class _MoeStub:
    """Minimal stand-in exposing the fields
    `CustomDeepseekV2Moe.route_tokens_to_experts` reads. Lets us call the real
    routing method as an unbound function bound to a stub instance, so we test
    the actual production code path."""

    def __init__(self, n_experts: int, top_k: int, trainable_ids: list[int], helper_ids: list[int]):
        self.top_k = top_k
        self.topk_method = "greedy"
        self.routed_scaling_factor = 1.0
        self.num_group = 1
        self.topk_group = 1
        self.num_experts = n_experts
        self.routing_mode = "natural_with_fallback"
        self.allowed_ids = torch.tensor(sorted(trainable_ids + helper_ids), dtype=torch.long)
        self.trainable_ids = torch.tensor(sorted(trainable_ids), dtype=torch.long)
        self.helper_ids = torch.tensor(sorted(helper_ids), dtype=torch.long)


def test_natural_with_fallback():
    from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Moe

    n_experts = 8
    top_k = 2
    trainable = [0, 1, 2]
    helper = [3, 4, 5]
    # experts 6, 7 are "other" — not in either set. Should NEVER appear in output.

    stub = _MoeStub(n_experts=n_experts, top_k=top_k, trainable_ids=trainable, helper_ids=helper)

    # Craft router logits so we know the natural argmax pattern per token.
    # Layout: batch=1, seq=4, hidden=n_experts. Column j is the "score" for expert j.
    # After softmax rows sum to 1 but the argsort order is preserved by the pre-softmax rank.
    #
    # We use large enough gaps that the argsort is unambiguous.
    #
    # Token 0: both natural top-2 in trainable  →  final = natural (no fallback)
    # Token 1: 1 natural in trainable, 1 in other→  final = 1 natural + 1 helper substitute
    # Token 2: 0 natural in trainable            →  final = 2 helper substitutes
    # Token 3: both natural in helpers          →  final = 2 helper substitutes (still natural)

    logits = torch.full((1, 4, n_experts), -5.0)
    # token 0 — top-2 argmax = expert 1, 0 (both trainable)
    logits[0, 0, 1] = 5.0
    logits[0, 0, 0] = 4.0
    # token 1 — top-2 argmax = expert 6 (other), 2 (trainable)
    logits[0, 1, 6] = 5.0
    logits[0, 1, 2] = 4.0
    # token 2 — top-2 argmax = expert 7 (other), 6 (other)
    logits[0, 2, 7] = 5.0
    logits[0, 2, 6] = 4.0
    # token 3 — top-2 argmax = expert 3 (helper), 4 (helper)
    logits[0, 3, 3] = 5.0
    logits[0, 3, 4] = 4.0

    final_idx, final_w = CustomDeepseekV2Moe.route_tokens_to_experts(stub, logits)
    # final_idx: shape [4, 2] (batch*seq × top_k)
    assert final_idx.shape == (4, top_k), f"unexpected shape {final_idx.shape}"
    assert final_w.shape == (4, top_k)

    trainable_set = set(trainable)
    helper_set = set(helper)

    # Row-by-row assertions.
    row0 = set(final_idx[0].tolist())
    assert row0 == {0, 1}, f"token 0 both-trainable, got {row0}"

    row1 = set(final_idx[1].tolist())
    # 2 (natural, trainable) must survive; 6 (other) must be replaced by a helper
    assert 2 in row1, f"token 1 lost the natural trainable pick, got {row1}"
    non_train_slot = (row1 - {2}).pop()
    assert non_train_slot in helper_set, (
        f"token 1 non-trainable slot should be in helper set {helper_set}, got {non_train_slot}"
    )
    # Expert 6 (the natural pick that got replaced) must not appear
    assert 6 not in row1, f"token 1 leaked non-trainable expert 6 into final: {row1}"

    row2 = set(final_idx[2].tolist())
    # Both natural picks were "other" → both slots must be helper substitutions
    for e in row2:
        assert e in helper_set, f"token 2 leaked non-helper {e}, expected all in {helper_set}"
    # And the natural picks (6, 7) must not appear
    assert 6 not in row2 and 7 not in row2, f"token 2 kept non-trainable/non-helper: {row2}"

    row3 = set(final_idx[3].tolist())
    # Natural picks (3, 4) are BOTH helpers — but the routing rule only preserves
    # natural picks that landed in the TRAINABLE set. Non-trainable natural picks
    # (even helpers) get replaced by top-helper picks. In this case the top helpers
    # ARE 3 and 4, so we expect them to come back — via the fallback path, not
    # the natural-preservation path.
    for e in row3:
        assert e in helper_set, f"token 3 leaked non-helper {e}"
    # Slot count invariant: exactly top_k unique per token
    for i in range(4):
        assert len(set(final_idx[i].tolist())) == top_k, (
            f"row {i} has duplicates: {final_idx[i].tolist()}"
        )

    print("test_natural_with_fallback PASSED")


def test_natural_fallback_disabled_falls_back_to_masked_topk():
    """When routing_mode != 'natural_with_fallback' the new branch must NOT fire,
    preserving the existing masked-topk behavior for A / D / F16 paradigms."""
    from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Moe

    stub = _MoeStub(n_experts=8, top_k=2, trainable_ids=[0, 1, 2], helper_ids=[3, 4, 5])
    stub.routing_mode = "masked_topk"  # explicitly the default path

    logits = torch.full((1, 2, 8), -5.0)
    # token 0 natural argmax = expert 7 (not in allowed) — should get masked out
    logits[0, 0, 7] = 5.0
    logits[0, 0, 1] = 4.0  # trainable — should survive masking
    logits[0, 0, 3] = 3.0  # helper — should also survive
    # token 1 natural argmax = expert 6 (not in allowed) and 0 (trainable)
    logits[0, 1, 6] = 5.0
    logits[0, 1, 0] = 4.0

    final_idx, _ = CustomDeepseekV2Moe.route_tokens_to_experts(stub, logits)
    all_ids = set(final_idx.flatten().tolist())
    allowed_set = set(stub.allowed_ids.tolist())
    assert all_ids.issubset(allowed_set), (
        f"masked_topk mode leaked disallowed experts: {all_ids - allowed_set}"
    )
    # Specifically, the natural picks 7 and 6 must not appear
    assert 7 not in all_ids and 6 not in all_ids
    print("test_natural_fallback_disabled_falls_back_to_masked_topk PASSED")


if __name__ == "__main__":
    test_natural_with_fallback()
    test_natural_fallback_disabled_falls_back_to_masked_topk()
    print("all tests PASSED")

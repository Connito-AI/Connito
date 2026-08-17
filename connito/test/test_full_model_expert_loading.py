"""Loading pretrained routed experts into a full model.

The full-model path used to be a bare `load_state_dict(strict=False)` against
the HuggingFace checkpoint. That silently loaded nothing into the routed
experts: the checkpoint names them `...experts.{N}.{gate,up,down}_proj.weight`
while `CustomDeepseekV2Experts` stores each layer's experts stacked with
gate/up fused, under `...experts.{N}.gate_up_proj`. Neither side matched, and
`strict=False` swallowed both halves of the mismatch.

These cover the two pieces the fix rests on, without needing a GPU or the
30 GB checkpoint:

  - `assignments_from_expert_modules` reads the local->global expert mapping
    off the model, which is what a full build must do (it has no group
    assignment to derive one from).
  - `_apply_pretrained_tensor_to_partial` routes HuggingFace-named per-expert
    tensors into the fused stacked destination.

The end-to-end regression needs real weights and lives in
`tools/quantization/gpu_full_load_check.py`: a correctly loaded full model
scores ~1.45 on the seeded eval batches, against 9.956 when the experts are
left at their random init.

Run with `python -m pytest connito/test/test_full_model_expert_loading.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from connito.shared.modeling.custom_deepseek_v2_lite import (
    CustomDeepseekV2Experts,
    _apply_pretrained_tensor_to_partial,
    assignments_from_expert_modules,
)

LAYER_ID = 1
PREFIX = f"model.layers.{LAYER_ID}.mlp.experts."
# Deliberately not 0..N-1: a full build should be the identity, but the helper
# must not bake that in — it reads whatever layout the module actually holds.
GLOBAL_IDS = [0, 1, 2, 3]


class _Cfg:
    """`num_experts` is the *global* routed-expert count, so it has to be large
    enough to index every global id a test hands out."""

    hidden_size = 16
    moe_intermediate_size = 12
    hidden_act = "silu"
    initializer_range = 0.02
    first_k_dense_replace = 1
    num_hidden_layers = 4

    def __init__(self, num_experts: int = 4) -> None:
        self.n_routed_experts = num_experts
        self.num_experts = num_experts


def _experts(expert_indices=None, num_experts: int = 4) -> CustomDeepseekV2Experts:
    torch.manual_seed(0)
    return CustomDeepseekV2Experts(
        _Cfg(num_experts),
        expert_indices=list(GLOBAL_IDS if expert_indices is None else expert_indices),
    )


def _prefixed_state(module: CustomDeepseekV2Experts) -> dict[str, torch.Tensor]:
    """`state_dict()` re-keyed to a full model's qualified names.

    The entries stay live views onto the stacked parameters, which is what lets
    the streaming loaders write through them.
    """
    return {f"{PREFIX}{key}": value for key, value in module.state_dict().items()}


# ── the mapping a full build has to derive from itself ───────────────────────
class _Mlp(nn.Module):
    def __init__(self, expert_indices, num_experts: int = 4) -> None:
        super().__init__()
        self.experts = _experts(expert_indices, num_experts)


class _Layer(nn.Module):
    def __init__(self, expert_indices, num_experts: int = 4) -> None:
        super().__init__()
        self.mlp = _Mlp(expert_indices, num_experts)


class _Model(nn.Module):
    """Just enough hierarchy to produce `model.layers.{L}.mlp.experts` names."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        # Layer 0 is dense (`first_k_dense_replace=1`), so it holds no experts.
        self.model.layers = nn.ModuleList([nn.Module(), _Layer(GLOBAL_IDS)])


def test_assignments_are_read_off_the_model():
    assignments = assignments_from_expert_modules(_Model())

    assert set(assignments) == {LAYER_ID}
    assert assignments[LAYER_ID] == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_assignments_follow_a_non_identity_layout():
    """The helper must report the module's real slots, not assume 0..N-1."""

    class _Sparse(_Model):
        def __init__(self) -> None:
            super().__init__()
            self.model.layers = nn.ModuleList(
                [nn.Module(), _Layer([2, 5, 7, 9], num_experts=12)]
            )

    assignments = assignments_from_expert_modules(_Sparse())

    assert assignments[LAYER_ID] == [(0, 2), (1, 5), (2, 7), (3, 9)]


def test_dense_layers_contribute_no_assignment():
    model = _Model()
    assert 0 not in assignments_from_expert_modules(model)


# ── the translation the full path was skipping ───────────────────────────────
def test_huggingface_per_expert_keys_land_in_the_fused_stack():
    module = _experts()
    partial_state = _prefixed_state(module)
    assignments = {LAYER_ID: [(i, g) for i, g in enumerate(GLOBAL_IDS)]}
    counts = {"full": 0, "sliced": 0}
    buf: dict[str, dict[str, torch.Tensor]] = {}

    torch.manual_seed(7)
    inter, hidden = _Cfg.moe_intermediate_size, _Cfg.hidden_size
    source = {}
    for global_id in GLOBAL_IDS:
        source[global_id] = {
            "gate_proj": torch.randn(inter, hidden),
            "up_proj": torch.randn(inter, hidden),
            "down_proj": torch.randn(hidden, inter),
        }

    for global_id, projections in source.items():
        for name, tensor in projections.items():
            _apply_pretrained_tensor_to_partial(
                key=f"{PREFIX}{global_id}.{name}.weight",
                source_tensor=tensor,
                partial_state=partial_state,
                assignments=assignments,
                loaded_counts=counts,
                gate_up_buf=buf,
            )

    # Every expert landed, and nothing is left waiting for its pair.
    assert counts["sliced"] == len(GLOBAL_IDS) * 2  # one gate+up fusion, one down
    assert buf == {}

    for local_idx, global_id in enumerate(GLOBAL_IDS):
        want = source[global_id]
        got_gate = module.gate_up_proj.data[local_idx, :inter, :]
        got_up = module.gate_up_proj.data[local_idx, inter:, :]
        got_down = module.down_proj.data[local_idx]
        assert torch.equal(got_gate, want["gate_proj"]), f"gate mismatch, expert {global_id}"
        assert torch.equal(got_up, want["up_proj"]), f"up mismatch, expert {global_id}"
        assert torch.equal(got_down, want["down_proj"]), f"down mismatch, expert {global_id}"


def test_gate_and_up_are_not_written_until_both_arrive():
    """The fusion is deferred; a lone `gate_proj` must not land half a tensor."""
    module = _experts()
    before = module.gate_up_proj.data.clone()
    partial_state = _prefixed_state(module)
    counts = {"full": 0, "sliced": 0}
    buf: dict[str, dict[str, torch.Tensor]] = {}

    _apply_pretrained_tensor_to_partial(
        key=f"{PREFIX}0.gate_proj.weight",
        source_tensor=torch.randn(_Cfg.moe_intermediate_size, _Cfg.hidden_size),
        partial_state=partial_state,
        assignments={LAYER_ID: [(0, 0)]},
        loaded_counts=counts,
        gate_up_buf=buf,
    )

    assert counts["sliced"] == 0
    assert torch.equal(module.gate_up_proj.data, before)
    assert list(buf) == [f"{PREFIX}0."]


def test_experts_are_untouched_without_the_translation():
    """Pins the original bug: the raw checkpoint keys match nothing.

    This is what `load_state_dict(strict=False)` saw — every incoming key
    unexpected, every destination key missing, and no error either way.
    """
    module = _experts()
    checkpoint_keys = {
        f"{PREFIX}{g}.{name}.weight"
        for g in GLOBAL_IDS
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    destination_keys = set(_prefixed_state(module))

    assert checkpoint_keys.isdisjoint(destination_keys)
    incompatible = module.load_state_dict(
        {k: torch.zeros(1) for k in checkpoint_keys}, strict=False
    )
    assert len(incompatible.unexpected_keys) == len(checkpoint_keys)

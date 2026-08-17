"""Per-slice fp8 quantization of the stacked expert tensors.

The load-bearing claim is that splitting expert storage in two — fp8 for the
frozen helper slices, a real `nn.Parameter` for the trainable ones — is
invisible from outside the module. `state_dict()` must still emit the same
per-expert keys with the same shapes and dtypes, because those keys feed the
submission hash; the trainable slices must still receive gradients, because they
are what the miner is training.

CPU only, tiny synthetic config: no GPU and no model download.
Run with `python -m pytest connito/test/test_expert_slice_quantization.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Experts
from connito.shared.modeling.quantization import (
    FP8_DTYPE,
    is_quantized,
    quantize_frozen_expert_slices_,
)

TRAINABLE_IDS = [1, 5]
HELPER_IDS = [3, 7]
LOCAL_IDS = sorted(TRAINABLE_IDS + HELPER_IDS)  # local slots 0..3 -> globals 1,3,5,7
HELPER_LOCAL = [LOCAL_IDS.index(g) for g in HELPER_IDS]


class _Cfg:
    n_routed_experts = 8
    num_experts = 8
    hidden_size = 16
    moe_intermediate_size = 12
    hidden_act = "silu"
    initializer_range = 0.02
    first_k_dense_replace = 1
    num_hidden_layers = 4


def _experts(seed: int = 0) -> CustomDeepseekV2Experts:
    torch.manual_seed(seed)
    return CustomDeepseekV2Experts(_Cfg(), expert_indices=list(LOCAL_IDS))


def _routing() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    hidden = torch.randn(6, _Cfg.hidden_size)
    top_k_index = torch.tensor([[1, 3], [3, 5], [5, 7], [7, 1], [1, 5], [3, 7]])
    top_k_weights = torch.full((6, 2), 0.5)
    return hidden, top_k_index, top_k_weights


def _is_helper_key(key: str) -> bool:
    return int(key.split(".")[-2]) in HELPER_IDS


# ── the hash-contract invariant ───────────────────────────────────────────────
def test_partial_quantization_leaves_state_dict_shape_identical():
    reference = {k: v.clone() for k, v in _experts().state_dict().items()}
    module = _experts()
    module.quantize_(local_indices=HELPER_LOCAL)
    produced = module.state_dict()

    assert sorted(produced) == sorted(reference)
    for key, ref in reference.items():
        assert produced[key].shape == ref.shape
        assert produced[key].dtype == ref.dtype


def test_trainable_slices_survive_bit_exact():
    """Quantizing helper slices must not perturb the ones being trained."""
    reference = {k: v.clone() for k, v in _experts().state_dict().items()}
    module = _experts()
    module.quantize_(local_indices=HELPER_LOCAL)
    produced = module.state_dict()

    for key, ref in reference.items():
        if _is_helper_key(key):
            assert not torch.equal(produced[key], ref)  # fp8 really was applied
        else:
            assert torch.equal(produced[key], ref)


def test_helper_slices_are_stored_in_fp8_and_trainable_ones_are_not():
    module = _experts()
    module.quantize_(local_indices=HELPER_LOCAL)

    assert module.gate_up_proj_fp8.dtype == FP8_DTYPE
    assert module.gate_up_proj_fp8.shape[0] == len(HELPER_LOCAL)
    # The surviving Parameter keeps its original name so `freeze_parameters`,
    # `named_parameters()` and the optimizer all still find it.
    kept = dict(module.named_parameters())
    assert set(kept) == {"gate_up_proj", "down_proj"}
    assert kept["gate_up_proj"].shape[0] == len(LOCAL_IDS) - len(HELPER_LOCAL)


# ── training must still work ──────────────────────────────────────────────────
def test_gradients_reach_the_trainable_slices_only():
    module = _experts()
    module.quantize_(local_indices=HELPER_LOCAL)

    module(*_routing()).sum().backward()

    grads = {name: p.grad for name, p in module.named_parameters()}
    assert set(grads) == {"gate_up_proj", "down_proj"}
    assert all(g is not None for g in grads.values())
    # The fp8 store is a buffer: no gradient can flow to it by construction.
    assert module.gate_up_proj_fp8.grad is None


def test_forward_only_perturbs_the_quantized_experts():
    dense, quantized = _experts(), _experts()
    quantized.quantize_(local_indices=HELPER_LOCAL)
    batch = _routing()

    delta = (dense(*batch) - quantized(*batch)).abs().max().item()
    assert delta > 0.0  # quantization is actually in the forward path
    assert delta < 0.05  # ...and only two of four experts moved


# ── round-tripping through load_state_dict ────────────────────────────────────
def test_load_state_dict_routes_slices_to_both_stores():
    source = {k: v.clone() for k, v in _experts(seed=7).state_dict().items()}
    module = _experts(seed=0)
    module.quantize_(local_indices=HELPER_LOCAL)

    incompatible = module.load_state_dict(source, strict=False)
    assert list(incompatible.missing_keys) == []
    assert list(incompatible.unexpected_keys) == []

    produced = module.state_dict()
    for key, want in source.items():
        if _is_helper_key(key):
            assert torch.allclose(produced[key], want, atol=0.05)
        else:
            assert torch.equal(produced[key], want)


def test_quantizing_every_slice_still_drops_the_parameter():
    """The validator path (`local_indices=None`) must be unchanged."""
    reference = sorted(_experts().state_dict())
    module = _experts()
    module.quantize_()

    assert list(module.named_parameters()) == []
    assert sorted(module.state_dict()) == reference


# ── the driver that picks helper slices ───────────────────────────────────────
class _FakeMoe(nn.Module):
    """Duck-typed stand-in: the driver needs `.experts` and `.helper_ids` only."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = _experts()
        self.register_buffer("trainable_ids", torch.tensor(TRAINABLE_IDS), persistent=False)
        self.register_buffer("helper_ids", torch.tensor(HELPER_IDS), persistent=False)


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeMoe(), _FakeMoe()])


def test_driver_quantizes_helper_slices_in_every_layer():
    model = _FakeModel()
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    converted = quantize_frozen_expert_slices_(model)

    assert converted == ["layers.0.experts", "layers.1.experts"]
    assert is_quantized(model)
    produced = model.state_dict()
    assert sorted(produced) == sorted(reference)
    for key, ref in reference.items():
        if _is_helper_key(key):
            assert not torch.equal(produced[key], ref)
        else:
            assert torch.equal(produced[key], ref)


def test_driver_is_a_noop_without_helper_ids():
    model = _FakeModel()
    for layer in model.layers:
        layer.helper_ids = torch.zeros(0, dtype=torch.long)

    assert quantize_frozen_expert_slices_(model) == []
    assert not is_quantized(model)

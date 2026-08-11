"""CPU tests for runtime int8 quantization.

Everything here runs on CPU against tiny synthetic modules — no GPU, no 16 GB
model download. Run with:

    python3 -m pytest connito/test/test_quantization.py
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from connito.shared.helper import get_model_hash, infer_storage_dtype
from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Experts
from connito.shared.modeling.quantization import (
    DEFAULT_LINEAR_DENYLIST,
    Int8Linear,
    dequantize_last_dim,
    is_denied,
    is_quantized,
    quantize_last_dim,
    quantize_linear_modules_,
    quantize_model_,
    require_not_quantized,
    state_dict_shapes,
)

# Per-output-row symmetric int8 has a worst-case relative error of one half-step
# in 127, i.e. ~0.4%. Anything materially above that means the scale axis or the
# fp32 accumulation was got wrong.
INT8_REL_TOLERANCE = 0.005


def _expert_config(n_experts: int = 8, hidden: int = 32, inter: int = 16):
    return types.SimpleNamespace(
        n_routed_experts=n_experts,
        num_experts=n_experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        hidden_act="silu",
        first_k_dense_replace=1,
        num_hidden_layers=4,
        initializer_range=0.02,
    )


def _max_rel_error(actual: dict, reference: dict) -> float:
    worst = 0.0
    for key, ref in reference.items():
        magnitude = ref.float().abs().max().item()
        if magnitude == 0:
            continue
        worst = max(worst, (actual[key].float() - ref.float()).abs().max().item() / magnitude)
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("shape", [(16, 32), (4, 16, 32)])
def test_quantize_round_trip_is_within_int8_bound(shape):
    """Scales are per *last-dim row*, for 2D linears and 3D stacked experts alike."""
    weight = torch.randn(*shape, dtype=torch.float32)
    values, scale = quantize_last_dim(weight)

    assert values.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert tuple(scale.shape) == shape[:-1]

    restored = dequantize_last_dim(values, scale, torch.float32)
    rel = (restored - weight).abs().max().item() / weight.abs().max().item()
    assert rel < INT8_REL_TOLERANCE


def test_quantize_handles_all_zero_rows():
    """An all-zero row must not produce a zero scale and divide by it."""
    weight = torch.zeros(4, 8)
    weight[1] = torch.randn(8)
    values, scale = quantize_last_dim(weight)
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert torch.isfinite(dequantize_last_dim(values, scale, torch.float32)).all()


# ─────────────────────────────────────────────────────────────────────────────
# Int8Linear
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bias", [True, False])
def test_int8_linear_state_dict_is_transparent(bias):
    """The whole design rests on this: same keys, shapes and dtypes as nn.Linear."""
    linear = nn.Linear(64, 32, bias=bias).to(torch.float16)
    reference = {k: v.clone() for k, v in linear.state_dict().items()}

    quantized = Int8Linear.from_linear(linear).state_dict()

    assert sorted(quantized) == sorted(reference)
    for key, ref in reference.items():
        assert quantized[key].shape == ref.shape
        assert quantized[key].dtype == ref.dtype
    assert _max_rel_error(quantized, reference) < INT8_REL_TOLERANCE


def test_int8_linear_buffers_are_non_persistent():
    """`persistent=False` is what keeps grafts symmetric in both directions."""
    quantized = Int8Linear.from_linear(nn.Linear(8, 4).to(torch.float16))
    keys = quantized.state_dict().keys()
    assert "weight_int8" not in keys
    assert "weight_scale" not in keys


def test_int8_linear_forward_matches_dense():
    linear = nn.Linear(64, 32).to(torch.float16)
    quantized = Int8Linear.from_linear(linear)
    x = torch.randn(4, 64, dtype=torch.float16)

    dense_out = linear(x).float()
    diff = (dense_out - quantized(x).float()).abs().max().item()
    assert diff / dense_out.abs().max().item() < 0.05


def test_int8_linear_graft_is_symmetric_both_ways():
    """fp16 -> quantized and quantized -> fp16 both load strictly and cleanly."""
    quantized = Int8Linear.from_linear(nn.Linear(64, 32).to(torch.float16))
    source = nn.Linear(64, 32).to(torch.float16)

    into_quantized = quantized.load_state_dict(source.state_dict(), strict=True)
    assert into_quantized.missing_keys == [] and into_quantized.unexpected_keys == []
    assert _max_rel_error(quantized.state_dict(), source.state_dict()) < INT8_REL_TOLERANCE

    plain = nn.Linear(64, 32).to(torch.float16)
    out_of_quantized = plain.load_state_dict(quantized.state_dict(), strict=True)
    assert out_of_quantized.missing_keys == [] and out_of_quantized.unexpected_keys == []


def test_int8_linear_reports_missing_weight_under_strict():
    quantized = Int8Linear.from_linear(nn.Linear(8, 4, bias=True).to(torch.float16))
    missing, unexpected = [], []
    quantized._load_from_state_dict({}, "", {}, True, missing, unexpected, [])
    assert "weight" in missing


def test_int8_linear_rejects_keep_vars():
    """Silently handing back a dequantized copy where a live view was asked for
    would turn an in-place write into a no-op."""
    quantized = Int8Linear.from_linear(nn.Linear(8, 4).to(torch.float16))
    with pytest.raises(RuntimeError, match="keep_vars"):
        quantized.state_dict(keep_vars=True)


def test_scale_survives_dtype_cast():
    """`Module.to(dtype=...)` casts floating buffers; the fp32 scale must not go
    with them, or the precision the fp32 scale exists to preserve is lost."""
    quantized = Int8Linear.from_linear(nn.Linear(8, 4).to(torch.float16))
    quantized.to(dtype=torch.bfloat16)
    assert quantized.weight_scale.dtype == torch.float32
    assert quantized.weight_int8.dtype == torch.int8


# ─────────────────────────────────────────────────────────────────────────────
# Module selection
# ─────────────────────────────────────────────────────────────────────────────
def _toy_model() -> nn.Module:
    model = nn.Module()
    model.lm_head = nn.Linear(8, 4)
    model.embed_tokens = nn.Embedding(10, 8)
    model.norm = nn.LayerNorm(8)
    attn = nn.Module()
    attn.q_proj = nn.Linear(8, 8)
    attn.kv_a_proj_with_mqa = nn.Linear(8, 8)
    attn.o_proj = nn.Linear(8, 8)
    mlp = nn.Module()
    mlp.gate = nn.Linear(8, 4)
    mlp.down_proj = nn.Linear(8, 8)
    model.attn = attn
    model.mlp = mlp
    return model.to(torch.float16)


def test_denylist_protects_router_head_and_kv_a_proj():
    model = _toy_model()
    model.requires_grad_(False)

    converted = quantize_linear_modules_(model)

    assert converted == ["attn.o_proj", "attn.q_proj", "mlp.down_proj"]
    # Non-Linear modules are excluded for free by the isinstance test.
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.mlp.gate, nn.Linear)


def test_denylist_matches_path_suffixes_not_substrings():
    """`mlp.gate` as a substring also matches `mlp.gate_proj`, which would
    silently exclude the dense MLPs the toggle is supposed to cover."""
    assert is_denied("model.layers.3.mlp.gate", DEFAULT_LINEAR_DENYLIST)
    assert is_denied("lm_head", DEFAULT_LINEAR_DENYLIST)
    assert is_denied("model.layers.3.self_attn.kv_a_proj_with_mqa", DEFAULT_LINEAR_DENYLIST)

    assert not is_denied("model.layers.3.mlp.gate_proj", DEFAULT_LINEAR_DENYLIST)
    assert not is_denied("model.layers.3.mlp.shared_experts.gate_proj", DEFAULT_LINEAR_DENYLIST)
    assert not is_denied("model.layers.3.self_attn.kv_b_proj", DEFAULT_LINEAR_DENYLIST)


def test_dense_mlp_gate_proj_is_quantized():
    """Regression for the substring bug: a `gate_proj` next to a router `gate`
    must still be converted."""
    model = nn.Module()
    mlp = nn.Module()
    mlp.gate = nn.Linear(8, 4)
    mlp.gate_proj = nn.Linear(8, 8)
    model.mlp = mlp
    model = model.to(torch.float16)
    model.requires_grad_(False)

    converted = quantize_linear_modules_(model)

    assert converted == ["mlp.gate_proj"]


def test_frozen_only_skips_trainable_weights():
    """A trainable weight turned into an int8 buffer would drop out of
    `named_parameters()` and therefore out of the optimizer, silently."""
    model = _toy_model()
    model.requires_grad_(False)
    model.attn.q_proj.weight.requires_grad_(True)

    converted = quantize_linear_modules_(model, frozen_only=True)

    assert "attn.q_proj" not in converted
    assert "attn.o_proj" in converted


def test_quantized_weights_leave_named_parameters_but_trainable_ones_stay():
    model = _toy_model()
    model.requires_grad_(False)
    model.mlp.down_proj.weight.requires_grad_(True)
    quantize_linear_modules_(model, frozen_only=True)

    names = dict(model.named_parameters())
    assert "mlp.down_proj.weight" in names
    assert "attn.o_proj.weight" not in names
    # ...but it is still reachable as a buffer, which is what keeps the miner's
    # post-setup finiteness guard covering it.
    assert "attn.o_proj.weight_int8" in dict(model.named_buffers())


# ─────────────────────────────────────────────────────────────────────────────
# Stacked experts
# ─────────────────────────────────────────────────────────────────────────────
def _experts(indices=(1, 3, 5)) -> CustomDeepseekV2Experts:
    return CustomDeepseekV2Experts(_expert_config(), expert_indices=list(indices)).to(torch.float16)


def test_experts_state_dict_is_transparent():
    experts = _experts()
    reference = {k: v.clone() for k, v in experts.state_dict().items()}

    experts.quantize_()
    quantized = experts.state_dict()

    assert sorted(quantized) == sorted(reference)
    for key, ref in reference.items():
        assert quantized[key].shape == ref.shape
        assert quantized[key].dtype == ref.dtype
    assert _max_rel_error(quantized, reference) < INT8_REL_TOLERANCE


def test_experts_graft_is_not_truncated_to_integers():
    """The corruption this guards: `.to(dtype=target_param.dtype)` where the
    target is now int8 silently rounds every incoming weight to an integer."""
    experts = _experts()
    experts.quantize_()
    source = _experts()

    result = experts.load_state_dict(source.state_dict(), strict=True)

    assert result.missing_keys == [] and result.unexpected_keys == []
    loaded = experts.state_dict()
    assert _max_rel_error(loaded, source.state_dict()) < INT8_REL_TOLERANCE
    sample = loaded["1.gate_up_proj"].float()
    assert not torch.all(sample == sample.round()), "weights were truncated to integers"


def test_experts_accept_a_full_stacked_tensor():
    """The `[global_experts, ...]` adaptation branch also has to route through
    the compute dtype rather than the storage dtype."""
    indices = [1, 3, 5]
    experts = _experts(indices)
    experts.quantize_()
    cfg = _expert_config()
    full = torch.randn(cfg.n_routed_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size, dtype=torch.float16)

    result = experts.load_state_dict({"gate_up_proj": full}, strict=False)

    assert result.unexpected_keys == []
    loaded = experts.state_dict()
    for global_idx in indices:
        expected = full[global_idx].float()
        actual = loaded[f"{global_idx}.gate_up_proj"].float()
        assert (actual - expected).abs().max().item() / expected.abs().max().item() < INT8_REL_TOLERANCE


def test_experts_stacked_key_is_written_exactly_once():
    """The overlay branch used to inject the stacked key *and* copy into the
    parameter, leaving super() to load it a second time."""
    experts = _experts()
    experts.quantize_()
    source = _experts()
    state = source.state_dict()

    missing, unexpected, errors = [], [], []
    experts._load_from_state_dict(dict(state), "", {}, True, missing, unexpected, errors)

    assert errors == []
    assert unexpected == []
    assert not any("gate_up_proj" == key for key in missing)


def test_experts_forward_matches_dense():
    experts = _experts()
    dense_state = {k: v.clone() for k, v in experts.state_dict().items()}
    hidden = torch.randn(6, 32, dtype=torch.float16)
    top_k_index = torch.randint(0, 8, (6, 2))
    top_k_weights = torch.rand(6, 2, dtype=torch.float16)

    dense_out = experts(hidden, top_k_index, top_k_weights).float()
    experts.quantize_()
    assert _max_rel_error(experts.state_dict(), dense_state) < INT8_REL_TOLERANCE
    quant_out = experts(hidden, top_k_index, top_k_weights).float()

    denominator = dense_out.abs().max().item()
    if denominator > 0:
        assert (dense_out - quant_out).abs().max().item() / denominator < 0.05


def test_experts_reject_keep_vars_when_quantized():
    experts = _experts()
    experts.quantize_()
    with pytest.raises(RuntimeError, match="keep_vars"):
        experts.state_dict(keep_vars=True)


# ─────────────────────────────────────────────────────────────────────────────
# Contracts the rest of the codebase depends on
# ─────────────────────────────────────────────────────────────────────────────
def test_model_hash_is_unchanged_by_quantization():
    """`serialize_torch_model_path` walks every key and has no int8 branch, so a
    quantized model must hash exactly as its dequantized state_dict does."""
    experts = _experts()
    experts.quantize_()

    quantized_hash = get_model_hash(experts.state_dict(), hex=True)
    reference_hash = get_model_hash(dict(experts.state_dict()), hex=True)

    assert quantized_hash == reference_hash
    # And the bytes are hashable at all — an int8 tensor reaching this would
    # change the digest silently rather than raise.
    assert isinstance(quantized_hash, str) and len(quantized_hash) > 0


def test_require_not_quantized_fires_only_when_quantized():
    model = _toy_model()
    model.requires_grad_(False)
    require_not_quantized(model, "before")  # no-op
    assert not is_quantized(model)

    quantize_linear_modules_(model)

    assert is_quantized(model)
    with pytest.raises(RuntimeError, match="int8"):
        require_not_quantized(model, "merge")


def test_state_dict_shapes_matches_state_dict_and_caches():
    model = _toy_model()
    model.requires_grad_(False)
    quantize_linear_modules_(model)

    shapes = state_dict_shapes(model)

    assert shapes == {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert state_dict_shapes(model) is shapes  # cached, not rebuilt per miner


def test_infer_storage_dtype_ignores_fp32_upcast_and_int8():
    model = _toy_model()
    model.requires_grad_(False)
    model.mlp.down_proj.weight.requires_grad_(True)
    model.mlp.down_proj.weight.data = model.mlp.down_proj.weight.data.float()
    quantize_linear_modules_(model, frozen_only=True)

    assert infer_storage_dtype(model) == torch.float16


def test_quantize_model_leaves_experts_alone_for_miners():
    """The miner cannot quantize the routed experts: each stacked tensor mixes
    its trainable group with the frozen helper group."""
    model = nn.Module()
    model.experts = _experts()
    model.proj = nn.Linear(8, 8).to(torch.float16)
    model.requires_grad_(False)

    quantize_model_(model, include_experts=False)

    assert isinstance(model.proj, Int8Linear)
    assert not model.experts._is_int8()

    quantize_model_(model, include_experts=True)
    assert model.experts._is_int8()


# ─────────────────────────────────────────────────────────────────────────────
# Validator eval-model seam
# ─────────────────────────────────────────────────────────────────────────────
def _validator_config(quantization: str):
    return types.SimpleNamespace(model=types.SimpleNamespace(quantization=quantization))


def test_foreground_eval_model_is_global_model_when_toggle_is_off():
    """With the toggle off the validator must build no third resident model.

    A persistent foreground copy costs every fp16 validator ~8-10 GB of VRAM,
    and off is the fleet-wide state for the whole shadow period. Returning
    `global_model` itself also makes the refactor a no-op by construction
    rather than by measurement.
    """
    from connito.validator.run import resolve_foreground_eval_model

    global_model = _toy_model()
    cache: dict = {}

    resolved = resolve_foreground_eval_model(
        config=_validator_config("off"),
        global_model=global_model,
        round_obj=types.SimpleNamespace(model_snapshot_cpu={}),
        cache=cache,
    )

    assert resolved is global_model
    assert cache == {}


def test_foreground_eval_model_is_a_quantized_copy_when_toggle_is_on(monkeypatch):
    from connito.validator.run import VALIDATOR_INT8_OVERRIDE_ENV, resolve_foreground_eval_model

    # Exercising the mechanism, so opt past the policy gate that normally
    # refuses int8 on a validator (see test_validator_int8_refuses_without_override).
    monkeypatch.setenv(VALIDATOR_INT8_OVERRIDE_ENV, "1")
    global_model = _toy_model()
    round_obj = types.SimpleNamespace(model_snapshot_cpu={})
    cache: dict = {}

    resolved = resolve_foreground_eval_model(
        config=_validator_config("int8"),
        global_model=global_model,
        round_obj=round_obj,
        cache=cache,
    )

    assert resolved is not global_model
    assert is_quantized(resolved)
    # global_model itself must stay fp16 — merge and the outer optimizer walk
    # named_parameters() and would silently skip int8 buffers.
    assert not is_quantized(global_model)
    # Persistent across rounds rather than rebuilt.
    assert resolve_foreground_eval_model(
        config=_validator_config("int8"),
        global_model=global_model,
        round_obj=round_obj,
        cache=cache,
    ) is resolved


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end on the real module tree
# ─────────────────────────────────────────────────────────────────────────────
def _tiny_deepseek():
    """A real `CustomDeekSeekMoE`, shrunk until it builds in a second on CPU.

    Worth the setup cost: the toy fixtures above cannot catch mistakes in module
    *selection*, which depends on the actual names transformers gives the MLA
    projections — and those differ between DeepSeek-V2 and V2-Lite.
    """
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Config

    from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeekSeekMoE

    config = DeepseekV2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=1, n_group=1, topk_group=1, q_lora_rank=None,
        kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        max_position_embeddings=64,
    )
    config.full = True
    config.num_experts = 8
    config.expert_group_assignment = None
    config.routing_mode = "masked_topk"
    return CustomDeekSeekMoE(config).to(torch.float16).eval()


def test_end_to_end_quantized_model_preserves_contract_and_loss():
    torch.manual_seed(0)
    model = _tiny_deepseek()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        reference = model(input_ids=ids, labels=ids)
    reference_sd = {k: v.clone() for k, v in model.state_dict().items()}

    model.requires_grad_(False)
    converted = quantize_model_(model, include_experts=True)

    # Selection: experts in, router and head out. `q_proj` (not q_a_proj) is
    # what V2-Lite actually has, since q_lora_rank is None.
    assert any(name.endswith(".experts") for name in converted)
    assert any(name.endswith("self_attn.q_proj") for name in converted)
    assert not any(name.endswith("mlp.gate") or name.endswith("lm_head") for name in converted)

    # The state_dict contract the hash and graft paths depend on.
    quantized_sd = model.state_dict()
    assert sorted(quantized_sd) == sorted(reference_sd)
    for key, ref in reference_sd.items():
        assert quantized_sd[key].shape == ref.shape
        assert quantized_sd[key].dtype == ref.dtype
    assert get_model_hash(quantized_sd, hex=True)

    with torch.no_grad():
        quantized = model(input_ids=ids, labels=ids)
    assert torch.isfinite(quantized.loss)
    # Loose on purpose. This asserts the plumbing is sane, NOT that production
    # eval loss is unaffected — this model is randomly initialised, so its loss
    # sits at ln(vocab) and is insensitive to weight perturbation. The number
    # that decides whether int8 is safe on a validator is rank preservation
    # over replayed rounds with real miner shards, on a GPU.
    assert abs(quantized.loss.item() - reference.loss.item()) < 0.1

    # The validator's per-miner graft: fp16 shard into the quantized base.
    result = model.load_state_dict(reference_sd, strict=False)
    assert result.missing_keys == [] and result.unexpected_keys == []


def test_compute_dtype_tracks_a_dtype_cast():
    """A stale `compute_dtype` would make `state_dict()` report the wrong dtype
    and silently break the graft path. Nothing in production casts a quantized
    model, but don't rely on call ordering to stay that way."""
    quantized = Int8Linear.from_linear(nn.Linear(8, 4).to(torch.float16))
    assert quantized.state_dict()["weight"].dtype == torch.float16

    quantized.to(dtype=torch.bfloat16)

    assert quantized.compute_dtype == torch.bfloat16
    assert quantized.state_dict()["weight"].dtype == torch.bfloat16
    assert quantized.weight_scale.dtype == torch.float32
    assert quantized.weight_int8.dtype == torch.int8


def test_experts_compute_dtype_tracks_a_dtype_cast():
    experts = _experts()
    experts.quantize_()
    assert experts.state_dict()["1.gate_up_proj"].dtype == torch.float16

    experts.to(dtype=torch.bfloat16)

    assert experts.state_dict()["1.gate_up_proj"].dtype == torch.bfloat16
    assert experts.gate_up_proj_scale.dtype == torch.float32
    assert experts.gate_up_proj_int8.dtype == torch.int8


def test_validator_int8_refuses_without_override(monkeypatch):
    """int8 eval is measured to corrupt scoring, so a validator must not be able
    to enable it by editing a YAML field. Hard failure, not a warning: both
    symptoms (weight divergence, tie-zeroing) are silent in production."""
    from connito.validator.run import VALIDATOR_INT8_OVERRIDE_ENV, quantize_eval_model_

    monkeypatch.delenv(VALIDATOR_INT8_OVERRIDE_ENV, raising=False)
    model = _toy_model()

    with pytest.raises(RuntimeError, match="not permitted on a validator"):
        quantize_eval_model_(_validator_config("int8"), model, role="foreground")
    assert not is_quantized(model)

    # ...and off stays a silent no-op.
    quantize_eval_model_(_validator_config("off"), model, role="foreground")
    assert not is_quantized(model)


def test_validator_int8_allowed_with_explicit_override(monkeypatch):
    from connito.validator.run import VALIDATOR_INT8_OVERRIDE_ENV, quantize_eval_model_

    monkeypatch.setenv(VALIDATOR_INT8_OVERRIDE_ENV, "1")
    model = _toy_model()

    quantize_eval_model_(_validator_config("int8"), model, role="foreground")

    assert is_quantized(model)

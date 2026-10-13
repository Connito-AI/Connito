"""Each family's frozen base is stored in fp8, converted while `from_pretrained` loads it.

Pins the silent failures: buffers recast by `.to(dtype)`, a router, `lm_head` or denied Linear
rounded, and any path calling `state_dict()`, which dequantizes every frozen expert.
CPU only, tiny models, nothing downloaded: `python -m pytest connito/test/test_fp8_fused_experts.py`.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch import nn
from transformers.models.deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.helper import get_model_hash
from connito.shared.modeling import mycelia, quantization
from connito.shared.modeling.families import FAMILIES
from connito.shared.modeling.quantization import (
    FP8_DTYPE,
    FP8FusedExperts,
    FP8Linear,
    fp8_load_config,
    quantize_,
)
from connito.test.conftest import TINY_DEEPSEEK_V2_CONFIG, TINY_TEXT_CONFIG

# Each family's MoE block, built from its tiny config.
BLOCK = {
    "qwen3_5_moe": (Qwen3_5MoeSparseMoeBlock, Qwen3_5MoeTextConfig, TINY_TEXT_CONFIG),
    "deepseek_v2": (DeepseekV2Moe, DeepseekV2Config, TINY_DEEPSEEK_V2_CONFIG),
}
# DeepSeek-V2's layer 0 is dense.
MOE_LAYERS = {"qwen3_5_moe": 4, "deepseek_v2": 2}


def _config(name: str, *, miner: str = "all", validator: str = "all", role: str = "validator",
            quantization: str = "fp8") -> SimpleNamespace:
    model = SimpleNamespace(
        family=name,
        quantization=quantization, quantization_miner=miner, quantization_validator=validator,
    )
    return SimpleNamespace(role=role, model=model)


def _point_at(monkeypatch, name: str, path: Path) -> None:
    monkeypatch.setitem(FAMILIES, name, replace(FAMILIES[name], model_path=str(path)))


def _load(name: str, scope: str = "all") -> nn.Module:
    return mycelia.get_base_model(_config(name, validator=scope))


def _load_bf16(name: str) -> nn.Module:
    return mycelia.get_base_model(_config(name, quantization="off"))


def _fp8_bytes(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: b.view(torch.uint8).clone() for n, b in model.named_buffers() if b.dtype == FP8_DTYPE}


# --- the experts block ------------------------------------------------------


@pytest.mark.parametrize("implementation", ["eager", "grouped_mm", "batched_mm"])
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_block_output_equals_upstream_on_the_rounded_weights(name, implementation):
    block_cls, config_cls, tiny = BLOCK[name]
    torch.manual_seed(0)
    config = config_cls(**tiny)
    config._experts_implementation = implementation
    block = block_cls(config)
    with torch.no_grad():
        for param in block.parameters():
            param.normal_(std=0.1)
    upstream = copy.deepcopy(block)
    quantize_(block, "experts", FAMILIES[name])
    assert isinstance(block.experts, FP8FusedExperts)
    with torch.no_grad():
        for tensor in ("gate_up_proj", "down_proj"):
            getattr(upstream.experts, tensor).copy_(block.experts.state_dict()[tensor])

    x = torch.randn(2, 5, tiny["hidden_size"])
    with torch.no_grad():
        assert torch.equal(block(x), upstream(x))


def test_gradient_reaches_the_input_through_frozen_experts(family_name):
    """Residual experts in earlier layers train through these ones."""
    model = _load(family_name, "experts")
    embeds = model.get_input_embeddings()(torch.zeros(1, 4, dtype=torch.long)).detach().requires_grad_(True)

    model(inputs_embeds=embeds).logits.float().sum().backward()

    assert embeds.grad is not None and embeds.grad.abs().sum() > 0


# --- the load path ----------------------------------------------------------


@pytest.mark.parametrize("scope", ["experts", "all"])
def test_fp8_load_keeps_state_dict_keys_and_shapes(family_name, scope):
    bf16 = _load_bf16(family_name).state_dict()
    fp8 = _load(family_name, scope)

    sd = fp8.state_dict()
    assert set(sd) == set(bf16)
    assert all(sd[k].shape == bf16[k].shape for k in sd)
    assert not any(".mlp.experts." in n for n, _ in fp8.named_parameters()), "no optimizer may see fp8 experts"


def test_experts_hold_the_rounding_of_the_fused_checkpoint_tensor(family_name):
    """DeepSeek-V2's checkpoint has per-expert tensors; they are fused before they are rounded."""
    bf16 = _load_bf16(family_name)
    model = _load(family_name, "experts")

    experts = [(n, m) for n, m in model.named_modules() if isinstance(m, FP8FusedExperts)]
    assert len(experts) == MOE_LAYERS[family_name]
    for name, module in experts:
        reference = FP8FusedExperts(bf16.get_submodule(name)).state_dict()
        for tensor in ("gate_up_proj", "down_proj"):
            assert torch.equal(module.state_dict()[tensor], reference[tensor])


@pytest.mark.parametrize("scope", ["experts", "all"])
def test_fp8_logits_stay_close_to_bf16(family_name, scope):
    reference = _load_bf16(family_name)
    model = _load(family_name, scope)
    input_ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits.float()
        got = model(input_ids=input_ids).logits.float()

    error = (got - expected).abs().mean() / expected.abs().mean()
    assert 0 < error < 0.05, f"relative error {error:.4f} outside the e4m3 budget"


def test_checkpoint_tensors_are_read_in_bf16_not_upcast_first(family_name, monkeypatch):
    seen: list[torch.dtype] = []
    to_fp8 = quantization._to_fp8
    monkeypatch.setattr(quantization, "_to_fp8", lambda w: seen.append(w.dtype) or to_fp8(w))

    _load(family_name, "all")

    assert set(seen) == {torch.bfloat16}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_scope_off_converts_nothing(name):
    block_cls, config_cls, tiny = BLOCK[name]
    block = block_cls(config_cls(**tiny))

    quantize_(block, "off", FAMILIES[name])

    assert not any(isinstance(m, FP8Linear | FP8FusedExperts) for m in block.modules())


def test_scope_experts_converts_only_the_fused_experts(family_name):
    model = _load(family_name, "experts")

    assert not any(isinstance(m, FP8Linear) for m in model.modules())
    assert sum(isinstance(m, FP8FusedExperts) for m in model.modules()) == MOE_LAYERS[family_name]


def test_scope_all_keeps_lm_head_router_and_family_deny_full_precision(family_name):
    bf16 = _load_bf16(family_name)
    model = _load(family_name, "all")

    routers = [n for n, _ in bf16.named_modules() if n.endswith("mlp.gate")]
    assert len(routers) == MOE_LAYERS[family_name]
    for name in routers:
        router, reference = model.get_submodule(name), bf16.get_submodule(name)
        assert type(router) is type(reference)
        assert isinstance(router.weight, nn.Parameter) and router.weight.dtype == torch.bfloat16
        assert torch.equal(router.weight, reference.weight)

    denied = [n for n, _ in bf16.named_modules() if n.rsplit(".", 1)[-1] in FAMILIES[family_name].fp8_deny]
    assert denied
    for name in [*denied, "lm_head"]:
        module = model.get_submodule(name)
        assert type(module) is nn.Linear and module.weight.dtype == torch.bfloat16
        assert torch.equal(module.weight, bf16.get_submodule(name).weight)
    assert any(isinstance(m, FP8Linear) for m in model.modules())


@pytest.mark.parametrize("off", [{"quantization": "off"}, {"validator": "off"}])
def test_off_loads_exactly_the_bf16_model(family_name, off):
    fam = FAMILIES[family_name]
    model = mycelia.get_base_model(_config(family_name, **off))
    reference = fam.causal_lm.from_pretrained(fam.model_path, revision=fam.revision, dtype=torch.bfloat16)

    assert not any(isinstance(m, FP8Linear | FP8FusedExperts) for m in model.modules())
    assert getattr(model.config, "quantization_config", None) is None
    sd, expected = model.state_dict(), reference.state_dict()
    assert set(sd) == set(expected)
    assert all(torch.equal(sd[k], expected[k]) for k in sd)


def test_scope_follows_the_role():
    config = _config("qwen3_5_moe", miner="experts", validator="all", role="miner")
    assert fp8_load_config(config).scope == "experts"
    config.role = "validator"
    assert fp8_load_config(config).scope == "all"
    assert fp8_load_config(_config("qwen3_5_moe", miner="off", role="miner")) is None
    assert fp8_load_config(_config("qwen3_5_moe", quantization="off")) is None


def test_cast_does_not_touch_the_fp8_buffers(family_name):
    """fp8 and the fp32 scales are floating dtypes, so `.to(dtype)` would silently recast both."""
    model = _load(family_name, "all")
    before = {n: b.clone() for n, b in model.named_buffers()}
    model.to(dtype=torch.bfloat16)

    fp8 = [m for m in model.modules() if isinstance(m, FP8Linear | FP8FusedExperts)]
    assert fp8 and all(
        getattr(m, name).dtype == FP8_DTYPE and getattr(m, f"{name}_scale").dtype == torch.float32
        for m in fp8 for name in m._fp8_names
    )
    stored = {n: b for n, b in model.named_buffers() if b.dtype in (FP8_DTYPE, torch.float32)}
    assert stored and all(torch.equal(b.view(torch.uint8), before[n].view(torch.uint8)) for n, b in stored.items())


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_each_row_has_its_own_scale(name):
    """One scale per tensor would push a row 1e5x smaller than the rest into e4m3's subnormals."""
    block_cls, config_cls, tiny = BLOCK[name]
    experts = block_cls(config_cls(**tiny)).experts
    linear = nn.Linear(tiny["hidden_size"], 8, bias=False)
    with torch.no_grad():
        for param in (experts.gate_up_proj, experts.down_proj, linear.weight):
            param.normal_()
            param[(0,) * (param.dim() - 1)] *= 1e-5
    weights = {n: p.detach().clone() for n, p in [*experts.named_parameters(), *linear.named_parameters()]}

    fused, fp8_linear = FP8FusedExperts(experts), FP8Linear(linear)

    stored = {**fused.state_dict(), **fp8_linear.state_dict()}
    for tensor, weight in weights.items():
        error = (stored[tensor] - weight).norm(dim=-1) / weight.norm(dim=-1)
        assert error.max() <= 2**-4, f"{tensor}: worst row error {error.max():.3f}"
    assert fused.gate_up_proj_scale.shape == (*weights["gate_up_proj"].shape[:2], 1)
    assert fused.down_proj_scale.shape == (*weights["down_proj"].shape[:2], 1)
    assert fp8_linear.weight_scale.shape == (8, 1)


def test_loading_full_precision_into_fp8_requantizes(family_name):
    """A raw copy would put bf16 values under the old scales."""
    model = _load(family_name, "all")
    key = next(k for k in model.state_dict() if k.endswith("experts.gate_up_proj"))
    incoming = torch.randn_like(model.state_dict()[key]).to(torch.bfloat16)

    model.load_state_dict({key: incoming}, strict=False)

    torch.testing.assert_close(model.state_dict()[key], incoming.float(), rtol=0.07, atol=1e-3)


# --- scope names this release dropped, in an operator's YAML ----------------


def _worker_yaml(tmp_path: Path, **model: str) -> Path:
    task = Path("expert_groups/exp_pmc_residual/config.yaml")  # the locked default task
    (tmp_path / task.parent).mkdir(parents=True)
    (tmp_path / task).write_text(task.read_text())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        # Pre-filled so from_path skips the chain lookup.
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
        "model": model,
    }))
    return path


@pytest.mark.parametrize("scope", ["attention+experts", "all"])
def test_an_old_validator_scope_is_reset_on_load(tmp_path, scope):
    """Locked, so a dropped scope or the old "all" default resets rather than stopping the boot."""
    path = _worker_yaml(tmp_path, quantization_validator=scope)

    config = ValidatorConfig.from_path(path, active_task=None, auto_update_config=True)

    assert config.model.quantization_validator == "experts"
    assert yaml.safe_load(path.read_text())["model"]["quantization_validator"] == "experts"


def test_a_dropped_validator_scope_kept_without_auto_update_fails_at_model_load(
    tmp_path, tiny_qwen_checkpoint, monkeypatch,
):
    path = _worker_yaml(tmp_path, quantization_validator="attention+experts")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    _point_at(monkeypatch, "qwen3_5_moe", tiny_qwen_checkpoint)

    config = ValidatorConfig.from_path(path, active_task=None, auto_update_config=False)

    assert config.model.quantization_validator == "attention+experts"
    with pytest.raises(ValueError, match=r"scope must be one of \('off', 'experts', 'all'\), got 'attention\+experts'"):
        mycelia.get_base_model(config)


def test_a_dropped_miner_scope_stops_a_miner_naming_the_field_and_the_scopes(tmp_path):
    path = _worker_yaml(tmp_path, quantization_miner="attention+experts")

    with pytest.raises(ValidationError, match=r"model.quantization_miner must be one of \('off', 'experts', 'all'\)"):
        MinerConfig.from_path(path, active_task=None, auto_update_config=True)


def test_a_dropped_miner_scope_does_not_stop_a_validator(tmp_path):
    """A validator never reads it."""
    path = _worker_yaml(tmp_path, quantization_miner="attention+experts")

    config = ValidatorConfig.from_path(path, active_task=None, auto_update_config=True)

    assert config.model.quantization_miner == "attention+experts"


# --- paths that must not dequantize the base --------------------------------


def _refuse(*_args, **_kwargs):
    raise AssertionError("state_dict() dequantizes every frozen expert")


def test_load_model_from_path_never_calls_state_dict(family_name, monkeypatch):
    """It runs once per miner evaluated."""
    from connito.validator import evaluator

    model = _load(family_name, "all")
    norm = torch.full_like(model.model.norm.weight, 0.5)
    monkeypatch.setattr(evaluator, "load_state_dict_from_path", lambda _p: {"model.norm.weight": norm})
    monkeypatch.setattr(model, "state_dict", _refuse)

    loaded = evaluator.load_model_from_path("ignored", model, torch.device("cpu"))

    assert torch.equal(loaded.model.norm.weight, norm)


def test_parameters_only_snapshot_restores_params_and_leaves_fp8_intact(family_name):
    """The validator's base shard takes parameters, not `state_dict()`; loading
    it back with `strict=False` must leave the fp8 buffers it lacks untouched."""
    model = _load(family_name, "all")
    snapshot = {k: v.detach().clone() for k, v in model.named_parameters()}
    fp8_before = _fp8_bytes(model)
    assert fp8_before and not set(snapshot) & set(fp8_before)
    with torch.no_grad():
        model.model.norm.weight.add_(1.0)

    incompatible = model.load_state_dict(snapshot, strict=False)

    assert not incompatible.unexpected_keys
    for name, param in model.named_parameters():
        assert torch.equal(param, snapshot[name])
    assert all(torch.equal(b, fp8_before[n]) for n, b in _fp8_bytes(model).items())


def test_trainable_state_is_detached_and_hashable_without_state_dict(family_name, monkeypatch):
    """`get_model_hash` serializes with `.numpy()`, which a live fp32 parameter
    refuses; the miner upcasts its trainable parameters to fp32."""
    from connito.miner.train import _trainable_state

    model = _load(family_name, "all")
    trainable = {"model.norm.weight", "model.layers.0.input_layernorm.weight"}
    for name, param in model.named_parameters():
        param.requires_grad_(name in trainable)
        if name in trainable:
            param.data = param.data.float()
    monkeypatch.setattr(model, "state_dict", _refuse)

    state = _trainable_state(model)

    assert set(state) == trainable
    assert all(t.requires_grad is False for t in state.values())
    assert get_model_hash(state, hex=True)
    with pytest.raises(RuntimeError, match="requires grad"):
        get_model_hash(dict(model.named_parameters()), hex=True)

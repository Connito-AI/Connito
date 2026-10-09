"""Residual experts on each family's MoE blocks, installed by `get_model_from_checkpoint`.

CPU only, tiny models, nothing downloaded: `python -m pytest connito/test/test_residual_experts.py`.
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from connito.shared import model as shared_model
from connito.shared.config import ResidualCfg
from connito.shared.modeling import mycelia
from connito.shared.modeling.families import FAMILIES
from connito.shared.modeling.residual import install_residual_experts

FIXTURE = {"qwen3_5_moe": "tiny_qwen_checkpoint", "deepseek_v2": "tiny_deepseek_v2_checkpoint"}
# DeepSeek-V2's layer 0 is dense.
MOE_LAYERS = {"qwen3_5_moe": [0, 1, 2, 3], "deepseek_v2": [1, 2]}
HIDDEN, WIDTH = 64, 32
TOKENS = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))


@pytest.fixture(params=sorted(FAMILIES))
def family_name(request, monkeypatch) -> str:
    """A family whose record points at its tiny checkpoint."""
    path = request.getfixturevalue(FIXTURE[request.param])
    monkeypatch.setitem(FAMILIES, request.param, replace(FAMILIES[request.param], model_path=str(path)))
    return request.param


def _config(name: str, scope: str = "off", k: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        role="miner",
        model=SimpleNamespace(
            family=name, torch_compile=False, device="cpu",
            quantization="off" if scope == "off" else "fp8", quantization_miner=scope,
        ),
        task=SimpleNamespace(exp=SimpleNamespace(residual=ResidualCfg(k=k, threshold=0.5, temperature=0.5))),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )


def _load(config) -> torch.nn.Module:
    model, _ = shared_model.get_model_from_checkpoint(rank=0, config=config, load_global_checkpoint=False)
    return model


def _logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model.eval()(input_ids=TOKENS).logits


def _residual_specs(name: str, k: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    specs = {}
    for layer in MOE_LAYERS[name]:
        prefix = f"model.layers.{layer}.mlp.residual_"
        for j in range(k):
            specs[f"{prefix}experts.{j}.gate_proj.weight"] = ((WIDTH, HIDDEN), torch.bfloat16)
            specs[f"{prefix}experts.{j}.up_proj.weight"] = ((WIDTH, HIDDEN), torch.bfloat16)
            specs[f"{prefix}experts.{j}.down_proj.weight"] = ((HIDDEN, WIDTH), torch.bfloat16)
        specs[f"{prefix}router.weight"] = ((k, HIDDEN), torch.float32)
    return specs


def _residual(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if ".residual_" in n}


@pytest.mark.parametrize("scope", ["off", "all"])
def test_logits_equal_the_base_at_init(family_name, scope):
    config = _config(family_name, scope)
    base = mycelia.get_base_model(config)

    model = _load(config)

    assert model.is_gradient_checkpointing
    assert torch.equal(_logits(model), _logits(base))
    # Training mode, with gradients and checkpointing on.
    assert torch.equal(model.train()(input_ids=TOKENS).logits, base.train()(input_ids=TOKENS).logits)


def test_residual_keys_shapes_and_dtypes(family_name):
    config = _config(family_name, k=2)
    base_keys = {n for n, _ in mycelia.get_base_model(config).named_parameters()}

    model = _load(config)

    residual = _residual(model)
    assert {n: (tuple(p.shape), p.dtype) for n, p in residual.items()} == _residual_specs(family_name, 2)
    assert {n for n, _ in model.named_parameters()} - set(residual) == base_keys
    experts = [e for layer in MOE_LAYERS[family_name] for e in model.model.layers[layer].mlp.residual_experts]
    assert {type(e) for e in experts} == {FAMILIES[family_name].mlp}
    assert {p.dtype for n, p in model.named_parameters() if n not in residual} == {torch.bfloat16}


def test_a_resumed_shard_keeps_the_router_fp32_and_the_rest_bf16(family_name, tmp_path, monkeypatch):
    """The shard is saved fp16, so only the model's own dtypes can make its tensors bf16 and fp32."""
    config = _config(family_name)
    saved = {n: torch.full_like(p, 0.25, dtype=torch.float16) for n, p in _residual(_load(config)).items()}
    save_file(saved, str(tmp_path / "model_expgroup_9.safetensors"))
    config.ckpt = SimpleNamespace(
        resume_from_ckpt=True, use_pretrained_only=False, validator_checkpoint_path=tmp_path, checkpoint_path=tmp_path,
    )
    monkeypatch.setattr(shared_model, "select_best_checkpoint",
                        lambda **kw: SimpleNamespace(path=str(tmp_path / "model_expgroup_9.safetensors")))

    model, _ = shared_model.get_model_from_checkpoint(rank=0, config=config)

    residual = _residual(model)
    assert {n: (tuple(p.shape), p.dtype) for n, p in residual.items()} == _residual_specs(family_name, 1)
    assert all(bool((p == 0.25).all()) for p in residual.values())
    assert {p.dtype for n, p in model.named_parameters() if n not in residual} == {torch.bfloat16}


def test_reinstalling_replaces_the_residual(family_name):
    config = _config(family_name, k=1)
    model = _load(config)
    expected = _logits(model)

    install_residual_experts(model, _config(family_name, k=3))

    assert {n: (tuple(p.shape), p.dtype) for n, p in _residual(model).items()} == _residual_specs(family_name, 3)
    for layer in MOE_LAYERS[family_name]:
        assert len(model.model.layers[layer].mlp._forward_hooks) == 1
    assert torch.equal(_logits(model), expected)


@pytest.mark.parametrize("scope", ["off", "experts"])
@torch.no_grad()
def test_init_scales_follow_the_base_experts_and_router(family_name, scope):
    """The base's gate half, up half and router are rescaled apart, so only scales read from them can match."""
    config = _config(family_name, scope, k=16)
    model = _load(config)
    for layer in MOE_LAYERS[family_name]:
        block = model.model.layers[layer].mlp
        # fp8 experts are rescaled through their per-row scales.
        values = block.experts.gate_up_proj_scale if scope == "experts" else block.experts.gate_up_proj
        values[:, :WIDTH] *= 3
        values[:, WIDTH:] *= 0.5
        block.gate.weight *= 5
    torch.manual_seed(0)

    install_residual_experts(model, config)

    for layer in MOE_LAYERS[family_name]:
        block = model.model.layers[layer].mlp
        weights = block.experts.gate_up_proj.float()
        if scope == "experts":
            weights = weights * block.experts.gate_up_proj_scale
        gate_std, up_std = (half.flatten(1).std(dim=1).mean() for half in weights.split(WIDTH, dim=1))
        experts = block.residual_experts
        assert torch.stack([e.gate_proj.weight.float() for e in experts]).std() == pytest.approx(gate_std, rel=0.03)
        assert torch.stack([e.up_proj.weight.float() for e in experts]).std() == pytest.approx(up_std, rel=0.03)
        assert all(bool((e.down_proj.weight == 0).all()) for e in experts)
        router_std = block.gate.weight.float().std()
        assert block.residual_router.weight.std() == pytest.approx(router_std, rel=0.1)


def _trainable(name: str) -> torch.nn.Module:
    """The model in training mode, with every residual expert's output non-zero so gradients reach the routers."""
    torch.manual_seed(0)
    model = _load(_config(name))
    with torch.no_grad():
        for n, p in _residual(model).items():
            if "down_proj" in n:
                p.normal_(std=0.02)
    return model.train()


class _RecomputeTracker:
    """Weak references to every tensor each checkpointed layer's recompute makes, by layer."""

    def __init__(self) -> None:
        self.made: dict[int, list[weakref.ref]] = {}
        self.calls = 0
        self.most_alive_from_other_layers = 0

    def alive(self, skip: int | None = None) -> int:
        return sum(ref() is not None for layer, refs in self.made.items() if layer != skip for ref in refs)

    def context_fn(self):
        layer, tracker = self.calls, self
        self.calls += 1

        class Recompute(TorchDispatchMode):
            def __enter__(self):
                tracker.most_alive_from_other_layers = max(tracker.most_alive_from_other_layers, tracker.alive(layer))
                return super().__enter__()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                refs = tracker.made.setdefault(layer, [])
                refs.extend(weakref.ref(t) for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor))
                return out

        return contextlib.nullcontext(), Recompute()


def test_checkpointing_keeps_no_layer_alive_into_another_layers_backward(family_name):
    """Storing the gates q on the module would fail this."""
    # It keeps every recomputed layer's graph alive through the rest of backward: 0.9 GiB per layer on real Qwen.
    model = _trainable(family_name)
    tracker = _RecomputeTracker()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False, "context_fn": tracker.context_fn},
    )

    loss = model(input_ids=TOKENS, labels=TOKENS).loss
    on_modules = [
        name for name, module in model.named_modules() for value in tree_flatten(list(vars(module).values()))[0]
        if isinstance(value, torch.Tensor) and value.grad_fn is not None
    ]
    loss.backward()

    assert on_modules == []
    assert tracker.calls == len(model.model.layers)
    assert tracker.made
    assert tracker.most_alive_from_other_layers == 0
    assert tracker.alive() == 0

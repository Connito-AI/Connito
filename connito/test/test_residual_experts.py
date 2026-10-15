"""Residual experts on each family's MoE blocks, installed by `get_model_from_checkpoint`.

CPU only, tiny models, nothing downloaded: `python -m pytest connito/test/test_residual_experts.py`.
"""

from __future__ import annotations

import contextlib
import math
import weakref
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from connito.shared import model as shared_model
from connito.shared.config import ResidualCfg
from connito.shared.evaluate import EvalDeadlineExceeded, evaluate_model
from connito.shared.modeling import mycelia
from connito.shared.modeling.families import FAMILIES
from connito.shared.modeling.residual import (
    GateUsage,
    freeze_all_but_residual,
    install_residual_experts,
    record_gate_usage,
)

# DeepSeek-V2's layer 0 is dense.
MOE_LAYERS = {"qwen3_5_moe": [0, 1, 2, 3], "deepseek_v2": [1, 2]}
HIDDEN, WIDTH = 64, 32
TOKENS = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))


def _config(name: str, scope: str = "off", k: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        role="miner",
        model=SimpleNamespace(
            family=name, device="cpu",
            quantization="off" if scope == "off" else "fp8", quantization_miner=scope,
        ),
        task=SimpleNamespace(exp=SimpleNamespace(
            group_id=9, residual=ResidualCfg(k=k, threshold=0.5, temperature=0.5),
        )),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )


def _load(config) -> torch.nn.Module:
    model, _ = shared_model.get_model_from_checkpoint(config=config, load_global_checkpoint=False)
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


def test_a_resumed_shard_keeps_the_router_fp32_and_the_rest_bf16(family_name, tmp_path):
    config = _config(family_name)
    saved = {n: torch.full_like(p, 0.25) for n, p in _residual(_load(config)).items()}
    (tmp_path / "globalver_1_inneropt_1").mkdir()
    save_file(saved, str(tmp_path / "globalver_1_inneropt_1" / "model_expgroup_9.safetensors"))
    config.ckpt = SimpleNamespace(
        resume_from_ckpt=True, use_pretrained_only=False, validator_checkpoint_path=tmp_path, checkpoint_path=tmp_path,
    )

    model, _ = shared_model.get_model_from_checkpoint(config=config)

    residual = _residual(model)
    assert {n: (tuple(p.shape), p.dtype) for n, p in residual.items()} == _residual_specs(family_name, 1)
    assert all(bool((p == 0.25).all()) for p in residual.values())
    assert {p.dtype for n, p in model.named_parameters() if n not in residual} == {torch.bfloat16}


def test_only_residual_parameters_train_routers_included(family_name):
    model = _load(_config(family_name))

    freeze_all_but_residual(model)

    assert {n for n, p in model.named_parameters() if p.requires_grad} == set(_residual_specs(family_name, 1))


def _step(model, optimizer) -> None:
    model(input_ids=TOKENS, labels=TOKENS).loss.backward()
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize("scope", ["off", "all"])
def test_one_step_leaves_the_base_unchanged_and_moves_the_output(family_name, scope):
    model = _load(_config(family_name, scope))
    freeze_all_but_residual(model)
    before = _logits(model)
    base = {n: t.clone() for n, t in [*model.named_parameters(), *model.named_buffers()] if ".residual_" not in n}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    model.train()
    model(input_ids=TOKENS, labels=TOKENS).loss.backward()
    # down_proj is 0 at init, so only it gets a gradient.
    grads = {n: bool(p.grad.any()) for n, p in _residual(model).items()}
    assert grads == {n: "down_proj" in n for n in grads}
    optimizer.step()

    after = {n: t for n, t in [*model.named_parameters(), *model.named_buffers()] if ".residual_" not in n}
    assert after.keys() == base.keys()
    assert all(torch.equal(after[n].view(torch.uint8), base[n].view(torch.uint8)) for n in base)
    assert not torch.equal(_logits(model), before)


def test_the_router_moves_once_down_proj_has(family_name):
    """The task loss alone trains the router: nothing at step 1, while down_proj is 0, then from step 2."""
    model = _load(_config(family_name)).train()
    freeze_all_but_residual(model)
    routers = {n: p for n, p in _residual(model).items() if "router" in n}
    initial = {n: p.detach().clone() for n, p in routers.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)

    _step(model, optimizer)
    assert all(torch.equal(p, initial[n]) for n, p in routers.items())
    _step(model, optimizer)

    assert all(not torch.equal(p, initial[n]) for n, p in routers.items())


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


def _batch(real_lengths: tuple[int, int]) -> dict[str, torch.Tensor]:
    """TOKENS, each row right-padded after its real length."""
    mask = (torch.arange(TOKENS.shape[1]) < torch.tensor(real_lengths)[:, None]).long()
    return {"input_ids": TOKENS, "attention_mask": mask, "labels": TOKENS.masked_fill(mask == 0, -100)}


def _padded_batch() -> dict[str, torch.Tensor]:
    return _batch((12, 5))


def test_eval_reports_the_mean_gate_over_real_tokens_and_moe_layers(family_name):
    """Σ over K = 2 experts, mean over layers and real tokens, across batches padded differently."""
    model = _load(_config(family_name, k=2))
    batches = [_batch((12, 5)), _batch((3, 12))]
    inputs = []
    blocks = [model.model.layers[i].mlp for i in MOE_LAYERS[family_name]]
    handles = [b.register_forward_pre_hook(lambda m, args: inputs.append((m, args[0]))) for b in blocks]

    usage = evaluate_model(0, model, batches, torch.device("cpu"), None)["gate_usage"]

    for h in handles:
        h.remove()
    assert len(inputs) == len(blocks) * len(batches)
    # ResidualCfg(threshold=0.5, temperature=0.5) in `_config`.
    gates = [
        torch.stack([
            torch.sigmoid((torch.sigmoid(x.float() @ m.residual_router.weight.T) - 0.5) / 0.5).sum(-1)
            for m, x in inputs[i * len(blocks):(i + 1) * len(blocks)]
        ])  # [layers, rows, positions]
        for i in range(len(batches))
    ]
    real = torch.cat([g[:, b["attention_mask"].bool()] for g, b in zip(gates, batches, strict=True)], dim=1)
    assert usage == pytest.approx(float(real.mean()), rel=1e-5)
    assert usage != pytest.approx(float(torch.cat(gates, dim=1).mean()), rel=1e-5)  # padding is left out


def test_a_batch_whose_loss_is_dropped_adds_nothing_to_the_gate_usage(family_name):
    model = _load(_config(family_name))
    unlabelled = {**_batch((3, 12)), "labels": torch.full_like(TOKENS, -100)}  # loss 0/0, dropped from val_loss

    alone = evaluate_model(0, model, [_padded_batch()], torch.device("cpu"), None)
    with_dropped = evaluate_model(0, model, [_padded_batch(), unlabelled], torch.device("cpu"), None)

    assert with_dropped["nan_batches"] == 1
    assert with_dropped["gate_usage"] == alone["gate_usage"]


def test_a_nan_gate_on_a_padded_position_leaves_the_usage_finite():
    usage = GateUsage()
    usage.mask = torch.tensor([[1, 1, 0]])
    usage.add(torch.tensor([[0.25, 0.5], [0.5, 0.25], [math.nan, math.nan]]))  # [tokens, K]

    assert usage.mean() == 0.75


@pytest.mark.parametrize("deadline", [None, 0.0])
def test_no_block_keeps_the_usage_recorder_after_an_eval(family_name, deadline):
    model = _load(_config(family_name))

    with contextlib.suppress(EvalDeadlineExceeded):
        evaluate_model(0, model, [_padded_batch()], torch.device("cpu"), None, deadline_monotonic=deadline)

    assert not any(hasattr(module, "gate_usage") for module in model.modules())


def test_a_model_without_residual_experts_reports_no_gate_usage():
    model = torch.nn.Linear(2, 2)
    with record_gate_usage(model) as usage:
        pass
    assert usage.mean() == 0.0

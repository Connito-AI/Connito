"""Residual experts: K soft-gated SwiGLU experts beside every MoE block, h = h_old(x) + Σ_j q_j(x)·E_j(x).

q_j = sigmoid((sigmoid(w_j·x) − A) / τ). Children of the block, run by a forward hook, so base keys are unchanged.
"""
from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from connito.shared.checkpoint_helper import TensorSpec
from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.modeling.families import ModelFamily, family


def _residual_hook(block: nn.Module, args, out: torch.Tensor, *, threshold: float, temperature: float):
    flat = args[0].reshape(-1, args[0].shape[-1])
    # fp32 even under bf16 autocast: near A, bf16 resolves r only in steps of ~0.004, a staircase gate at τ = 0.05.
    with torch.autocast(flat.device.type, enabled=False):
        r = torch.sigmoid(F.linear(flat.float(), block.residual_router.weight))
        q = torch.sigmoid((r - threshold) / temperature)
    usage = getattr(block, "gate_usage", None)
    if usage is not None:
        usage.add(q)
    h = sum(q[:, j, None].to(flat.dtype) * expert(flat) for j, expert in enumerate(block.residual_experts))
    return out + h.view_as(out)


class GateUsage:
    """Σ_j q_j summed over real tokens and MoE layers, while `record_gate_usage` attaches it."""

    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None  # the current batch's attention_mask; 1 marks a real token
        self.total = self.tokens = 0

    def add(self, q: torch.Tensor) -> None:
        real = self.mask.reshape(-1).to(q.device).bool()
        # Selected, not multiplied by the mask: a NaN gate on a padded position would poison the sum (NaN·0 = NaN).
        gates = torch.where(real, q.sum(-1), 0.0).sum(dtype=torch.float64)
        # Kept on the mask's device as 0-dim tensors: no sync per layer, and a model split across GPUs still adds up.
        # Rebound, not added in place, so `evaluate_model` can restore a snapshot when it drops a batch.
        self.total = self.total + gates.to(self.mask.device)
        self.tokens = self.tokens + real.sum().to(self.mask.device)

    def mean(self) -> float:
        """Mean over MoE layers and real tokens; 0.0 when no residual gate ran."""
        return float(self.total / self.tokens) if self.tokens else 0.0


@contextlib.contextmanager
def record_gate_usage(model: nn.Module) -> Iterator[GateUsage]:
    """Attach one `GateUsage` to every residual block for the duration; each block's hook adds its gates."""
    usage = GateUsage()
    blocks = [module for module in model.modules() if hasattr(module, "residual_hook")]
    for block in blocks:
        block.gate_usage = usage
    try:
        yield usage
    finally:
        for block in blocks:
            del block.gate_usage


def _attach(block: nn.Module, fam: ModelFamily, text, k: int) -> None:
    """Hang K residual experts, in the block's dtype, and their fp32 router on the block, weights as built."""
    ref = block.gate.weight
    experts = nn.ModuleList(fam.mlp(text, intermediate_size=text.moe_intermediate_size) for _ in range(k))
    block.residual_experts = experts.to(ref.device, ref.dtype)
    block.residual_router = nn.Linear(text.hidden_size, k, bias=False, dtype=torch.float32, device=ref.device)


def install_residual_experts(model: nn.Module, config: MinerConfig | ValidatorConfig) -> None:
    """Attach the task's K residual experts to every MoE block, replacing any already there; down_proj starts at 0."""
    cfg, text, fam = config.task.exp.residual, model.config, family(config)
    for decoder in model.model.layers:
        block = decoder.mlp
        if not isinstance(block, fam.moe_block):
            continue
        if hasattr(block, "residual_hook"):
            block.residual_hook.remove()
        ref, gate_up = block.gate.weight, block.experts.gate_up_proj
        scale = getattr(block.experts, "gate_up_proj_scale", None)  # present when the experts are stored in fp8
        stds = torch.zeros(2)
        for e in range(gate_up.shape[0]):  # one expert in fp32 at a time
            w = gate_up[e].float() if scale is None else gate_up[e].float() * scale[e]
            stds += torch.stack([w[: text.moe_intermediate_size].std(), w[text.moe_intermediate_size :].std()]).cpu()
        gate_std, up_std = (stds / gate_up.shape[0]).tolist()
        _attach(block, fam, text, cfg.k)
        with torch.no_grad():
            for expert in block.residual_experts:
                nn.init.normal_(expert.gate_proj.weight, std=gate_std)
                nn.init.normal_(expert.up_proj.weight, std=up_std)
                nn.init.zeros_(expert.down_proj.weight)
            nn.init.normal_(block.residual_router.weight, std=ref.float().std().item())
        block.residual_hook = block.register_forward_hook(functools.partial(
            _residual_hook, threshold=cfg.threshold, temperature=cfg.temperature,
        ))


def residual_shapes(config: MinerConfig | ValidatorConfig) -> dict[str, TensorSpec]:
    """Every residual key of the family and task K, with its shape and dtype, read from config.json alone."""
    fam = family(config)
    text = fam.causal_lm.config_class.from_pretrained(fam.model_path, revision=fam.revision)
    # A meta-device skeleton holds no weights, and walking it skips dense layers as the install does.
    with torch.device("meta"):
        skeleton = fam.causal_lm(text).to(torch.bfloat16)
        for decoder in skeleton.model.layers:
            if isinstance(decoder.mlp, fam.moe_block):
                _attach(decoder.mlp, fam, text, config.task.exp.residual.k)
    return {n: TensorSpec(tuple(p.shape), p.dtype) for n, p in skeleton.named_parameters() if ".residual_" in n}


def freeze_all_but_residual(model: nn.Module) -> None:
    """Train the residual experts and their routers; freeze everything else. Refuse a model that has none."""
    for name, param in model.named_parameters():
        param.requires_grad_(".residual_" in name)
    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError("the model has no residual experts (`.residual_` parameters) to train")

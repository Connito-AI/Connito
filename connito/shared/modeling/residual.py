"""Residual experts: K soft-gated SwiGLU experts beside every MoE block, h = h_old(x) + Σ_j q_j(x)·E_j(x).

q_j = sigmoid((sigmoid(w_j·x) − A) / τ). Children of the block, run by a forward hook, so base keys are unchanged.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.modeling.families import family


def _residual_hook(block: nn.Module, args, out: torch.Tensor, *, threshold: float, temperature: float):
    flat = args[0].reshape(-1, args[0].shape[-1])
    # fp32 even under bf16 autocast: near A, bf16 resolves r only in steps of ~0.004, a staircase gate at τ = 0.05.
    with torch.autocast(flat.device.type, enabled=False):
        r = torch.sigmoid(F.linear(flat.float(), block.residual_router.weight))
        q = torch.sigmoid((r - threshold) / temperature)
    h = sum(q[:, j, None].to(flat.dtype) * expert(flat) for j, expert in enumerate(block.residual_experts))
    return out + h.view_as(out)


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
        experts = nn.ModuleList(fam.mlp(text, intermediate_size=text.moe_intermediate_size) for _ in range(cfg.k))
        router = nn.Linear(text.hidden_size, cfg.k, bias=False, dtype=torch.float32)
        with torch.no_grad():
            for expert in experts:
                nn.init.normal_(expert.gate_proj.weight, std=gate_std)
                nn.init.normal_(expert.up_proj.weight, std=up_std)
                nn.init.zeros_(expert.down_proj.weight)
            nn.init.normal_(router.weight, std=ref.float().std().item())
        block.residual_experts = experts.to(ref.device, ref.dtype)
        block.residual_router = router.to(ref.device)
        block.residual_hook = block.register_forward_hook(functools.partial(
            _residual_hook, threshold=cfg.threshold, temperature=cfg.temperature,
        ))


def freeze_all_but_residual(model: nn.Module) -> None:
    """Train the residual experts and their routers; freeze everything else. Refuse a model that has none."""
    for name, param in model.named_parameters():
        param.requires_grad_(".residual_" in name)
    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError("the model has no residual experts (`.residual_` parameters) to train")

"""Exponential moving average (EMA) tracker for model parameters.

Rationale: EMA-averaged weights almost always have lower validation loss
than the most recent checkpoint at no extra training compute. Because
validator scoring is `(baseline_loss - miner_val_loss) ** 1.2`
(see connito/validator/evaluator.py), lower val_loss = higher score.

This module is opt-in via `config.training.ema.enabled` and defaults
to off so existing miner behavior is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from connito.shared.app_logging import configure_logging, structlog

configure_logging()
logger = structlog.get_logger(__name__)


# Filename used by both `connito.miner.train` (writer) and
# `connito.miner.model_io` (reader) for the EMA shadow snapshot saved
# alongside each checkpoint. Shared here so the two sites cannot drift.
EMA_SHADOW_FILENAME = "ema_shadow.safetensors"


class ModelEma:
    """Maintain `θ_ema = decay * θ_ema + (1 - decay) * θ_live` in-place.

    Shadow weights live on the same device as the live params by default,
    but can be relocated (e.g. to CPU) via the `device` argument to save
    VRAM. Only trainable parameters are tracked by default — backbone /
    frozen params are reconstructable elsewhere and tracking them wastes
    memory.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: str | torch.device | None = None,
        only_trainable: bool = True,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        self.decay = float(decay)
        self.device = device
        self.only_trainable = bool(only_trainable)
        self.num_updates = 0

        # Initialize shadow state with detached clones of (trainable) params.
        # Keep the same dtype as the live params to avoid silent precision loss.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if only_trainable and not p.requires_grad:
                continue
            shadow_p = p.detach().clone()
            if device is not None:
                shadow_p = shadow_p.to(device)
            self.shadow[name] = shadow_p

        logger.info(
            "ModelEma initialized",
            decay=self.decay,
            num_tracked_params=len(self.shadow),
            only_trainable=self.only_trainable,
            shadow_device=str(device) if device is not None else "live",
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Apply one EMA step against the live params.

        Skips any tracked param that has gone missing (defensive: model
        graph mutations between init and update are unusual but possible).
        """
        for name, p in model.named_parameters():
            shadow_p = self.shadow.get(name)
            if shadow_p is None:
                continue
            # Cast live param into the shadow's device+dtype before the
            # in-place fused add. `.add_(other, alpha=...)` requires
            # matching dtype on the receiver; matching the shadow's dtype
            # keeps shadow accumulation precision stable and avoids
            # surprises when the shadow lives on CPU/fp32 while the live
            # model runs in fp16/bf16 autocast.
            live = p.detach()
            if live.device != shadow_p.device or live.dtype != shadow_p.dtype:
                live = live.to(device=shadow_p.device, dtype=shadow_p.dtype)
            # shadow = decay * shadow + (1 - decay) * live
            shadow_p.mul_(self.decay).add_(live, alpha=1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a clone of the shadow state for serialization."""
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore shadow tensors from a state dict (e.g. after resume).

        Only keys that already exist in `self.shadow` are loaded — extras
        are ignored and missing keys keep their current value. This lets
        resume work even if the trainable-param set changed slightly
        between runs.
        """
        loaded = 0
        for name, tensor in state_dict.items():
            shadow_p = self.shadow.get(name)
            if shadow_p is None:
                continue
            shadow_p.data.copy_(tensor.to(device=shadow_p.device, dtype=shadow_p.dtype))
            loaded += 1
        logger.info("ModelEma loaded state_dict", loaded=loaded, total_tracked=len(self.shadow))

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap model params with EMA values in-place.

        Returns a dict of original (live) tensors so `restore` can put
        them back exactly. Use this around a commit/eval block:

            original = ema.apply_to(model)
            try:
                ...  # commit or eval against EMA weights
            finally:
                ema.restore(model, original)
        """
        original: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            shadow_p = self.shadow.get(name)
            if shadow_p is None:
                continue
            original[name] = p.detach().clone()
            p.data.copy_(shadow_p.to(device=p.device, dtype=p.dtype))
        return original

    @torch.no_grad()
    def restore(self, model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        """Restore previously-saved live params (counterpart to `apply_to`)."""
        for name, p in model.named_parameters():
            saved = original.get(name)
            if saved is None:
                continue
            p.data.copy_(saved.to(device=p.device, dtype=p.dtype))

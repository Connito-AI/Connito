"""Bootstrap + model-loading helpers for the owner eval daemon.

The read-only config load, device resolution, precision handling and
model-release helpers are adapted from
``notebook/diagnose_eval_parity.py`` (the offline eval-parity diagnostic) so the
two share the same proven setup path. Loading the *latest validator HF
checkpoint* into the full model is owner-eval-specific and goes through
``connito.shared.model.load_model``.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
import yaml

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


def load_config(config_path: str) -> Any:
    """Load an ``OwnerEvalConfig`` from a YAML file (read-only)."""
    from connito.shared.config import OwnerEvalConfig

    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return OwnerEvalConfig(**data)


def resolve_device(config: Any) -> torch.device:
    configured = getattr(config.model, "device", None)
    if configured:
        if str(configured).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("Configured CUDA device unavailable; falling back to CPU",
                           configured_device=configured)
            return torch.device("cpu")
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def release_model(model: torch.nn.Module | None) -> None:
    """Drop a model and reclaim GPU memory between runs."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_dtype(config: Any) -> torch.dtype:
    precision = getattr(config.model, "precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        precision = "fp16-mixed"
    return torch.bfloat16 if precision == "bf16-mixed" else torch.float16


def load_base_model(config: Any, expert_manager: Any, device: torch.device) -> torch.nn.Module:
    """Load the pretrained base full model directly (no chain/wallet).

    Used for canary / plumbing tests (``eval_pipeline.model_source == "base"``).
    """
    from connito.shared.modeling.mycelia import get_base_model

    model = get_base_model(config, expert_manager=expert_manager, group_ids=None, partial=False)
    model = model.to(device=device, dtype=_model_dtype(config)).eval()
    return model


def load_latest_full_model(
    config: Any,
    expert_manager: Any,
    subtensor: Any,
    wallet: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    """Build the full model under test.

    With ``eval_pipeline.model_source == "base"`` loads the pretrained base model
    directly (canary mode). Otherwise wraps ``connito.shared.model.load_model``
    with ``partial=False`` and ``current_checkpoint=None`` (always take the newest
    validator HF checkpoint). Returns ``(model, ModelCheckpoint | None)``; the
    checkpoint's ``global_ver`` stamps the model revision into telemetry.
    """
    if getattr(config.eval_pipeline, "model_source", "chain") == "base":
        return load_base_model(config, expert_manager, device), None

    from connito.shared.model import load_model

    model, checkpoint = load_model(
        rank=0,
        config=config,
        expert_manager=expert_manager,
        subtensor=subtensor,
        wallet=wallet,
        current_checkpoint=None,
        partial=False,
        checkpoint_device=device,
    )
    model.eval()
    return model, checkpoint


def model_revision_label(checkpoint: Any) -> str:
    """Human/Prometheus-friendly revision string for a loaded checkpoint."""
    global_ver = getattr(checkpoint, "global_ver", None)
    if global_ver is None:
        return "unknown"
    return f"globalver_{global_ver}"

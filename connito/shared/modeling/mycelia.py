from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
import gc
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Suppress noisy upstream warnings from transformers/rope config (unless DEBUG)
if logging.root.level > logging.DEBUG:
    warnings.filterwarnings("ignore", message=r".*rope_parameters.*")
    warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")
    logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)

from connito.shared.app_logging import structlog
from connito.shared.helper import get_nested_attr
from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.expert_manager import ExpertManager
from connito.shared.helper import *
# ── Model backend selection ──────────────────────────────────────────────────
# Change MODEL_BACKEND to swap implementations:
#   "deepseek_v2"  → connito.shared.modeling.custom_deepseek_v2_lite
MODEL_BACKEND = "deepseek_v2"

if MODEL_BACKEND == "deepseek_v2":
    from connito.shared.modeling.custom_deepseek_v2_lite import (
        CustomDeekSeekMoE as _CausalLMClass,
        get_moe_model_config as _get_moe_model_config_impl,
    )

    def get_moe_model_config(config, topk, group_ids_trainable, group_ids_helper, expert_manager, full = False):
        # Pipe through the 2Fnat knobs. Default routing_mode="natural_with_fallback"
        # for the miner/validator path — the routing branch self-gates on
        # trainable_ids and helper_ids both being non-empty, so single-group
        # loads (only the trainable group; group_ids_helper=None) silently fall
        # through to masked-topk with byte-identical numerics. When helper
        # groups are also loaded, 2Fnat activates automatically.
        routing_mode = str(get_nested_attr(config, "task.routing_mode", "natural_with_fallback"))
        return _get_moe_model_config_impl(
            config, topk, group_ids_trainable, expert_manager,
            full=full,
            routing_mode=routing_mode,
            group_ids_helper=group_ids_helper,
        )

else:
    raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND!r}")

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def load_pretrained_state_dict(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load pretrained state dict from a HuggingFace model path (local or hub).

    `dtype` selects the load-time tensor dtype. Default fp32 preserves
    historical behavior; callers under memory pressure (e.g. the partial
    loading path) should pass `torch.float16` / `torch.bfloat16` to halve
    the host RAM footprint of the transient state dict."""
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    state_dict = hf_model.state_dict()
    del hf_model
    logger.debug(
        "Loaded pretrained state dict",
        path=model_path,
        num_keys=len(state_dict),
        dtype=str(dtype),
    )
    return state_dict


def load_pretrained_model_low_mem(
    model_class: type[nn.Module],
    model_path: str,
    moe_config,
    model_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load pretrained weights directly into the custom model with low CPU memory usage."""
    model = model_class.from_pretrained(
        model_path,
        config=moe_config,
        # `torch_dtype` (not `dtype`): transformers 4.x has no `dtype` alias for
        # `from_pretrained`, so passing `dtype` there is silently ignored and the
        # model loads at the default fp32 — doubling host RAM on this low-mem path.
        torch_dtype=model_dtype,
        # Materialises shard by shard, so host RAM stays bounded by one shard
        # rather than a whole state dict (~31 GB for the 64-expert topology).
        # False previously because the custom `_load_from_state_dict` that
        # translated per-expert keys needed a materialised state dict; parameter
        # names are the checkpoint's own now, so nothing translates.
        low_cpu_mem_usage=True,
    )

    meta_expert_params = [
        name
        for name, param in model.named_parameters()
        if getattr(param, "is_meta", False) and ".mlp.experts." in name
    ]
    if meta_expert_params:
        logger.warning(
            "Detected unresolved meta expert tensors after from_pretrained; falling back to explicit state_dict load",
            meta_expert_param_count=len(meta_expert_params),
            sample_meta_expert_params=meta_expert_params[:8],
            path=model_path,
        )
        del model
        gc.collect()

        model = model_class(moe_config).to(dtype=model_dtype)
        pretrained_sd = load_pretrained_state_dict(model_path, dtype=model_dtype)
        model.load_state_dict(pretrained_sd, strict=False)
        del pretrained_sd

    logger.info("Loaded pretrained model directly", path=model_path, dtype=str(model_dtype))
    return model


def get_base_model(
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    group_ids_trainable: list | None = None,
    group_ids_helper: list | None = None,
    partial=False,
) -> nn.Module | None:
    """
    Load the pretrained base model for this role's expert topology.

    Miners and validators take the same path; `partial` and the group ids pick
    the routing top-k and which experts are declared.
    """
    precision = getattr(config.model, "precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        precision = "fp16-mixed"
    model_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16

    topk = config.moe.partial_topk if partial else config.moe.full_topk

    moe_config = get_moe_model_config(
        config, topk, group_ids_trainable, group_ids_helper, expert_manager, full = not partial,
    )

    # `partial` no longer selects which experts exist — the model declares the
    # experts its group assignment names, and `from_pretrained` fills them by
    # name. It picks the routing top-k and which group stays trainable.
    model = load_pretrained_model_low_mem(
        model_class=_CausalLMClass,
        model_path=config.model.model_path,
        moe_config=moe_config,
        model_dtype=model_dtype,
    )

    if model is not None and get_nested_attr(config, "model.torch_compile", False):
        model = torch.compile(model)

    return model


def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    """
    Load the tokenizer for `config.model.model_path`.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    # tokenizer.pad_token = "</s>"
    return tokenizer


def merge_state_dicts_with_priority(
    state_dicts: list[dict[str, torch.Tensor]],
    model: torch.nn.Module | None = None,
) -> tuple[OrderedDict, list[str] | None]:
    """
    Merge a list of state_dicts where earlier dicts have *higher* priority.
    Unexpected keys (not present in the model) are removed automatically.

    Args:
        state_dicts: list of state dicts, in priority order.
                     state_dicts[0] has highest priority, state_dicts[-1] lowest.
        model: optional model, used to filter out unexpected keys
               and check for missing keys.

    Returns:
        merged_state_dict: OrderedDict with cleaned + merged parameters.
        missing_keys: keys that the model expects but are not in merged
    """
    if not state_dicts:
        raise ValueError("state_dicts must be a non-empty list")

    merged = OrderedDict()

    # Build merged dict: earlier dicts override later ones.
    for sd in reversed(state_dicts):
        for k, v in sd.items():
            if k not in merged:
                merged[k] = v

    # If no model provided, return as is
    if model is None:
        return merged, None

    # Filter out unexpected keys
    model_keys = set(model.state_dict().keys())
    cleaned = OrderedDict((k, v) for k, v in merged.items() if k in model_keys)

    # Compute missing keys
    missing_keys = sorted(model_keys - set(cleaned.keys()))

    return cleaned, missing_keys
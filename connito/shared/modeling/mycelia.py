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
from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.expert_manager import ExpertManager
from connito.shared.helper import *
# ── Model backend selection ──────────────────────────────────────────────────
# Change MODEL_BACKEND to swap implementations:
#   "deepseek_v2"  → connito.shared.modeling.custom_deepseek_v2_lite
#   "qwen3_next"   → connito.shared.modeling.custom_qwen3_next
MODEL_BACKEND = "deepseek_v2"

if MODEL_BACKEND == "deepseek_v2":
    from connito.shared.modeling.custom_deepseek_v2_lite import (
        CustomDeekSeekMoE as _CausalLMClass,
        get_moe_model_config as _get_moe_model_config_impl,
    )

    def get_moe_model_config(config, topk, group_ids, expert_manager, full = False):
        return _get_moe_model_config_impl(config, topk, group_ids, expert_manager, full = full)

elif MODEL_BACKEND == "qwen3_next":
    from connito.shared.modeling.custom_qwen3_next import (
        CustomQwen3NextForCausalLM as _CausalLMClass,
        get_moe_model_config as _get_moe_model_config_impl,
    )

    def get_moe_model_config(config, topk, group_ids, expert_manager):
        return _get_moe_model_config_impl(config, topk, group_ids, expert_manager)

else:
    raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND!r}")

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def load_pretrained_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load pretrained state dict from a HuggingFace model path (local or hub)."""
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    state_dict = hf_model.state_dict()
    del hf_model
    logger.debug("Loaded pretrained state dict", path=model_path, num_keys=len(state_dict))
    return state_dict


def load_pretrained_model_low_mem(
    model_class: type[nn.Module],
    model_path: str,
    moe_config,
) -> nn.Module:
    """Load pretrained weights directly into the custom model with low CPU memory usage."""
    if bool(getattr(moe_config, "full", False)):
        logger.info(
            "Using direct explicit state_dict load for full model to avoid meta-tensor finalize issues",
            path=model_path,
        )
        model = model_class(moe_config)
        pretrained_sd = load_pretrained_state_dict(model_path)
        model.load_state_dict(pretrained_sd, strict=False)
        logger.info("Loaded full model via explicit state_dict", path=model_path, dtype="float16")
        return model

    model = model_class.from_pretrained(
        model_path,
        config=moe_config,
        dtype=torch.float16,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
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

        model = model_class(moe_config)
        pretrained_sd = load_pretrained_state_dict(model_path)
        model.load_state_dict(pretrained_sd, strict=False)

    logger.info("Loaded pretrained model directly", path=model_path, dtype="float16")
    return model



def get_base_model(
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    group_ids: list | None = None,
    partial=False,
) -> nn.Module | None:
    """
    Load base model with role-specific optimizations.

    Validators: Load with 4-bit quantization + Unsloth for memory efficiency
    Miners: Load standard model for training
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    topk = config.moe.partial_topk if partial else config.moe.full_topk
    model_path = config.model.model_path.lower()
    
    moe_config = get_moe_model_config(config, topk, group_ids, expert_manager, full = not partial)
        
    is_validator = config.role == "validator"
    use_quantization = get_nested_attr(config, "model.use_quantization", False) and is_validator
    use_unsloth = get_nested_attr(config, "model.use_unsloth", False) and is_validator

    # === QUANTIZED PATH (Validators only) ===
    if use_quantization:
        logger.info("Loading with 4-bit quantization for validator")

        # Try Unsloth first (fastest)
        if use_unsloth:
            try:
                from unsloth import FastLanguageModel

                model, _ = FastLanguageModel.from_pretrained(
                    model_name=config.model.model_path,
                    max_seq_length=moe_config.max_position_embeddings,
                    dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(model)
                logger.info("✓ Loaded with Unsloth optimizations")
                return model
            except Exception as e:
                logger.warning(f"Unsloth failed, falling back to BitsAndBytes: {e}")

        # Fallback to BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        max_memory = get_nested_attr(config, "model.max_memory", None)
        if max_memory is None:
            max_memory = {0: "46GB", "cpu": "100GB"}

        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_path,
            config=moe_config,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        logger.info("✓ Loaded with BitsAndBytes quantization")
        return model

    # === STANDARD PATH (Miners / non-quantized validators) ===
    # For full models, load directly into the custom class with low_cpu_mem_usage
    # to avoid materializing an extra full HF model + full intermediate state_dict.
    if not partial:
        model = load_pretrained_model_low_mem(
            model_class=_CausalLMClass,
            model_path=config.model.model_path,
            moe_config=moe_config,
        )
    else:
        model = _CausalLMClass(moe_config)

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
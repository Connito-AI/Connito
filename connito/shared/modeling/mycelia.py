from __future__ import annotations

import contextlib
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
from connito.shared.helper import get_nested_attr, resolve_model_dtype
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
        assignments_from_expert_modules as _assignments_from_expert_modules_impl,
        get_moe_model_config as _get_moe_model_config_impl,
        merge_group_assignments_for_streaming as _merge_group_assignments_for_streaming_impl,
        stream_pretrained_state_dict_to_partial_model as _stream_pretrained_to_partial_impl,
        stream_safetensors_to_partial_model as _stream_safetensors_to_partial_impl,
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


@contextlib.contextmanager
def build_at_dtype(dtype: torch.dtype):
    """Construct modules directly at `dtype` instead of fp32-then-cast.

    `nn.Parameter(torch.empty(...))` and `nn.Linear` take torch's *default*
    dtype, which is fp32. Building a model and casting afterwards therefore
    peaks at 2x its final size, because the fp32 storage for every parameter is
    live until that parameter is individually replaced.

    Measured on the full DeepSeek-V2-Lite topology (15.71 B params): 59.6 GB
    peak resident on a 62 GB host for a model whose final size is ~31 GB. The
    build alone was within a couple of GB of exhausting the machine, before the
    gradient buffers or any eval copy existed.

    Callers should keep their trailing `.to(dtype=...)`: `Tensor.to` returns
    self when the dtype already matches, so it costs nothing and the end state
    stays pinned to the same invariant if a submodule ever hard-codes fp32.

    `set_default_dtype` is process-global, so this must wrap construction only —
    never a region that yields to concurrent model building. Model construction
    happens once at startup on the main thread, which is why this is safe here
    and would not be inside the round loop.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


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
    if bool(getattr(moe_config, "full", False)):
        # Build at `model_dtype` from the start: a fp32 default + later cast
        # peaks at ~2x the final size for a 15B-param model. The trailing
        # `.to()` is a no-op once construction is already at the right dtype
        # (`Tensor.to` returns self) and is kept as the invariant's backstop.
        with build_at_dtype(model_dtype):
            model = model_class(moe_config)
        model = model.to(dtype=model_dtype)

        # Stream through the same per-key translator the partial path uses.
        # A bare `load_state_dict(strict=False)` here loaded NOTHING into the
        # routed experts: the checkpoint names them
        # `...experts.{N}.{gate,up,down}_proj.weight`, one tensor per expert per
        # projection, while `CustomDeepseekV2Experts` stores each layer's experts
        # stacked and gate/up fused, under `...experts.{N}.gate_up_proj`. Neither
        # side matched, `strict=False` swallowed both the 4992 unexpected keys
        # and the 52 missing ones, and every routed expert kept its
        # `_init_weights` random values — 14.39 B of 15.71 B parameters, with no
        # exception and no warning. Measured on an L40S: the model scored 9.956
        # against the stock HF model's 1.449 on identical batches.
        assignments = _assignments_from_expert_modules_impl(model)
        try:
            model = _stream_safetensors_to_partial_impl(
                model, model_path, assignments, model_dtype,
            )
        except Exception as exc:  # noqa: BLE001 - fall back, then report
            # Same fallback ladder as the partial path: a checkpoint with no
            # safetensors (or an unreadable index) still loads, just at the
            # cost of holding the state dict in host RAM.
            logger.warning(
                "Safetensors streaming unavailable for full model; falling back "
                "to an in-RAM state dict",
                path=model_path,
                error=f"{type(exc).__name__}: {exc}",
            )
            pretrained_sd = load_pretrained_state_dict(model_path, dtype=model_dtype)
            model = _stream_pretrained_to_partial_impl(model, pretrained_sd, assignments)
            del pretrained_sd

        logger.info(
            "Loaded full model",
            path=model_path,
            dtype=str(model_dtype),
            moe_layers=len(assignments),
            routed_experts=sum(len(v) for v in assignments.values()),
        )
        return model

    model = model_class.from_pretrained(
        model_path,
        config=moe_config,
        # `torch_dtype` (not `dtype`): transformers 4.x has no `dtype` alias for
        # `from_pretrained`, so passing `dtype` there is silently ignored and the
        # model loads at the default fp32 — doubling host RAM on this low-mem path.
        torch_dtype=model_dtype,
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

        with build_at_dtype(model_dtype):
            model = model_class(moe_config)
        model = model.to(dtype=model_dtype)
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
    Load the base model.

    Weights are always loaded at `model.precision` (fp16/bf16). Runtime fp8
    quantization, when enabled, is applied by the caller *after* loading — see
    `connito.shared.modeling.quantization`. It cannot be done here: the partial
    path below never goes through `from_pretrained`, so a transformers
    `quantization_config` would have no effect, and the streaming loaders rely
    on `state_dict()` returning live views into unquantized parameters.
    """
    model_dtype = resolve_model_dtype(config)

    topk = config.moe.partial_topk if partial else config.moe.full_topk

    moe_config = get_moe_model_config(
        config, topk, group_ids_trainable, group_ids_helper, expert_manager, full = not partial,
    )

    # For full models, load directly into the custom class with low_cpu_mem_usage
    # to avoid materializing an extra full HF model + full intermediate state_dict.
    if not partial:
        model = load_pretrained_model_low_mem(
            model_class=_CausalLMClass,
            model_path=config.model.model_path,
            moe_config=moe_config,
            model_dtype=model_dtype,
        )
    else:
        # Partial path: a bare `_CausalLMClass(moe_config)` would leave every
        # parameter at its random `_init_weights` default. Downstream
        # `load_checkpoint` only restores the active expert group, so the
        # backbone / embeddings / lm_head / attention / dense MLPs / shared
        # experts would all stay random.
        #
        # Strategy (keeps peak memory bounded for DeepSeek-V2-Lite — the
        # earlier "build full model on CPU + convert" approach peaked at
        # ~76 GB host RAM):
        #   1. Build the partial model and move it to its target device
        #      (typically GPU) at `model_dtype` before any pretrained data
        #      is loaded. Partial parameters live in VRAM; host RAM for
        #      the partial model returns to 0.
        #   2. Open each safetensors shard directly via `safe_open` and
        #      stream one tensor at a time into the partial model,
        #      slicing owned experts in the same per-key pass. The full
        #      state dict is never materialized in CPU RAM, so host RAM
        #      stays bounded by a single shard's file-backed mmap
        #      (~6 GB) plus one materialized tensor.
        #   3. Fall back to `load_pretrained_state_dict` + in-RAM
        #      streaming only if the safetensors-direct path is not
        #      available for the current backend.
        target_device = getattr(config.model, "device", "cpu")
        if target_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "config.model.device='cuda' but CUDA not available; keeping partial model on CPU",
            )
            target_device = "cpu"

        # Same fp32-then-cast trap as the full path, an order of magnitude
        # smaller: the partial model is ~4 GB, so the old peak was ~8 GB rather
        # than ~60 GB. Fixed here too because it is the same bug and the same
        # one-line remedy, not because it was hurting anything.
        with build_at_dtype(model_dtype):
            model = _CausalLMClass(moe_config)
        # Merge every loaded group's per-layer expert assignment into a single
        # mapping whose my_expert_ids match the merged partial model's local
        # slot layout (sorted-by-org_expert_id). This lets us stream pretrained
        # weights in ONE pass instead of once per group — no overlapping writes
        # from per-group my_expert_ids that both start at 0.
        stream_targets: list[int] = [
            *(int(gid) for gid in (group_ids_trainable or [])),
            *(int(gid) for gid in (group_ids_helper or [])),
        ]

        if stream_targets and _merge_group_assignments_for_streaming_impl is not None:
            merged_assignments = _merge_group_assignments_for_streaming_impl(
                expert_manager.expert_group_assignment,
                stream_targets,
            )
        else:
            merged_assignments = None

        if _stream_safetensors_to_partial_impl is not None and merged_assignments is not None:
            model = model.to(device=target_device, dtype=model_dtype)
            model = _stream_safetensors_to_partial_impl(
                partial_model=model,
                model_path=config.model.model_path,
                assignments=merged_assignments,
                dtype=model_dtype,
            )
            gc.collect()
            logger.info(
                "Streamed pretrained safetensors directly into partial model",
                stream_targets=stream_targets,
                num_layers=len(merged_assignments),
                device=str(target_device),
                dtype=str(model_dtype),
            )
        elif _stream_pretrained_to_partial_impl is not None and merged_assignments is not None:
            # Backend has a state-dict streaming helper but no safetensors-direct
            # path. Higher CPU peak (~30 GB for DeepSeek-V2-Lite) but still
            # better than the legacy full-model-on-CPU approach. Load once and
            # consume in one pass.
            model = model.to(device=target_device, dtype=model_dtype)
            pretrained_sd = load_pretrained_state_dict(
                config.model.model_path, dtype=model_dtype,
            )
            try:
                model = _stream_pretrained_to_partial_impl(
                    partial_model=model,
                    state_dict=pretrained_sd,
                    assignments=merged_assignments,
                )
            finally:
                del pretrained_sd
                gc.collect()
            logger.info(
                "Streamed pretrained backbone + owned-expert slices into partial model",
                stream_targets=stream_targets,
                num_layers=len(merged_assignments),
                device=str(target_device),
                dtype=str(model_dtype),
            )
        else:
            logger.warning(
                "Partial model returned with random weights — no pretrained-state-dict "
                "streaming helper available for this backend or no groups provided",
                backend=MODEL_BACKEND,
                group_ids_trainable=group_ids_trainable,
                group_ids_helper=group_ids_helper,
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



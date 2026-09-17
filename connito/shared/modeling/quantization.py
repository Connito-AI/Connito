"""Weight-only fp8 (e4m3) storage for modules that are never trained.

`FP8Linear` is ported from `~/experiment/partial_moe.py` so loss measurements
transfer between the two codebases.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from connito.shared.app_logging import structlog
from connito.shared.expert_manager import get_layer_expert_id
from connito.shared.helper import get_nested_attr

logger = structlog.get_logger(__name__)

# e4m3 not e5m2: 3 mantissa bits put the precision where weights need it.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

SCOPES = ("off", "attention", "experts", "attention+experts", "all")

# Matched by dotted-path *suffix*. As a substring, "mlp.gate" also hits
# "mlp.gate_proj" and would drop the dense MLPs.
#   lm_head            - feeds the loss directly, error is not averaged away
#   mlp.gate           - the router; perturbing it changes *which* experts fire
#   kv_a_proj_with_mqa - feeds an RMSNorm that amplifies weight error
DENY = ("lm_head", "mlp.gate", "kv_a_proj_with_mqa")

# q_a/q_b exist on full V2; Lite sets q_lora_rank=null and has q_proj instead.
ATTENTION = ("q_proj", "q_a_proj", "q_b_proj", "kv_b_proj", "o_proj")


def _to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric quantization. One global scale would let the
    largest row waste e4m3's mantissa for every other row."""
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (weight / scale).to(FP8_DTYPE), scale.float()


class FP8Linear(nn.Module):
    """A frozen Linear held in fp8 and dequantized per call.

    Stored as buffers, so `named_parameters()` never sees them and no optimizer
    can pick them up.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        values, scale = _to_fp8(linear.weight.data)
        self.register_buffer("weight_fp8", values)
        self.register_buffer("scale", scale)
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_fp8.to(x.dtype) * self.scale.to(x.dtype))

    def _apply(self, fn, recurse: bool = True):
        # fp8 is a floating dtype, so `.to(dtype=...)` upcasts the buffer and the
        # saving disappears with no error. Casting back is lossless.
        out = super()._apply(fn, recurse)
        if self.weight_fp8.dtype != FP8_DTYPE:
            self.weight_fp8.data = self.weight_fp8.data.to(FP8_DTYPE)
        if self.scale.dtype != torch.float32:
            self.scale.data = self.scale.data.float()
        return out

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Emit a dequantized `weight` so state_dict() matches an unquantized
        # model. fp32: a row with amax < ~1.3e-5 dequantizes to zeros in fp16.
        if keep_vars:
            raise RuntimeError("FP8Linear cannot honour state_dict(keep_vars=True)")
        destination[f"{prefix}weight"] = (
            self.weight_fp8.to("cpu", torch.float32) * self.scale.to("cpu")
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        key = f"{prefix}weight"
        if key not in state_dict:
            if strict:
                missing_keys.append(key)
            return
        incoming = state_dict.pop(key)
        if tuple(incoming.shape) != (self.out_features, self.in_features):
            error_msgs.append(f"size mismatch for {key}: got {tuple(incoming.shape)}")
            return
        values, scale = _to_fp8(incoming.to(self.scale.device))
        self.weight_fp8.copy_(values)
        self.scale.copy_(scale)


def _denied(name: str) -> bool:
    parts = name.split(".")
    return any(parts[-len(t.split(".")):] == t.split(".") for t in DENY)


def _in_scope(name: str, expert_id: int | None, scope: str) -> bool:
    if expert_id is not None:
        return scope in ("experts", "attention+experts", "all")
    if name.rsplit(".", 1)[-1] in ATTENTION:
        return scope in ("attention", "attention+experts", "all")
    return scope == "all"  # shared_experts and the dense MLP


def quantize_(model: nn.Module, scope: str, assignment: dict) -> list[str]:
    """Replace in-scope `nn.Linear` modules with `FP8Linear`, in place.

    `assignment` is `expert_group_assignment[group_id]` — the experts it names
    are this group's trainable set and are never converted. Selecting on the
    assignment rather than `requires_grad` keeps miner and validator on one rule,
    since only the miner marks anything frozen.
    """
    if scope not in SCOPES:
        raise ValueError(f"quantization scope must be one of {SCOPES}, got {scope!r}")
    if scope == "off":
        return []

    trainable = {
        (int(layer_id), int(org_expert_id))
        for layer_id, pairs in assignment.items()
        for _, org_expert_id in pairs
    }
    converted: list[str] = []

    # Names, not modules. Holding the module objects keeps every original
    # `nn.Linear` alive for the whole loop, so no replaced weight can be freed
    # while the fp8 copies accumulate and the peak is both sets at once. On the
    # 1664-expert topology that is 42.0 GiB against 29.3 GiB releasing as we go
    # — the difference between needing a 48 GB card and fitting on any of ours.
    #
    # Resolving each name late is safe only because every replacement is a leaf
    # `nn.Linear`: no captured name can name a module underneath one we swap.
    for parent_name in [name for name, _ in model.named_modules()]:
        parent = model.get_submodule(parent_name)
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            name = f"{parent_name}.{child_name}" if parent_name else child_name
            if _denied(name):
                continue
            layer_id, expert_id = get_layer_expert_id(name)
            if not _in_scope(name, expert_id, scope):
                continue
            if expert_id is not None and (layer_id, expert_id) in trainable:
                continue
            setattr(parent, child_name, FP8Linear(child))
            converted.append(name)

    logger.info("fp8 quantization applied", scope=scope, converted_modules=len(converted))
    return sorted(converted)


def unquantize_(model: nn.Module, weights: dict[str, torch.Tensor], dtype: torch.dtype) -> list[str]:
    """Put `FP8Linear` modules back to full-precision `nn.Linear`, in place.

    The inverse of `quantize_` in effect, not in mechanism: fp8 rounding is
    lossy, so the original weights have to be supplied. `weights` is keyed like
    a state dict (`....gate_proj.weight`), and an entry naming a module that is
    already `nn.Linear` is skipped — so handing it a whole group's tensors
    restores exactly the ones that are quantized and leaves the rest untouched.

    `dtype` is the model's, not the checkpoint's: boot loads pretrained weights
    at the run's precision, and a restored module has to match it.
    """
    restored: list[str] = []
    for key, tensor in weights.items():
        name = key.rsplit(".", 1)[0]          # drop the trailing `.weight`
        module = model.get_submodule(name)
        if not isinstance(module, FP8Linear):
            continue
        linear = nn.Linear(module.in_features, module.out_features, bias=False,
                           device=module.scale.device, dtype=dtype)
        with torch.no_grad():
            linear.weight.copy_(tensor)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, linear)
        restored.append(name)

    logger.info("fp8 modules restored to full precision", restored_modules=len(restored))
    return sorted(restored)


def apply_from_config(model: nn.Module, config, expert_manager, role: str) -> list[str]:
    """Quantize `model` per `model.quantization_<role>`. No-op unless the format
    switch `model.quantization` is "fp8"."""
    if get_nested_attr(config, "model.quantization", "off") != "fp8":
        return []
    group_id = config.task.exp.group_id
    return quantize_(
        model,
        get_nested_attr(config, f"model.quantization_{role}", "off"),
        expert_manager.expert_group_assignment.get(group_id, {}),
    )

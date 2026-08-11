"""Runtime weight-only int8 quantization.

Why this is hand-rolled rather than `bitsandbytes` / `torchao`:

* Production never loads weights through `from_pretrained` — both roles take the
  partial path in `mycelia.get_base_model`, which constructs a bare module and
  streams safetensors into it. A transformers `quantization_config` has no
  effect there.
* Both libraries only rewrite `nn.Linear`. The routed experts in
  `CustomDeepseekV2Experts` are stacked raw 3D `nn.Parameter`s consumed by
  `F.linear` on a slice, so either library would silently skip the tensors that
  dominate memory — a toggle that looks like it worked and saves nothing.

The representation is symmetric, weight-only, per-output-row int8. There is no
LLM.int8-style outlier threshold because that exists to protect *activation*
quantization, which we do not do.

The load-bearing property is that `state_dict()` is **dequantization
transparent**: a quantized module serialises exactly the keys, shapes and dtypes
it would have unquantized, and accepts plain fp16/bf16 on the way back in. That
is what lets the checkpoint hash contract (`helper.serialize_torch_model_path`),
the validator's per-miner graft (`evaluator.load_model_from_path`) and the save
path keep working untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

INT8_MAX = 127.0
_SCALE_EPS = 1e-12

# Attribute marker set on every module holding int8 weights. Used by
# `is_quantized` / `require_not_quantized` so callers can assert on the
# invariant without importing concrete classes (which would be circular).
QUANT_MARKER = "_connito_int8_quantized"

# Dotted-path *suffix* denylist applied to fully-qualified module names.
#
# * `lm_head` — output projection straight into the loss; int8 error here is
#   not averaged away by anything downstream.
# * `mlp.gate` — the MoE router. Tiny, and perturbing routing logits changes
#   *which* experts fire, a categorical error on top of a numerical one.
# * `kv_a_proj_with_mqa` — feeds an RMSNorm before `kv_b_proj`, which
#   renormalises and therefore amplifies small weight error. Excluded in v1;
#   measure adding it back once the rank-preservation gate has a baseline.
#
# Matching is by path suffix, not substring: `mlp.gate` as a substring also
# matches `mlp.gate_proj`, which would silently exclude the dense MLPs in the
# first `first_k_dense_replace` layers — the opposite of what is intended.
DEFAULT_LINEAR_DENYLIST: tuple[str, ...] = ("lm_head", "mlp.gate", "kv_a_proj_with_mqa")


def is_denied(qualified_name: str, denylist: tuple[str, ...]) -> bool:
    """True if *qualified_name* ends with any dotted path in *denylist*."""
    parts = qualified_name.split(".")
    for token in denylist:
        token_parts = token.strip(".").split(".")
        if len(token_parts) <= len(parts) and parts[-len(token_parts):] == token_parts:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────
def quantize_last_dim(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization along the final axis.

    Works uniformly for a 2D `nn.Linear` weight ``[out, in]`` (one scale per
    output row) and a stacked 3D expert tensor ``[experts, out, in]`` (one scale
    per expert per output row).

    Returns ``(int8_values, fp32_scales)`` where ``scales.shape ==
    weight.shape[:-1]``. The arithmetic is done in fp32 regardless of the input
    dtype: computing ``amax`` and the division in fp16 loses enough precision to
    visibly widen the round-trip error.
    """
    source = weight.detach().to(torch.float32)
    scale = source.abs().amax(dim=-1).clamp_min(_SCALE_EPS) / INT8_MAX
    values = (source / scale.unsqueeze(-1)).round_().clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    return values, scale


def dequantize_last_dim(
    values: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of `quantize_last_dim`, materialised at `dtype`."""
    return (values.to(torch.float32) * scale.unsqueeze(-1)).to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Int8Linear
# ─────────────────────────────────────────────────────────────────────────────
class Int8Linear(nn.Module):
    """Drop-in replacement for a frozen `nn.Linear` holding int8 weights.

    Dequantizes on use rather than running an int8 GEMM: the goal is resident
    memory, not FLOPs, and `torch._int_mm` is CUDA-only with shape constraints
    that the MLA projections do not all satisfy.

    Deliberately exposes no `weight` attribute. A property returning a freshly
    dequantized tensor would make `module.weight.data.copy_(...)` a silent
    no-op; an `AttributeError` is the safer failure.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: nn.Parameter | None,
        compute_dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        setattr(self, QUANT_MARKER, True)
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        # persistent=False keeps both out of `state_dict()` entirely, which is
        # what makes quantized <-> unquantized grafts symmetric in *both*
        # directions: an unquantized module never sees stray `_int8`/`_scale`
        # keys, and a quantized one never demands them.
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float32, device=device),
            persistent=False,
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int8Linear:
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias,
            compute_dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        values, scale = quantize_last_dim(linear.weight)
        module.weight_int8.copy_(values)
        module.weight_scale.copy_(scale)
        return module

    def dequantized_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_last_dim(self.weight_int8, self.weight_scale, dtype or self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_last_dim(self.weight_int8, self.weight_scale, x.dtype)
        bias = self.bias if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, int8=True"
        )

    def _apply(self, *args, **kwargs):
        # `Module.to(dtype=...)` casts every floating-point buffer, so probe
        # what it did to a throwaway tensor rather than trying to parse the
        # caller's arguments: whatever dtype a float lands in is the module's
        # new compute dtype, and `state_dict()` has to keep reporting that or
        # the graft path starts seeing dtype mismatches.
        probe = torch.zeros(1, dtype=self.compute_dtype)
        out = super()._apply(*args, **kwargs)
        try:
            probed = args[0](probe).dtype if args else probe.dtype
        except Exception:  # noqa: BLE001 - a fn that rejects our probe tells us nothing
            probed = self.compute_dtype
        if probed.is_floating_point:
            self.compute_dtype = probed
        # The int8 values are immune (torch skips the dtype on non-floating
        # tensors) but the scales are not, and demoting them to fp16
        # reintroduces exactly the precision loss the fp32 scale exists to avoid.
        if self.weight_scale.dtype != torch.float32:
            self.weight_scale.data = self.weight_scale.data.float()
        return out

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if keep_vars:
            raise RuntimeError(
                f"{type(self).__name__} cannot honour state_dict(keep_vars=True): the "
                f"'weight' entry is a dequantized copy, not a live view onto a parameter."
            )
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Dequantize onto CPU. `checkpoint_helper.save_checkpoint` calls
        # `model.state_dict()` before it shards or moves anything, so building
        # this on-device would spike VRAM by a full fp16 copy at every save —
        # on a miner that only fits *because* of int8, that is an OOM.
        destination[f"{prefix}weight"] = self.dequantized_weight().to("cpu")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        key = f"{prefix}weight"
        if key in state_dict:
            incoming = state_dict.pop(key)
            expected = (self.out_features, self.in_features)
            if tuple(incoming.shape) != expected:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{tuple(incoming.shape)} from checkpoint, the shape in current "
                    f"model is {expected}."
                )
            else:
                values, scale = quantize_last_dim(incoming.to(self.weight_int8.device))
                self.weight_int8.copy_(values)
                self.weight_scale.copy_(scale)
        elif strict:
            missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────
def quantize_linear_modules_(
    root: nn.Module,
    *,
    denylist: tuple[str, ...] = DEFAULT_LINEAR_DENYLIST,
    frozen_only: bool = True,
) -> list[str]:
    """Replace `nn.Linear` children of *root* with `Int8Linear`, in place.

    Selection is a denylist over fully-qualified names, not an allowlist of
    projection names: DeepSeek-V2-**Lite** sets ``q_lora_rank=null`` and so has
    ``q_proj`` where full V2 has ``q_a_proj``/``q_b_proj``, and an allowlist
    would silently miss whichever variant it was not written for. Embeddings and
    `DeepseekV2RMSNorm` are excluded for free by the `nn.Linear` test.

    With ``frozen_only`` (the default) a module whose weight still requires grad
    is skipped — int8 buffers receive no gradient, so quantizing a trainable
    weight would silently drop it out of both `named_parameters()` and the
    optimizer.

    Returns the sorted list of converted module names, for the caller to log.
    """
    converted: list[str] = []
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qualified = f"{name}.{child_name}" if name else child_name
            if is_denied(qualified, denylist):
                continue
            if frozen_only and child.weight.requires_grad:
                continue
            setattr(module, child_name, Int8Linear.from_linear(child))
            converted.append(qualified)
    return sorted(converted)


def quantize_model_(
    model: nn.Module,
    *,
    include_experts: bool,
    denylist: tuple[str, ...] = DEFAULT_LINEAR_DENYLIST,
    frozen_only: bool = True,
) -> list[str]:
    """Quantize *model* in place and return the converted module names.

    ``include_experts`` is the miner/validator split, and it is not a
    preference. On a validator eval model the whole module is frozen, so the
    stacked routed-expert tensors can be quantized wholesale — that is where
    the memory is. On a miner they cannot: each layer's stacked tensor
    interleaves the trainable group with the frozen helper group and
    `freeze_parameters` necessarily marks the whole thing trainable.

    Discovery of the expert modules is duck-typed (`quantize_` + a
    `_STACKED_PARAMS` attribute) rather than by isinstance, because the
    concrete class lives in a module that imports this one.
    """
    converted = quantize_linear_modules_(model, denylist=denylist, frozen_only=frozen_only)

    if include_experts:
        for name, module in model.named_modules():
            if not hasattr(module, "_STACKED_PARAMS") or not hasattr(module, "quantize_"):
                continue
            if frozen_only and any(p.requires_grad for p in module.parameters(recurse=False)):
                logger.debug("Skipping trainable expert module", module=name)
                continue
            module.quantize_()
            converted.append(name)

    return sorted(converted)


# ─────────────────────────────────────────────────────────────────────────────
# Invariants
# ─────────────────────────────────────────────────────────────────────────────
def is_quantized(model: nn.Module) -> bool:
    """True if any submodule holds int8 weights."""
    return any(getattr(module, QUANT_MARKER, False) for module in model.modules())


def quantized_module_names(model: nn.Module) -> list[str]:
    return sorted(
        name for name, module in model.named_modules() if getattr(module, QUANT_MARKER, False)
    )


def require_not_quantized(model: nn.Module, context: str) -> None:
    """Raise if *model* is quantized.

    Guards the paths that operate on live `nn.Parameter` objects rather than on
    `state_dict()`, where quantization does not raise but silently does nothing:

    * hivemind gradient pack/unpack and `populate_global_grads_from_local` walk
      `named_parameters()`, and an int8 weight is a *buffer* — merge would skip
      it with no exception and no missing-key warning, showing up only as
      slowly degrading vtrust.
    * the pretrained streaming loaders mutate the tensors returned by
      `state_dict()` in place, which works only because they are live views.
    """
    if is_quantized(model):
        raise RuntimeError(
            f"{context}: model holds int8 weights, but this path requires live "
            f"fp16/bf16 parameters. Quantized modules: "
            f"{quantized_module_names(model)[:8]}"
        )


def state_dict_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """`{key: shape}` for `model.state_dict()`, materialised at most once.

    On a quantized model `state_dict()` allocates a full dequantized copy, so
    callers that only need keys and shapes — the validator's per-miner
    compatibility diff, which runs once per miner against the *same* base model
    — must not call it repeatedly. Keys and shapes are fixed for a given
    architecture, so the result is cached on the module.
    """
    cached = getattr(model, "_connito_sd_shapes", None)
    if cached is None:
        cached = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        model._connito_sd_shapes = cached
    return cached

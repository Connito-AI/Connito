"""Weight-only fp8 (e4m3) storage for the frozen base.

Conversion runs inside `from_pretrained`, one checkpoint tensor at a time, so
the bf16 base (Qwen: 67 GiB) never has to fit in host or device memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.core_model_loading import ConversionOps
from transformers.quantizers import HfQuantizer, register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from connito.shared.app_logging import structlog
from connito.shared.helper import get_nested_attr
from connito.shared.modeling.families import FAMILIES, ModelFamily, family

logger = structlog.get_logger(__name__)

# e4m3 not e5m2: 3 mantissa bits put the precision where weights need it.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

# "experts": the family's fused routed experts only. "all": also every bias-free nn.Linear not denied.
SCOPES = ("off", "experts", "all")

# lm_head feeds the loss directly. No router entry: routers are excluded by type (not nn.Linear).
DENY = ("lm_head",)

_QUANT_METHOD = "connito_fp8"


def _to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quantization over the last dimension. One global scale
    would let the largest row waste e4m3's mantissa for every other row."""
    scale = weight.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) / FP8_MAX
    return weight.to(torch.float32, copy=True).div_(scale).to(FP8_DTYPE), scale


class _FP8Storage(nn.Module):
    """Frozen tensors held as fp8 buffers with fp32 scales.

    Buffers, so no optimizer sees them. Each keeps its checkpoint name, which
    is where `from_pretrained` writes the converted tensor.
    """

    _fp8_names: tuple[str, ...] = ()

    def _store(self, name: str, tensor: torch.Tensor) -> None:
        values, scale = _to_fp8(tensor)
        self.register_buffer(name, values)
        self.register_buffer(f"{name}_scale", scale)

    def _dequantized(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        return getattr(self, name).to(dtype).mul_(getattr(self, f"{name}_scale").to(dtype))

    def _apply(self, fn, recurse: bool = True):
        # `.to(dtype=...)` casts every floating buffer, fp8 values and fp32 scales alike: keep the stored ones.
        before = dict(self._buffers)
        out = super()._apply(fn, recurse)
        for name, old in before.items():
            if self._buffers[name].dtype != old.dtype:
                self._buffers[name] = old.to(self._buffers[name].device)
        return out

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # The unquantized names and shapes, so state_dict() matches a bf16 model.
        # On CPU, or a caller holding the dict would hold every dequantized
        # tensor on the GPU at once.
        if keep_vars:
            raise RuntimeError(f"{type(self).__name__} cannot honour state_dict(keep_vars=True)")
        for name in self._fp8_names:
            values = getattr(self, name)
            if values.is_meta:
                # Else transformers reads the checkpoint tensor in this dtype, not get_base_model's bf16.
                destination[f"{prefix}{name}"] = torch.empty_like(values, dtype=torch.bfloat16)
            else:
                destination[f"{prefix}{name}"] = values.to("cpu", torch.float32) * getattr(self, f"{name}_scale").cpu()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name in self._fp8_names:
            key = f"{prefix}{name}"
            if key not in state_dict:
                if strict:
                    missing_keys.append(key)
                continue
            incoming, values = state_dict.pop(key), getattr(self, name)
            if incoming.shape != values.shape:
                error_msgs.append(f"size mismatch for {key}: got {tuple(incoming.shape)}")
                continue
            new_values, new_scale = _to_fp8(incoming.to(values.device))
            values.copy_(new_values)
            getattr(self, f"{name}_scale").copy_(new_scale)


class FP8Linear(_FP8Storage):
    """A frozen bias-free Linear held in fp8 and dequantized per call."""

    _fp8_names = ("weight",)

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self._store("weight", linear.weight.data)
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._dequantized("weight", x.dtype))


class FP8FusedExperts(_FP8Storage):
    """A family's fused routed experts held in fp8, run through upstream's own forward."""

    _fp8_names = ("gate_up_proj", "down_proj")

    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        for name in self._fp8_names:
            self._store(name, getattr(experts, name).data)
        # Unregistered, so its meta parameters stay out of named_parameters() and
        # state_dict(); it carries the config-selected experts kernel, not weights.
        object.__setattr__(self, "_upstream", experts.to("meta"))

    def forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        # The whole layer, not only the experts hit: at seq 4096 every expert is hit.
        weights = {name: self._dequantized(name, hidden_states.dtype) for name in self._fp8_names}
        return torch.func.functional_call(
            self._upstream, weights, (hidden_states, top_k_index, top_k_weights)
        )


def quantize_(model: nn.Module, scope: str, fam: ModelFamily) -> None:
    """Replace the in-scope frozen modules with fp8 storage, in place."""
    if scope not in SCOPES:
        raise ValueError(f"quantization scope must be one of {SCOPES}, got {scope!r}")
    if scope == "off":
        return

    deny = DENY + fam.fp8_deny
    converted = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, fam.experts):
            model.set_submodule(name, FP8FusedExperts(module))
        elif (
            scope == "all" and type(module) is nn.Linear and module.bias is None
            and name.rsplit(".", 1)[-1] not in deny
        ):
            model.set_submodule(name, FP8Linear(module))
        else:
            continue
        converted += 1

    logger.info("fp8 quantization applied", scope=scope, converted_modules=converted)


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
                           device=module.weight_scale.device, dtype=dtype)
        with torch.no_grad():
            linear.weight.copy_(tensor)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, linear)
        restored.append(name)

    logger.info("fp8 modules restored to full precision", restored_modules=len(restored))
    return sorted(restored)


@register_quantization_config(_QUANT_METHOD)
class FP8StorageConfig(QuantizationConfigMixin):
    """Asks `from_pretrained` to store the base in fp8 under `scope` as it loads."""

    # The family by name: transformers keeps this object as `model.config.quantization_config`.
    def __init__(self, scope: str, family: str) -> None:
        self.quant_method = _QUANT_METHOD
        self.scope = scope
        self.family = family


class _QuantizeOnLoad(ConversionOps):
    def convert(self, input_dict, **kwargs) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in input_dict.items():
            # A renamed tensor arrives as a one-element list; one merged from per-expert keys as a tensor.
            out[key], out[f"{key}_scale"] = _to_fp8(value[0] if isinstance(value, list) else value)
        return out


@register_quantizer(_QUANT_METHOD)
class _FP8StorageQuantizer(HfQuantizer):
    def _process_model_before_weight_loading(self, model, **kwargs):
        cfg = self.quantization_config
        quantize_(model, cfg.scope, FAMILIES[cfg.family])

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        return isinstance(model.get_submodule(param_name.rpartition(".")[0]), _FP8Storage)

    def get_quantize_ops(self) -> ConversionOps:
        return _QuantizeOnLoad()

    def is_serializable(self) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return True


def fp8_load_config(config) -> FP8StorageConfig | None:
    """The `from_pretrained` quantization for this node's role; None loads bf16."""
    if get_nested_attr(config, "model.quantization", "off") != "fp8":
        return None
    scope = get_nested_attr(config, f"model.quantization_{config.role}", "off")
    return None if scope == "off" else FP8StorageConfig(scope, family(config).name)

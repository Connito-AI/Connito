from __future__ import annotations

import itertools
import json
import logging
import warnings
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedModel
from transformers.utils.hub import cached_file

# Suppress noisy upstream warnings from transformers/rope config (unless DEBUG)
if logging.root.level > logging.DEBUG:
    warnings.filterwarnings("ignore", message=r".*rope_parameters.*")
    logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)

from connito.shared.app_logging import structlog
from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.expert_manager import get_layer_expert_id
from connito.shared.modeling.families import family
from connito.shared.modeling.quantization import fp8_load_config

logger = structlog.get_logger(__name__)


def load_pretrained_expert_tensors(
    config: MinerConfig | ValidatorConfig, layer_map: dict[int, list[tuple[int, int]]],
) -> dict[str, torch.Tensor]:
    """The checkpoint's own tensors for the experts one group's table names,
    read by name from its safetensors files: no model and no full state
    dict, so a running process can write another group's pretrained shard
    without a second 30 GB build."""
    fam = family(config)
    wanted = {(int(layer), int(org)) for layer, pairs in layer_map.items() for _, org in pairs}
    index_path = cached_file(fam.model_path, "model.safetensors.index.json", revision=fam.revision)
    index = json.loads(Path(index_path).read_text())
    names_by_file: dict[str, list[str]] = {}
    for name, file in index["weight_map"].items():
        if get_layer_expert_id(name) in wanted:
            names_by_file.setdefault(file, []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for file, names in names_by_file.items():
        with safe_open(cached_file(fam.model_path, file, revision=fam.revision), framework="pt") as f:
            tensors.update({name: f.get_tensor(name) for name in names})
    return tensors


def get_base_model(config: MinerConfig | ValidatorConfig) -> PreTrainedModel:
    """The text-only causal LM of the family's pinned checkpoint, in bf16."""
    fam = family(config)
    # transformers drops Qwen's language_model. prefix and visual.*/mtp.*, and fuses V2's per-expert tensors.
    model, info = fam.causal_lm.from_pretrained(
        fam.model_path, revision=fam.revision, dtype=torch.bfloat16, output_loading_info=True,
        quantization_config=fp8_load_config(config),
    )
    # transformers only warns, and leaves a missing tensor randomly initialised.
    if info["missing_keys"]:
        raise ValueError(f"checkpoint is missing {sorted(info['missing_keys'])}")
    # Keeps Qwen's router aux loss out of `outputs.loss`; after load, as DeepSeek-V2's __init__ rejects the kwarg.
    model.config.output_router_logits = False
    _refuse_split(model)
    logger.info("Loaded base model", family=fam.name, path=fam.model_path, revision=fam.revision)
    return model


def _refuse_split(model: PreTrainedModel) -> None:
    """Raise unless the whole model sits on one device."""
    mapped = set(getattr(model, "hf_device_map", {}).values())
    placed = {t.device for t in itertools.chain(model.parameters(), model.buffers())}
    if len(mapped) > 1 or len(placed) > 1:
        raise RuntimeError(
            f"the base model is split over devices {sorted(map(str, mapped | placed))}. Load it onto one GPU: "
            "split over several GPUs (transformers device_map), Qwen gives wrong outputs."
        )


def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    fam = family(config)
    # transformers 5 ignores DeepSeek-V2's own add_bos_token: true (its tokenizer.json has no BOS template).
    return AutoTokenizer.from_pretrained(fam.model_path, revision=fam.revision, add_bos_token=fam.add_bos_token)

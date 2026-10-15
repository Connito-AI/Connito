from __future__ import annotations

import itertools
import logging
import warnings

import torch
from transformers import AutoTokenizer, PreTrainedModel

# Suppress noisy upstream warnings from transformers/rope config (unless DEBUG)
if logging.root.level > logging.DEBUG:
    warnings.filterwarnings("ignore", message=r".*rope_parameters.*")
    logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)

from connito.shared.app_logging import structlog
from connito.shared.config import MinerConfig, ValidatorConfig
from connito.shared.modeling.families import family
from connito.shared.modeling.quantization import fp8_load_config

logger = structlog.get_logger(__name__)


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

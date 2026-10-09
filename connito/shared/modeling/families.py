"""The model families a node can run, as data.

Each record owns its checkpoint id and pinned revision, so switching model is
the one locked `model.family` config value.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel
from transformers.models.deepseek_v2 import DeepseekV2ForCausalLM
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

from connito.shared.config import MinerConfig, ValidatorConfig


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_path: str
    revision: str
    causal_lm: type[PreTrainedModel]
    add_bos_token: bool


QWEN3_5_MOE = ModelFamily(
    name="qwen3_5_moe",
    model_path="Qwen/Qwen3.6-35B-A3B",
    revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    causal_lm=Qwen3_5MoeForCausalLM,
    add_bos_token=False,
)
DEEPSEEK_V2 = ModelFamily(
    name="deepseek_v2",
    model_path="deepseek-ai/DeepSeek-V2-Lite",
    revision="604d5664dddd88a0433dbae533b7fe9472482de0",
    causal_lm=DeepseekV2ForCausalLM,
    add_bos_token=True,
)
FAMILIES = {f.name: f for f in (QWEN3_5_MOE, DEEPSEEK_V2)}


def family(config: MinerConfig | ValidatorConfig) -> ModelFamily:
    return FAMILIES[config.model.family]

"""The model families a node can run, as data.

Each record owns its checkpoint id and pinned revision, so switching model is
the one locked `model.family` config value.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from transformers import PreTrainedModel
from transformers.models.deepseek_v2 import DeepseekV2ForCausalLM
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Experts, DeepseekV2MLP, DeepseekV2Moe
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)

from connito.shared.config import MinerConfig, ValidatorConfig


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_path: str
    revision: str
    causal_lm: type[PreTrainedModel]
    add_bos_token: bool
    # The fused 3-D routed experts, stored in fp8.
    experts: type[nn.Module]
    # Linear leaf names kept full precision under the "all" scope.
    fp8_deny: tuple[str, ...]
    # The MoE block residual experts attach to; dense layers are other types.
    moe_block: type[nn.Module]
    # The dense SwiGLU MLP each residual expert is, built with `intermediate_size=`.
    mlp: type[nn.Module]


QWEN3_5_MOE = ModelFamily(
    name="qwen3_5_moe",
    model_path="Qwen/Qwen3.6-35B-A3B",
    revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    causal_lm=Qwen3_5MoeForCausalLM,
    add_bos_token=False,
    experts=Qwen3_5MoeExperts,
    fp8_deny=("shared_expert_gate",),  # one row that scales the whole shared expert
    moe_block=Qwen3_5MoeSparseMoeBlock,
    mlp=Qwen3_5MoeMLP,
)
DEEPSEEK_V2 = ModelFamily(
    name="deepseek_v2",
    model_path="deepseek-ai/DeepSeek-V2-Lite",
    revision="604d5664dddd88a0433dbae533b7fe9472482de0",
    causal_lm=DeepseekV2ForCausalLM,
    add_bos_token=True,
    experts=DeepseekV2Experts,
    fp8_deny=("kv_a_proj_with_mqa",),  # feeds an RMSNorm that amplifies weight error
    moe_block=DeepseekV2Moe,
    mlp=DeepseekV2MLP,
)
FAMILIES = {f.name: f for f in (QWEN3_5_MOE, DEEPSEEK_V2)}


def family(config: MinerConfig | ValidatorConfig) -> ModelFamily:
    return FAMILIES[config.model.family]

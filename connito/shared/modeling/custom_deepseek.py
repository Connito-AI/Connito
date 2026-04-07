from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    PretrainedConfig,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)

from connito.shared.app_logging import structlog
from connito.shared.config import MinerConfig
from connito.shared.helper import *

logger = structlog.get_logger(__name__)


class TopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config, available_experts=None):
        super().__init__(config)
        if available_experts is not None:
            self.available_experts = torch.as_tensor(available_experts)

        self.weight = nn.Parameter(torch.zeros((self.n_routed_experts, config.hidden_size)))

    def _mask_routing_weights(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Zero-out routing weights for experts not present in this group.
        """
        mask_1d = torch.zeros(x.size(dim), dtype=torch.bool, device=x.device)
        mask_1d[self.available_experts.to(x.device)] = True

        # Broadcast mask across all other dims
        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        mask = mask_1d.view(shape).to(dtype=x.dtype)
        return x * mask

    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        scores_for_choice = self._mask_routing_weights(scores_for_choice)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices


class SparseMoeBlock(DeepseekV3MoE):
    """
    Sparse MoE block that only uses a subset of experts assigned to the current group.

    Parameters
    ----------
    config : DeepseekV3Config-like
        Must provide: hidden_size, num_experts, num_experts_per_tok, norm_topk_prob.
    group_ids : int
        The group this rank belongs to.
    expert_group_assignment : Dict[int, Dict[int, list[int]]]
        Mapping: layer_id -> (group_id -> list of expert ids available to that group).
    layer_id : int
        The layer index used to pick the allowed experts.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        num_experts: int | None = None,
        group_ids: int | None = None,
        expert_group_assignment: dict[int, dict[int, list[int]]] | None = None,
    ):
        super().__init__(config)
        self.num_experts: int = config.num_experts
        self.top_k: int = config.num_experts_per_tok
        self.norm_topk_prob: bool = getattr(config, "norm_topk_prob", True)
        self.layer_id: int = layer_id
        self.expert_group_assignment = expert_group_assignment

        # Only instantiate experts owned by this group at this layer
        if group_ids is not None and expert_group_assignment is not None:
            allowed = expert_group_assignment[layer_id][group_ids]

        elif num_experts is not None:
            allowed = list(range(num_experts))

        self.experts = nn.ModuleDict(
            {str(k): DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) for k in allowed}
        )

        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

        self.available_experts = torch.as_tensor([int(k) for k in self.experts.keys()])

        self.gate = TopkRouter(config, self.available_experts)


class CustomDeekSeekMoE(DeepseekV3ForCausalLM):
    """
    DeepseekV3 variant that interleaves MoE and dense blocks and optionally restricts experts
    to those owned by the calling group.

    If `partial=True`, MoE blocks become `myceliaSparseMoeBlock` limited to the group’s experts.
    Otherwise, standard `SparseMoeBlock` is used for MoE layers.
    """

    def __init__(
        self,
        config: MinerConfig,
        model_config: PretrainedConfig,
        group_ids: int | None = None,
        expert_group_assignment: dict[int, dict[int, list[int]]] | None = None,
        partial: bool = False,
    ):
        super().__init__(model_config)
        layers: list[nn.Module] = []

        for i in range(model_config.num_hidden_layers):
            layer = DeepseekV3DecoderLayer(model_config, layer_idx=i)

            # layer.self_attn = DeepseekAttention(model_config)

            # Interleave MoE and dense layers (MoE on odd indices if interleave=True)
            if getattr(model_config, "interleave", True) and (i + 1) % model_config.decoder_sparse_step == 0:
                if not partial:
                    layer.mlp = SparseMoeBlock(
                        model_config,
                        i,
                        num_experts=config.moe.num_experts,
                        expert_group_assignment=expert_group_assignment,
                    )  # full MoE (all experts)
                else:
                    assert (
                        group_ids is not None and expert_group_assignment is not None
                    ), "partial=True requires group_ids and expert_group_assignment"
                    layer.mlp = SparseMoeBlock(
                        model_config, i, group_ids=group_ids, expert_group_assignment=expert_group_assignment
                    )

            elif i == 0:
                layer.mlp = DeepseekV3MLP(model_config, intermediate_size=10944)

            else:
                layer.mlp = DeepseekV3MLP(model_config)

            layers.append(layer)

        self.model.layers = nn.ModuleList(layers)


def get_moe_model_config(config: MinerConfig, topk: int, org_model_config: AutoConfig = None) -> PretrainedConfig:
    # get the base config from qwen model
    base_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")  # TODO: need user permission
    # base_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True) #TODO: need user permission

    # merge the existing model config into the base config
    if org_model_config is not None:
        for k, v in org_model_config.to_dict().items():
            setattr(base_config, k, v)

    # merge our subnet config to the base config
    base_config.num_experts = int(config.moe.num_experts)
    base_config.n_routed_experts = int(config.moe.num_experts)
    base_config.n_group = config.moe.num_worker_groups
    base_config.topk_group = 1
    base_config.num_experts_per_tok = int(topk)
    base_config.interleave = bool(config.moe.interleave)
    base_config.intermediate_size = base_config.moe_intermediate_size
    base_config.decoder_sparse_step = 2 if bool(config.moe.interleave) else 1
    base_config.output_router_logits = get_nested_attr(config, "moe.aux_load_balance", False)
    base_config.router_aux_loss_coef = get_nested_attr(config, "moe.router_aux_loss_coef", False)
    base_config.norm_topk_prob = True
    base_config.max_position_embeddings = config.task.exp.data.sequence_length

    return base_config

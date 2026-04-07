from __future__ import annotations

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_next.configuration_qwen3_next import (
    Qwen3NextConfig,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    OutputRecorder,
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
    Qwen3NextSparseMoeBlock,
)

from connito.shared.app_logging import structlog
from connito.shared.config import MinerConfig
from connito.shared.expert_manager import ExpertManager
from connito.shared.helper import *

logger = structlog.get_logger(__name__)


class TopKRouter(nn.Module):
    def __init__(self, config, available_experts=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        if available_experts is not None:
            self.available_experts = torch.as_tensor(available_experts)

    def _mask_routing_weights(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Zero-out routing weights for experts not present in this group.
        """
        # mask_1d = torch.full((x.size(dim),), float('-inf'), device=x.device)
        mask_1d = torch.zeros(x.size(dim), dtype=torch.bool, device=x.device)
        mask_1d[self.available_experts.to(x.device)] = True

        # Broadcast mask across all other dims
        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        mask = mask_1d.view(shape).to(dtype=x.dtype)

        mask_bool = mask.to(torch.bool)  # True = keep
        fill = max(x.min().item() - 1e-6, 0)  # scalar, computed BEFORE masking
        x_masked = x.masked_fill(~mask_bool, fill)
        return x_masked

    def forward(self, hidden_states):
        router_logits = self.weight(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # a
        routing_weights = self._mask_routing_weights(routing_weights)  # b
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts


@torch._dynamo.disable
def _compute_overlap(expert_hit, available_experts):
    expert_hit_set = set(expert_hit.detach().cpu().flatten().tolist())
    available_experts_set = set(available_experts.tolist())
    return torch.tensor(sorted(expert_hit_set.intersection(available_experts_set))).view(-1, 1)


class SparseMoeBlock(Qwen3NextSparseMoeBlock):
    def __init__(
        self,
        config,
        layer_id: int,
    ):
        super().__init__(config)

        if config.expert_group_assignment is not None:
            if config.group_ids is None:
                group_ids = config.expert_group_assignment.keys()
            else:
                group_ids = config.group_ids

            allowed_expert_id = []
            for group_id in group_ids:
                allowed_expert_id += [
                    my_expert_id for my_expert_id, org_expert_id in config.expert_group_assignment[group_id][layer_id]
                ]
        else:
            allowed_expert_id = list(range(config.num_experts))

        self.available_experts = torch.as_tensor([int(k) for k in allowed_expert_id])

        self.gate = TopKRouter(config, self.available_experts)

        self.experts = nn.ModuleDict(
            {str(i): Qwen3NextMLP(config, intermediate_size=config.moe_intermediate_size) for i in allowed_expert_id}
        )

        self.num_experts = config.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        qualified_expert_set = _compute_overlap(expert_hit, self.available_experts)

        for expert_idx in qualified_expert_set:
            expert_layer = self.experts[str(expert_idx.item())]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_hidden_state = expert_layer(current_state)
            current_hidden_states = expert_hidden_state * routing_weights[top_x, idx, None]
            # `index_add_` only supports torch tensors for indexing so we use `top_x`
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits


class DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
    ):
        super().__init__(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = SparseMoeBlock(
                config,
                layer_id=layer_idx,
            )


class CustomPreTrainedModel(Qwen3NextPreTrainedModel):
    config: Qwen3NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3NextSparseMoeBlock, index=1),
        "hidden_states": DecoderLayer,
        "attentions": Qwen3NextAttention,
    }
    _is_stateful = True


class CustomQwen3NextModel(Qwen3NextModel):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])


class CustomQwen3NextForCausalLM(Qwen3NextForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen3NextModel(config)


def get_moe_model_config(
    config: MinerConfig, topk: int, group_ids: list | None, expert_manager: ExpertManager
) -> PretrainedConfig:
    # get the base config from qwen model
    base_config = AutoConfig.from_pretrained(config.model.model_path)

    # full/partial dependent configuration
    base_config.num_experts_per_tok = int(topk)
    base_config.group_ids = group_ids  # in list, cause you may load a partial model that contains multiple group id

    # merge our subnet config to the base config
    base_config.n_group = config.moe.num_worker_groups
    base_config.max_position_embeddings = config.task.exp.data.sequence_length
    base_config.num_experts = (
        expert_manager.num_experts
    )  # this stays the same regardless of full/partial cause we keep the same router size either case
    base_config.output_router_logits = get_nested_attr(config, "moe.aux_load_balance", False)
    base_config.router_aux_loss_coef = get_nested_attr(config, "moe.router_aux_loss_coef", False)
    base_config.expert_group_assignment = expert_manager.expert_group_assignment

    return base_config

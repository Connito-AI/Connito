from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import re

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    PretrainedConfig,
)
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_file,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    ACT2FN,
    DeepseekV2Attention,
    DeepseekV2Config,
    DeepseekV2DecoderLayer,
    DeepseekV2Experts,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2Moe,
    DeepseekV2PreTrainedModel,
    DeepseekV2RotaryEmbedding,
    DeepseekV2RMSNorm,
)

from connito.shared.app_logging import structlog
from connito.shared.helper import *

from connito.shared.expert_manager import get_layer_expert_id

if TYPE_CHECKING:
    from connito.shared.config import MinerConfig
    from connito.shared.expert_manager import ExpertManager
else:
    MinerConfig = Any

logger = structlog.get_logger(__name__)
_EXPERT_PREFIX_LAYER_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.$")


def _validate_assignment_bounds(
    expert_group_assignment: dict[int, dict[int, list[tuple[int, int]]]],
    num_experts: int,
    num_hidden_layers: int,
    group_ids: list[int] | None = None,
) -> None:
    """Validate expert assignment indices used by DeepSeek-V2-Lite.

    Only model layers in [0, num_hidden_layers) are validated.
    """
    errors: list[str] = []

    groups_to_validate: list[int]
    if group_ids is None:
        groups_to_validate = sorted(expert_group_assignment.keys())
    else:
        groups_to_validate = sorted({int(group_id) for group_id in group_ids})

    for group_id in groups_to_validate:
        layer_assignments = expert_group_assignment.get(group_id)
        if layer_assignments is None:
            errors.append(f"group={group_id}: missing from expert_group_assignment")
            continue

        for layer_id, mappings in layer_assignments.items():
            if layer_id < 0 or layer_id >= num_hidden_layers:
                continue

            for mapping in mappings:
                if len(mapping) != 2:
                    errors.append(
                        f"group={group_id}, layer={layer_id}: invalid mapping format {mapping!r}"
                    )
                    continue

                my_expert_id, org_expert_id = int(mapping[0]), int(mapping[1])

                if not (0 <= my_expert_id < num_experts):
                    errors.append(
                        f"group={group_id}, layer={layer_id}: my_expert_id={my_expert_id} out of range [0, {num_experts - 1}]"
                    )
                if not (0 <= org_expert_id < num_experts):
                    errors.append(
                        f"group={group_id}, layer={layer_id}: org_expert_id={org_expert_id} out of range [0, {num_experts - 1}]"
                    )

    if errors:
        preview = "\n".join(errors[:10])
        if len(errors) > 10:
            preview += f"\n... and {len(errors) - 10} more"
        raise ValueError(
            "Invalid expert_assignment indices for DeepSeek-V2-Lite. "
            "All my_expert_id/org_expert_id must be within model routed-expert bounds.\n"
            f"{preview}"
        )


class CustomDeepseekV2Experts(DeepseekV2Experts):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config, expert_indices):
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.num_local_experts = len(expert_indices)
        self.expert_indices = expert_indices
        self.first_moe_layer = int(getattr(config, "first_k_dense_replace", 0))
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        self.last_moe_layer = int(num_hidden_layers) - 1 if num_hidden_layers is not None else None

        # Map Global ID -> Local Tensor Index
        self.global_to_local_map = torch.full((config.num_experts,), -1, dtype=torch.long)

        for local_idx, global_idx in enumerate(expert_indices):
            self.global_to_local_map[global_idx] = local_idx

        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_dim, self.intermediate_dim))
        self._reset_expert_parameters(config)
        self.act_fn = ACT2FN[config.hidden_act]

    def _reset_expert_parameters(self, config: DeepseekV2Config) -> None:
        """Initialize stacked expert parameters to finite values."""
        init_std = float(getattr(config, "initializer_range", 0.02))
        with torch.no_grad():
            nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
            nn.init.normal_(self.down_proj, mean=0.0, std=init_std)

    # ── State-dict compatibility ──────────────────────────────────────────
    # The stacked parameters (shape [num_local_experts, ...]) are serialised
    # as individual per-expert slices so that checkpoints look like:
    #   prefix.experts.0.gate_up_proj   shape [2*D, H]
    #   prefix.experts.1.gate_up_proj   shape [2*D, H]
    #   ...
    # This lets us load weights saved from per-expert Linear modules directly.

    # nn.Parameter names (no .weight suffix — these are raw 3D tensors, not nn.Linear)
    _STACKED_PARAMS = ("gate_up_proj", "down_proj")

    # ── Save: expand stacked 3D tensors into per-expert slices ────────────
    # Override _save_to_state_dict (not state_dict) so expansion works
    # when called from the parent model's state_dict() traversal.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name in self._STACKED_PARAMS:
            stacked_key = f"{prefix}{name}"
            if stacked_key not in destination:
                continue
            stacked = destination.pop(stacked_key)
            for local_idx, global_idx in enumerate(self.expert_indices):
                destination[f"{prefix}{global_idx}.{name}"] = stacked[local_idx]

    # ── Load: overlay per-expert checkpoint weights onto existing stacked tensors ──
    # Uses the current model weights as defaults. For each expert found in the
    # checkpoint, replaces that expert's slice in the stacked tensor.
    # Checkpoint format: prefix.{idx}.gate_up_proj / prefix.{idx}.down_proj
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        logger.debug(
            "_load_from_state_dict called",
            prefix=prefix,
            expert_indices=self.expert_indices,
            num_state_dict_keys=len(state_dict),
        )
        merge_events: list[dict[str, int | float | str]] = []
        for name in self._STACKED_PARAMS:
            stacked_key = f"{prefix}{name}"
            if stacked_key in state_dict:
                incoming = state_dict[stacked_key]
                target_param = getattr(self, name)
                incoming_shape = tuple(incoming.shape)
                target_shape = tuple(target_param.shape)

                if incoming_shape == target_shape:
                    logger.debug(
                        "stacked key present with matching shape",
                        stacked_key=stacked_key,
                        shape=incoming_shape,
                    )
                    continue

                # Adapt full stacked tensor [global_experts, ...] to this module's local experts.
                if (
                    incoming.ndim == target_param.ndim
                    and incoming_shape[1:] == target_shape[1:]
                    and len(self.expert_indices) == target_shape[0]
                ):
                    if all(0 <= int(expert_id) < incoming_shape[0] for expert_id in self.expert_indices):
                        adapted = incoming[self.expert_indices].to(
                            device=target_param.device,
                            dtype=target_param.dtype,
                        )
                        state_dict[stacked_key] = adapted
                        logger.info(
                            "Adapted stacked expert tensor using global expert indices",
                            stacked_key=stacked_key,
                            source_shape=incoming_shape,
                            target_shape=target_shape,
                            sample_expert_indices=self.expert_indices[:8],
                        )
                        continue

                    # Backward compatibility for older local-id checkpoints.
                    if incoming_shape[0] >= target_shape[0]:
                        adapted = incoming[: target_shape[0]].to(
                            device=target_param.device,
                            dtype=target_param.dtype,
                        )
                        state_dict[stacked_key] = adapted
                        logger.warning(
                            "Adapted stacked expert tensor using local-index fallback",
                            stacked_key=stacked_key,
                            source_shape=incoming_shape,
                            target_shape=target_shape,
                        )
                        continue

                logger.warning(
                    "Dropping incompatible stacked expert tensor",
                    stacked_key=stacked_key,
                    source_shape=incoming_shape,
                    target_shape=target_shape,
                )
                state_dict.pop(stacked_key, None)
                continue

            # Start from the current (base) model weights as default
            base_param = getattr(self, name)  # [num_local_experts, ...]
            stacked = base_param.data.clone()
            found_any = False
            loaded_experts = []

            for local_idx, global_idx in enumerate(self.expert_indices):
                expert_key = f"{prefix}{global_idx}.{name}"
                if expert_key in state_dict:
                    stacked[local_idx] = state_dict.pop(expert_key)
                    found_any = True
                    loaded_experts.append(global_idx)


            if found_any:
                state_dict[stacked_key] = stacked
                # Explicitly write to the parameter as well
                param = getattr(self, name)
                org_param_sum = param.sum()
                param.data.copy_(stacked)
                merge_events.append(
                    {
                        "param": name,
                        "loaded_count": len(loaded_experts),
                        "loaded_first": loaded_experts[0],
                        "loaded_last": loaded_experts[-1],
                        "prev": round(org_param_sum.item(), 4),
                        "new": round(param.sum().item(), 4),
                    }
                )

        if merge_events:
            prefix_match = _EXPERT_PREFIX_LAYER_RE.search(prefix)
            layer_id = int(prefix_match.group(1)) if prefix_match is not None else None
            should_log_merge = True
            first_or_last = "single"
            if layer_id is not None and self.last_moe_layer is not None:
                should_log_merge = layer_id in {self.first_moe_layer, self.last_moe_layer}
                if layer_id == self.first_moe_layer and layer_id == self.last_moe_layer:
                    first_or_last = "single"
                elif layer_id == self.first_moe_layer:
                    first_or_last = "first"
                elif layer_id == self.last_moe_layer:
                    first_or_last = "last"

            if should_log_merge:
                logger.debug(
                    "merged expert weights into param",
                    **merge_events[0],
                    position=first_or_last,
                    layer_id=layer_id,
                    total_merged=len(merge_events),
                )

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for global_expert_idx in expert_hit:
        
            global_expert_idx = global_expert_idx[0].item()

            local_expert_idx = self.global_to_local_map[global_expert_idx]
            
            if local_expert_idx == -1:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[global_expert_idx])

            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[local_expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[local_expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
    
class CustomDeepseekV2Moe(DeepseekV2Moe):
    def __init__(self, config: DeepseekV2Config, layer_id: int | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.num_experts = config.n_routed_experts

        full_mode = bool(getattr(config, "full", False))

        # --- Determine allowed experts ---
        if full_mode:
            allowed_expert_id = list(range(config.n_routed_experts))
        elif config.expert_group_assignment is not None:
            group_ids = config.group_ids if config.group_ids is not None else config.expert_group_assignment.keys()
            allowed_expert_id = []
            for group_id in group_ids:
                group_id = int(group_id)
                layer_assignments = config.expert_group_assignment[group_id].get(layer_id, [])
                allowed_expert_id += [int(org_expert_id) for _, org_expert_id in layer_assignments]
        else:
            total_experts = getattr(config, "num_experts", None)
            if total_experts is None:
                total_experts = getattr(config, "n_routed_experts")
            allowed_expert_id = list(range(total_experts))

        available_experts = sorted({int(expert_id) for expert_id in allowed_expert_id})
        invalid_experts = [
            expert_id
            for expert_id in available_experts
            if not (0 <= expert_id < config.n_routed_experts)
        ]
        if invalid_experts:
            raise ValueError(
                "Detected out-of-range expert ids in allowed_expert_id for layer routing. "
                f"layer_id={layer_id}, group_ids={config.group_ids}, "
                f"invalid={invalid_experts[:10]}"
            )
        if len(available_experts) == 0:
            raise ValueError(
                f"No routed experts assigned for layer_id={layer_id}. "
                f"group_ids={config.group_ids}"
            )
        if full_mode and len(available_experts) != config.n_routed_experts:
            raise ValueError(
                "Full model mode must include all routed experts. "
                f"layer_id={layer_id}, expected={config.n_routed_experts}, got={len(available_experts)}"
            )

        self.expert_indices = available_experts
        self.register_buffer("allowed_ids", torch.tensor(self.expert_indices, dtype=torch.long), persistent=False)

        first_moe_layer = int(getattr(config, "first_k_dense_replace", 0))
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        last_moe_layer = int(num_hidden_layers) - 1 if num_hidden_layers is not None else None

        should_log_layout = True
        layout_position = "single"
        if layer_id is not None and last_moe_layer is not None:
            should_log_layout = layer_id in {first_moe_layer, last_moe_layer}
            if layer_id == first_moe_layer and layer_id == last_moe_layer:
                layout_position = "single"
            elif layer_id == first_moe_layer:
                layout_position = "first"
            elif layer_id == last_moe_layer:
                layout_position = "last"

        if should_log_layout:
            logger.info(
                "Initialized MoE expert layout",
                layer_id=layer_id,
                full_mode=full_mode,
                num_local_experts=len(self.expert_indices),
                expert_first=self.expert_indices[0],
                expert_last=self.expert_indices[-1],
                position=layout_position,
            )

        # --- Initialize experts and router ---
        self.experts = CustomDeepseekV2Experts(config, expert_indices=available_experts)
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.top_k = config.num_experts_per_tok
        self.topk_group = config.topk_group

    def route_tokens_to_experts(self, router_logits):
        batch_size, seq_len, hidden_dim = router_logits.shape
        router_logits = router_logits.view(-1, hidden_dim)
        router_logits = router_logits.softmax(dim=-1, dtype=torch.float32)

        if self.allowed_ids is not None and self.allowed_ids.numel() > 0 and self.allowed_ids.numel() < router_logits.size(-1):
            allowed_ids = self.allowed_ids.to(device=router_logits.device)

            # We create a new tensor to avoid in-place modification issues
            masked_logits = torch.full_like(router_logits, -1e4)  # Use a very large negative

            # Scatter 0.0 only to specific expert indices
            # If the allowed_ids are [5, 6, 7, 8, 9, 10], ONLY these will have non-infinite scores
            masked_logits.scatter_(
                1,
                allowed_ids.unsqueeze(0).expand(router_logits.size(0), -1),
                router_logits.gather(1, allowed_ids.unsqueeze(0).expand(router_logits.size(0), -1))
            )

            router_logits = masked_logits

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = router_logits.view(batch_size * seq_len, self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
                .reshape(batch_size * seq_len, -1)
            )
            tmp_scores = router_logits.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight
    
class CustomDeepseekV2DecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV2Attention(config=config, layer_idx=layer_idx)
        self.mlp = (
            CustomDeepseekV2Moe(config, layer_id=layer_idx)
            if layer_idx >= config.first_k_dense_replace
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class CustomDeepseekV2Model(DeepseekV2Model):
    def __init__(self, config: DeepseekV2Config):
        DeepseekV2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CustomDeepseekV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False


class CustomDeekSeekMoE(DeepseekV2ForCausalLM):
    """
    DeepseekV3 variant that interleaves MoE and dense blocks and optionally restricts experts
    to those owned by the calling group.

    If `partial=True`, MoE blocks become `myceliaSparseMoeBlock` limited to the group’s experts.
    Otherwise, standard `SparseMoeBlock` is used for MoE layers.
    """

    _keys_to_ignore_on_load_missing = [
        r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_up_proj|down_proj)$",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$",
    ]
    _virtual_expert_key_pattern = re.compile(
        r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_up_proj|down_proj)$"
    )

    def _move_missing_keys_from_meta_to_device(self, missing_keys, device_map, device_mesh, hf_quantizer) -> None:
        filtered_missing_keys = []
        ignored_virtual = 0
        for key in missing_keys:
            if self._virtual_expert_key_pattern.fullmatch(str(key)):
                ignored_virtual += 1
                continue
            filtered_missing_keys.append(key)

        if ignored_virtual > 0:
            logger.debug(
                "Skipping virtual expert missing keys during meta-to-device materialization",
                ignored_virtual=ignored_virtual,
            )

        return super()._move_missing_keys_from_meta_to_device(
            filtered_missing_keys,
            device_map,
            device_mesh,
            hf_quantizer,
        )

    def __init__(self, config):
        # IMPORTANT: avoid constructing the full DeepseekV2Model twice.
        # DeepseekV2ForCausalLM.__init__ builds DeepseekV2Model(config),
        # which causes a large transient CPU RAM spike for DeepSeek-V2-Lite.
        # We initialize the pretrained base directly, then attach only our
        # custom partial-aware model once.
        DeepseekV2PreTrainedModel.__init__(self, config)
        self.model = CustomDeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def convert_full_to_partial_model(
    partial_model: CustomDeekSeekMoE,
    full_model: DeepseekV2ForCausalLM | str | Path,
    expert_group_assignment: dict[int, dict[int, list[tuple[int, int]]]],
    target_group: int,
) -> CustomDeekSeekMoE:
    """
    Convert/load a full DeepSeek-V2-lite checkpoint into a partial model.
    Supports either:
    - `source` as an instantiated full model, or
    - `source` as a local/remote pretrained checkpoint path (possibly sharded).
    """
    partial_state = partial_model.state_dict()
    assignments = expert_group_assignment.get(target_group, {})
    loaded_counts = {"full": 0, "sliced": 0, "transposed": 0}
    assignments = expert_group_assignment.get(target_group, {})
    source_state = full_model.state_dict()

    for key, source_tensor in source_state.items():
        if key not in partial_state:
            continue

        dst_tensor = partial_state[key]

        # CASE 1: exact shape match for shared weights.
        if tuple(source_tensor.shape) == tuple(dst_tensor.shape):
            dst_tensor.copy_(source_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))
            loaded_counts["full"] += 1
            continue

        layer_idx, expert_idx = get_layer_expert_id(key)
        if layer_idx is None:
            continue

        # CASE 2: [not used] MoE gate row slicing (if partial gate shape differs).
        if ".mlp.gate.weight" in key and source_tensor.ndim == 2:
            layer_map = assignments.get(layer_idx, [])
            if not layer_map:
                continue

            for my_expert_id, org_expert_id in layer_map:
                if my_expert_id < dst_tensor.shape[0] and org_expert_id < source_tensor.shape[0]:
                    dst_tensor[my_expert_id] = source_tensor[org_expert_id].to(
                        device=dst_tensor.device, dtype=dst_tensor.dtype
                    )
                    loaded_counts["sliced"] += 1
            continue

        # CASE 3: expert tensor slicing from full tensors.
        if ".mlp.experts." in key and source_tensor.ndim >= 2:
            layer_assignments = assignments.get(layer_idx, [])

            valid_src_indices: list[int] = []
            valid_dst_indices: list[int] = []

            for dst_local_idx, org_expert_id in layer_assignments:
                org_expert_id = int(org_expert_id)
                dst_local_idx = int(dst_local_idx)

                if 0 <= org_expert_id < source_tensor.shape[0]:
                    valid_src_indices.append(org_expert_id)
                    valid_dst_indices.append(dst_local_idx)

            if not valid_src_indices:
                continue

            extracted = source_tensor[valid_src_indices]

            target_slice_shape = dst_tensor[valid_dst_indices].shape
            if tuple(extracted.shape) != tuple(target_slice_shape):
                logger.warning(
                    "Skipping expert slice due to shape mismatch",
                    key=key,
                    source_shape=tuple(extracted.shape),
                    target_shape=tuple(target_slice_shape),
                )
                continue

            dst_tensor[valid_dst_indices] = extracted.to(device=dst_tensor.device, dtype=dst_tensor.dtype)
            loaded_counts["sliced"] += 1

    partial_model.load_state_dict(partial_state, strict=False)
    logger.info("Converted full model to partial model", loaded_counts=loaded_counts, target_group=target_group)
    return partial_model


def get_moe_model_config(
    config: MinerConfig,
    topk: int,
    group_ids: list | None,
    expert_manager: ExpertManager,
    org_model_config: AutoConfig = None,
    full: bool = False,
) -> PretrainedConfig:
    # Load the hub config for its field values, then re-construct using the
    # installed DeepseekV2Config so that __init__ sets derived fields like head_dim.
    hub_cfg = AutoConfig.from_pretrained(config.model.base_arch_model, trust_remote_code=True)
    hub_dict = hub_cfg.to_dict()
    hub_dict.pop("model_type", None)
    hub_dict.pop("transformers_version", None)
    if isinstance(hub_dict.get("rope_scaling"), dict):
        rope = hub_dict["rope_scaling"]
        for field in ("factor", "beta_fast", "beta_slow"):
            if field in rope:
                rope[field] = float(rope[field])
    base_config = DeepseekV2Config(**hub_dict)

    # merge the existing model config into the base config
    if org_model_config is not None:
        for k, v in org_model_config.to_dict().items():
            setattr(base_config, k, v)

    num_routed_experts = int(hub_dict.get("n_routed_experts", 16))
    num_hidden_layers = int(getattr(base_config, "num_hidden_layers", 0))
    _validate_assignment_bounds(
        expert_group_assignment=expert_manager.expert_group_assignment,
        num_experts=num_routed_experts,
        num_hidden_layers=num_hidden_layers,
        group_ids=group_ids,
    )

    # merge our subnet config to the base config
    base_config.full = bool(full)
    base_config.num_experts = num_routed_experts
    base_config.n_group = config.moe.num_worker_groups
    base_config.topk_group = 1
    base_config.num_experts_per_tok = int(topk)
    base_config.interleave = bool(config.moe.interleave)
    base_config.decoder_sparse_step = 2 if bool(config.moe.interleave) else 1
    base_config.output_router_logits = get_nested_attr(config, "moe.aux_load_balance", False)
    base_config.router_aux_loss_coef = get_nested_attr(config, "moe.router_aux_loss_coef", False)
    base_config.norm_topk_prob = True
    base_config.max_position_embeddings = config.task.exp.data.sequence_length
    base_config.expert_group_assignment = expert_manager.expert_group_assignment
    base_config.group_ids = group_ids

    return base_config

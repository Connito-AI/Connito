from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    PretrainedConfig,
)
# transformers 4.x names the MoE block `DeepseekV2MoE` (vs `DeepseekV2Moe` on
# v5) and its `forward` routes through a `DeepseekV2MoEGate` that would bypass
# our masked routing, so `CustomDeepseekV2Moe` defines its own.
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2Config,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2PreTrainedModel,
    DeepseekV2RotaryEmbedding,
    DeepseekV2RMSNorm,
)

from connito.shared.app_logging import structlog
from connito.shared.helper import *

if TYPE_CHECKING:
    from connito.shared.config import MinerConfig
    from connito.shared.expert_manager import ExpertManager
else:
    MinerConfig = Any

logger = structlog.get_logger(__name__)


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




class CustomDeepseekV2Moe(nn.Module):
    """MoE block whose experts are `DeepseekV2MLP` modules keyed by global id.

    Keying the `ModuleDict` by global id makes parameter names identical to the
    checkpoint's, so `from_pretrained` and `load_state_dict` fill a partial model
    with no key translation — the subset is chosen by which experts are declared.
    """

    def __init__(self, config: DeepseekV2Config, layer_id: int | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.num_experts = config.n_routed_experts

        full_mode = bool(getattr(config, "full", False))

        # --- Determine allowed experts ---
        # Trainable and helper group sets are supplied independently on the
        # config (see get_moe_model_config). `allowed_expert_id` is their union
        # — used by masked-topk routing. `trainable_expert_id` / `helper_expert_id`
        # remain split so the natural-with-fallback routing rule can preserve
        # natural picks that land in the trainable set and substitute helpers
        # for the rest.
        trainable_expert_id: list[int] = []
        helper_expert_id: list[int] = []
        if full_mode:
            allowed_expert_id = list(range(config.n_routed_experts))
        elif config.expert_group_assignment is not None:
            trainable_group_ids = getattr(config, "group_ids_trainable", None)
            helper_group_ids = getattr(config, "group_ids_helper", None)
            # When neither is set explicitly, treat every assigned group as
            # trainable (matches the pre-split default of loading everything).
            if trainable_group_ids is None and helper_group_ids is None:
                trainable_group_ids = list(config.expert_group_assignment.keys())
                helper_group_ids = []
            trainable_group_ids = list(trainable_group_ids or [])
            helper_group_ids = list(helper_group_ids or [])

            def _collect(group_ids: list) -> list[int]:
                out: list[int] = []
                for group_id in group_ids:
                    layer_assignments = config.expert_group_assignment[int(group_id)].get(layer_id, [])
                    out += [int(org_expert_id) for _, org_expert_id in layer_assignments]
                return out

            trainable_expert_id = _collect(trainable_group_ids)
            helper_expert_id = _collect(helper_group_ids)
            allowed_expert_id = trainable_expert_id + helper_expert_id
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
                f"layer_id={layer_id}, "
                f"group_ids_trainable={getattr(config, 'group_ids_trainable', None)}, "
                f"group_ids_helper={getattr(config, 'group_ids_helper', None)}, "
                f"invalid={invalid_experts[:10]}"
            )
        if len(available_experts) == 0:
            raise ValueError(
                f"No routed experts assigned for layer_id={layer_id}. "
                f"group_ids_trainable={getattr(config, 'group_ids_trainable', None)}, "
                f"group_ids_helper={getattr(config, 'group_ids_helper', None)}"
            )
        if full_mode and len(available_experts) != config.n_routed_experts:
            raise ValueError(
                "Full model mode must include all routed experts. "
                f"layer_id={layer_id}, expected={config.n_routed_experts}, got={len(available_experts)}"
            )

        self.expert_indices = available_experts
        self.register_buffer("allowed_ids", torch.tensor(self.expert_indices, dtype=torch.long), persistent=False)

        # Separate buffers for the natural-with-fallback routing mode. When the
        # mode is not enabled these are unused but still present (harmless, small).
        _trainable = sorted({int(e) for e in trainable_expert_id})
        _helper = sorted({int(e) for e in helper_expert_id if int(e) not in _trainable})
        self.register_buffer(
            "trainable_ids",
            torch.tensor(_trainable, dtype=torch.long) if _trainable else torch.zeros(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "helper_ids",
            torch.tensor(_helper, dtype=torch.long) if _helper else torch.zeros(0, dtype=torch.long),
            persistent=False,
        )
        self.routing_mode = str(getattr(config, "routing_mode", "masked_topk"))

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
        self.experts = nn.ModuleDict(
            {
                str(expert_id): DeepseekV2MLP(
                    config=config, intermediate_size=config.moe_intermediate_size
                )
                for expert_id in available_experts
            }
        )
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

        # ── natural-with-fallback (2Fnat) ──
        # The base gate ranks all n_routed_experts natively (no masking). Slots
        # in the natural top-k that land on a trainable expert are kept as-is
        # (the trainable expert receives that token with its natural gate
        # weight). Slots that land on a non-trainable expert get REPLACED by the
        # token's top-scoring helper expert (ranked by the same gate over the
        # helper subset). Net effect: trainable experts only ever see tokens the
        # base gate naturally routes to them (exact train-eval alignment), and
        # non-trainable demand is absorbed by frozen helpers.
        if (
            self.routing_mode == "natural_with_fallback"
            and self.trainable_ids.numel() > 0
            and self.helper_ids.numel() > 0
            and self.topk_method == "greedy"
        ):
            n_tokens = router_logits.size(0)
            device = router_logits.device
            trainable_ids = self.trainable_ids.to(device=device)
            helper_ids = self.helper_ids.to(device=device)

            # Natural top-k over ALL experts, no masking
            natural_w, natural_idx = torch.topk(router_logits, k=self.top_k, dim=-1, sorted=False)

            # Which of each token's natural picks landed in the trainable set?
            is_trainable = torch.isin(natural_idx, trainable_ids)  # [n_tokens, top_k] bool

            # Helper-only masked scores → top-k helpers per token
            helper_masked = torch.full_like(router_logits, -1e4)
            helper_masked.scatter_(
                1,
                helper_ids.unsqueeze(0).expand(n_tokens, -1),
                router_logits.gather(1, helper_ids.unsqueeze(0).expand(n_tokens, -1)),
            )
            fb_w, fb_idx = torch.topk(helper_masked, k=self.top_k, dim=-1, sorted=False)

            # For each of the k slots, pick natural if trainable else next-fallback
            # cumsum over ~is_trainable gives 1-indexed position within fallback list
            fb_pos = ((~is_trainable).long().cumsum(-1) - 1).clamp(min=0)
            final_idx = torch.where(is_trainable, natural_idx, fb_idx.gather(1, fb_pos))
            final_w = torch.where(is_trainable, natural_w, fb_w.gather(1, fb_pos))

            return final_idx, final_w * self.routed_scaling_factor
        # ── end 2Fnat branch ──

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

    def moe_infer(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch each token to the experts it was routed to.

        Moved here from `CustomDeepseekV2Experts.forward`. The only change is
        that one expert's compute is `self.experts[str(gid)](x)` — a plain
        `DeepseekV2MLP` — instead of two `F.linear` calls against slices of a
        stacked parameter. Experts not materialised on this module are skipped,
        as the `global_to_local_map == -1` check used to do.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for global_expert_idx in expert_hit:

            global_expert_idx = global_expert_idx[0].item()

            expert_key = str(global_expert_idx)

            if expert_key not in self.experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[global_expert_idx])

            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_key](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Explicit forward replicating transformers v5 `DeepseekV2Moe.forward`.
        # On v5 this was inherited; on 4.x the base `DeepseekV2MoE.forward`
        # routes via a `DeepseekV2MoEGate` + ModuleList loop and would bypass our
        # masked routing, so we define it ourselves to keep numerics identical
        # across transformers versions.
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = nn.functional.linear(
            hidden_states.type(torch.float32), self.gate.weight.type(torch.float32)
        )
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe_infer(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        if getattr(self, "shared_experts", None) is not None:
            hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


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
    """DeepSeek-V2-Lite with `CustomDeepseekV2Moe` on every MoE layer.

    Parameter names match the stock checkpoint, so nothing here translates keys:
    `from_pretrained` fills the declared experts and a miner shard overlays with
    `strict=False`.
    """

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


def get_moe_model_config(
    config: MinerConfig,
    topk: int,
    group_ids_trainable: list | None,
    expert_manager: ExpertManager,
    org_model_config: AutoConfig = None,
    full: bool = False,
    routing_mode: str = "masked_topk",
    group_ids_helper: list | None = None,
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
    # Validate every group we plan to load (union of trainable + helper). If
    # both are None the validator falls back to all-groups.
    _merged_group_ids: list | None
    if group_ids_trainable is None and group_ids_helper is None:
        _merged_group_ids = None
    else:
        _merged_group_ids = list(group_ids_trainable or []) + list(group_ids_helper or [])
    _validate_assignment_bounds(
        expert_group_assignment=expert_manager.expert_group_assignment,
        num_experts=num_routed_experts,
        num_hidden_layers=num_hidden_layers,
        group_ids=_merged_group_ids,
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
    base_config.group_ids_trainable = list(group_ids_trainable) if group_ids_trainable is not None else None
    base_config.group_ids_helper = list(group_ids_helper) if group_ids_helper is not None else None
    base_config.routing_mode = str(routing_mode)

    return base_config

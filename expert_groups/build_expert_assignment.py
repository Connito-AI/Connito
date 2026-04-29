"""Build an inference-derived expert assignment for DeepSeek-V2-Lite.

This script runs the full model in eval/inference mode on a deterministic slice
of the task dataset, records which routed experts are actually selected by the
router in each MoE layer, and emits an assignment JSON with exactly N experts
per routed layer.

By default it writes a generated file next to the current task assignment so it
can be inspected before replacing the live assignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from connito.shared.config import MinerConfig
from connito.shared.dataloader import get_dataloader
from connito.shared.expert_manager import ExpertManager
from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Moe
from connito.shared.modeling.mycelia import get_base_model, get_base_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "checkpoints"
            / "miner"
            / "miner-george"
            / "main-hk2"
            / "finney"
            / "config.yaml"
        ),
        help="Path to miner config.yaml",
    )
    parser.add_argument("--task", default="exp_math", help="Expert group task name")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "expert_groups" / "exp_math" / "expert_assignment.generated.json"),
        help="Output path for generated assignment JSON",
    )
    parser.add_argument("--num-batches", type=int, default=20, help="Number of inference batches to process")
    parser.add_argument("--experts-per-layer", type=int, default=8, help="Experts to keep per routed layer")
    parser.add_argument(
        "--selection-metric",
        choices=["weight", "count"],
        default="weight",
        help="Rank experts by summed router weight or raw selection count",
    )
    parser.add_argument("--rank", type=int, default=0, help="Dataset shard rank")
    parser.add_argument("--world-size", type=int, default=1, help="Dataset shard world size")
    parser.add_argument(
        "--validation-split",
        action="store_true",
        help="Use the validation split instead of the training split",
    )
    return parser.parse_args()


def load_config(config_path: str, task_name: str) -> MinerConfig:
    config = MinerConfig.from_path(config_path, auto_update_config=True)
    config._update_by_task(expert_group_name=task_name)
    config.ckpt.resume_from_ckpt = False
    config.model.torch_compile = False
    return config


def get_target_device_and_dtype(config: MinerConfig) -> tuple[torch.device, torch.dtype]:
    precision = getattr(config.model, "precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    device_name = getattr(config.model, "device", "cuda")
    if not torch.cuda.is_available():
        device_name = "cpu"

    return torch.device(device_name), dtype


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def attach_router_trackers(model: torch.nn.Module):
    counts_per_layer: dict[int, Counter[int]] = defaultdict(Counter)
    weights_per_layer: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    routed_layers: list[int] = []
    current_attention_mask: torch.Tensor | None = None

    def set_attention_mask(attention_mask: torch.Tensor | None) -> None:
        nonlocal current_attention_mask
        current_attention_mask = attention_mask

    def make_wrapper(layer_id: int, module: CustomDeepseekV2Moe, original_route_fn):
        routed_layers.append(layer_id)

        def wrapped(router_logits: torch.Tensor):
            topk_idx, topk_weight = original_route_fn(router_logits)

            flat_idx = topk_idx
            flat_weight = topk_weight

            if current_attention_mask is not None:
                flat_mask = current_attention_mask.reshape(-1).bool()
                if flat_mask.numel() == flat_idx.shape[0]:
                    flat_idx = flat_idx[flat_mask]
                    flat_weight = flat_weight[flat_mask]

            idx_cpu = flat_idx.detach().cpu()
            weight_cpu = flat_weight.detach().cpu().to(torch.float32)

            if idx_cpu.numel() > 0:
                counts_per_layer[layer_id].update(idx_cpu.reshape(-1).tolist())
                for expert_id, weight in zip(idx_cpu.reshape(-1).tolist(), weight_cpu.reshape(-1).tolist()):
                    weights_per_layer[layer_id][int(expert_id)] += float(weight)

            return topk_idx, topk_weight

        module.route_tokens_to_experts = wrapped

    for layer_id, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, CustomDeepseekV2Moe):
            make_wrapper(layer_id, mlp, mlp.route_tokens_to_experts)

    return counts_per_layer, weights_per_layer, sorted(routed_layers), set_attention_mask


def build_assignment(
    routed_layers: list[int],
    counts_per_layer: dict[int, Counter[int]],
    weights_per_layer: dict[int, dict[int, float]],
    experts_per_layer: int,
    selection_metric: str,
) -> dict[str, list[list[int]]]:
    assignment: dict[str, list[list[int]]] = {}

    for layer_id in routed_layers:
        if selection_metric == "weight":
            layer_scores = weights_per_layer.get(layer_id, {})
            ranked = sorted(layer_scores.items(), key=lambda item: (-item[1], item[0]))
        else:
            ranked = sorted(counts_per_layer.get(layer_id, {}).items(), key=lambda item: (-item[1], item[0]))

        if len(ranked) < experts_per_layer:
            raise ValueError(
                f"Layer {layer_id} only observed {len(ranked)} experts; need {experts_per_layer}. "
                "Increase --num-batches or switch metric."
            )

        selected = ranked[:experts_per_layer]
        assignment[str(layer_id)] = [[local_idx, int(expert_id)] for local_idx, (expert_id, _) in enumerate(selected)]

    return assignment


def main() -> int:
    args = parse_args()

    config = load_config(args.config, args.task)
    tokenizer = get_base_tokenizer(config)
    expert_manager = ExpertManager(config)

    print(f"[1/4] Building full inference model for task={args.task} ...")
    model = get_base_model(
        config=config,
        expert_manager=expert_manager,
        group_ids=None,
        partial=False,
    )
    target_device, target_dtype = get_target_device_and_dtype(config)
    model = model.to(device=target_device, dtype=target_dtype)
    model.eval()
    device = get_model_device(model)

    print(f"[2/4] Attaching router trackers on {device} ...")
    counts_per_layer, weights_per_layer, routed_layers, set_attention_mask = attach_router_trackers(model)
    if not routed_layers:
        raise RuntimeError("No routed MoE layers found in model")

    use_train_split = not args.validation_split
    print(
        f"[3/4] Running inference over {args.num_batches} batches "
        f"(split={'train' if use_train_split else 'validation'}, rank={args.rank}, world_size={args.world_size}) ..."
    )
    dataloader = get_dataloader(
        config=config,
        tokenizer=tokenizer,
        seed=None,
        rank=args.rank,
        world_size=args.world_size,
        train=use_train_split,
    )

    batches_processed = 0
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            set_attention_mask(attention_mask)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            batches_processed += 1

            if batches_processed % 5 == 0 or batches_processed == args.num_batches:
                print(f"      processed {batches_processed}/{args.num_batches} batches")

            if batches_processed >= args.num_batches:
                break

    print(f"[4/4] Building {args.experts_per_layer}-experts-per-layer assignment ...")
    assignment = build_assignment(
        routed_layers=routed_layers,
        counts_per_layer=counts_per_layer,
        weights_per_layer=weights_per_layer,
        experts_per_layer=args.experts_per_layer,
        selection_metric=args.selection_metric,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(assignment, indent=2) + "\n", encoding="utf-8")

    print(f"Saved assignment for {len(assignment)} routed layers to {output_path}")
    for layer_id in routed_layers[:5]:
        layer_assignment = assignment[str(layer_id)]
        print(f"  layer {layer_id}: {[expert_id for _, expert_id in layer_assignment]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
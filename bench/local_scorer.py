"""Local validator-replica scorer.

Reverse-engineered from `connito/validator/evaluator.py`:

    delta       = max(0.0, baseline_loss - miner_val_loss)
    round_score = delta ** 1.2

The validator evaluates miners by:
  1. Building a fresh streaming dataloader seeded by `combined_seed`
     (`get_dataloader`) — 50% C4 + 50% Nemotron-CC-Math, with a 50_000-row
     buffer-shuffle and a 50_000-row random skip per source so the eval
     pool can't be memorized.
  2. Running `evaluate_model` for `EVAL_MAX_BATCHES = 50` batches:
     `val_loss = (loss_sum - aux_loss_sum) / scored_batches`; NaN/Inf
     batches drop out of the divisor.
  3. Loading the miner checkpoint into a `copy.deepcopy(base_model)` and
     running the same eval on the same dataloader.

This module reproduces that pipeline locally so a miner can score a
candidate checkpoint against a baseline without running a real
validator. The whole point is to let downstream optimization units
measure their effect on `score` before committing to chain.

CLI:

    python -m bench.local_scorer \\
        --checkpoint path/to/miner.safetensors \\
        --baseline-checkpoint path/to/baseline.safetensors \\
        --seed deadbeef --max-batches 50 --out report.json

Multi-seed mode samples `--multi-seed N` random seeds (or evaluates
`--validator-seeds s1,s2,...`) and reports mean/std/min/max score —
useful for measuring score variance across the eval pool the validator
might draw.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import secrets
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model
from connito.shared.helper import load_state_dict_from_path

logger = structlog.get_logger(__name__)


# Mirror `connito/validator/evaluator.py:580`. Keep in sync if the validator
# ever changes its eval cap.
EVAL_MAX_BATCHES = 50


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_device(device_arg: str | None) -> torch.device:
    """Resolve `--device` flag → torch.device, defaulting to cuda:0 when
    available, else cpu. Aligns with the validator's GPU-first preference."""
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _normalize_seed(seed: int | str) -> str:
    """Validator seeds are hex strings (sha256 hexdigest). Accept either
    a raw int or a hex string and normalize to lowercase hex.

    The dataloader does `int(str(seed)[:8], 16)` to derive an int seed, so
    the input must be valid hex. An integer input is hex-encoded.
    """
    if isinstance(seed, int):
        # `secrets.token_hex` style — same length as sha256 hexdigest so
        # multi-seed-summary output stays compact in JSON.
        return f"{seed:064x}"
    s = str(seed).lower().strip()
    if s.startswith("0x"):
        s = s[2:]
    # Validate hex-only chars; otherwise the dataloader's
    # `int(seed[:8], 16)` cast would raise.
    int(s[:8], 16)
    return s


def _random_seed() -> str:
    """Generate a uniformly-random 256-bit hex seed (sha256-equivalent)."""
    return secrets.token_hex(32)


# -----------------------------------------------------------------------------
# Scoring core
# -----------------------------------------------------------------------------
@dataclass
class SeedResult:
    seed: str
    baseline_loss: float
    miner_val_loss: float
    delta: float
    score: float
    scored_batches: int
    nan_batches: int
    per_batch_loss: list[float]
    per_source_loss: dict[str, float] | None


def _make_per_batch_eval(
    *,
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    max_eval_batches: int,
) -> tuple[float, int, int, list[float]]:
    """Run a single eval pass and return (val_loss, scored, nan, per_batch).

    Mirrors `evaluate_model` (see `connito/shared/evaluate.py:28-154`) but
    captures the per-batch loss for the report. Aux_loss is subtracted out
    of `loss_sum` so the returned `val_loss` matches the validator's
    `(loss_sum - aux_loss_sum) / scored_batches` exactly.

    Re-implementing the loop (rather than reusing `evaluate_model`) is the
    only way to get per-batch losses out without modifying the production
    code path — `evaluate_model` only returns the aggregated mean. The
    arithmetic is intentionally identical line-for-line.
    """
    model.to(device)
    model.eval()

    loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    scored_batches: int = 0
    nan_batches: int = 0
    per_batch_loss: list[float] = []

    with torch.no_grad():
        for batch_step, batch in enumerate(dataloader):
            device_batch: dict[str, torch.Tensor] = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(model.device)
            if device_batch.get("attention_mask") is None and "input_ids" in device_batch:
                device_batch["attention_mask"] = torch.ones_like(device_batch["input_ids"])

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            eval_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.amp.autocast(autocast_device, dtype=eval_dtype):
                outputs = model(**device_batch)
                loss_val = outputs.loss
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    nan_batches += 1
                else:
                    batch_loss = float(loss_val.item())
                    batch_aux = (
                        float(outputs.aux_loss.item())
                        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None
                        else 0.0
                    )
                    loss_sum += batch_loss
                    aux_loss_sum += batch_aux
                    per_batch_loss.append(batch_loss - batch_aux)
                    scored_batches += 1

            del device_batch, outputs
            gc.collect()

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

    if scored_batches == 0:
        val_loss = float("inf")
    else:
        val_loss = (loss_sum - aux_loss_sum) / scored_batches

    return val_loss, scored_batches, nan_batches, per_batch_loss


def _per_source_losses(
    *,
    config: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    seed: str,
    device: torch.device,
    max_eval_batches: int,
) -> dict[str, float]:
    """Evaluate the model independently on each configured dataset source.

    `interleave_datasets` removes any link between a yielded batch and the
    source it came from, so the only way to attribute loss per-source is
    to build a single-source dataloader for each source and run the eval
    independently. The full mixed-source loss is still the canonical
    `val_loss` — this is purely diagnostic.

    Returns a mapping of source-name → mean cross-entropy loss, where
    source-name is the `path` field from the config (e.g. "allenai/c4",
    "nvidia/Nemotron-CC-Math-v1"). Sources without a name use `f"src_{i}"`
    as the key.
    """
    sources = getattr(config.task.exp.data, "dataset_sources", None)
    if not sources:
        return {}

    original = list(sources)
    result: dict[str, float] = {}
    for i, src in enumerate(original):
        key = getattr(src, "path", None) or f"src_{i}"
        # Temporarily restrict dataset_sources to a single entry. Pydantic
        # validates the list is non-empty, which a single-element list
        # satisfies. We restore the full list at the end.
        try:
            config.task.exp.data.dataset_sources = [src]
            dataloader = get_dataloader(
                config=config,
                tokenizer=tokenizer,
                seed=seed,
                rank=0,
                world_size=config.dataloader.world_size,
            )
            metrics = evaluate_model(
                0, model, dataloader, device, max_eval_batches, rank=None,
            )
            result[key] = float(metrics.get("val_loss", float("inf")))
            del dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            config.task.exp.data.dataset_sources = original

    return result


def score_checkpoint_for_seed(
    *,
    config: Any,
    tokenizer: Any,
    base_model: torch.nn.Module,
    miner_state_dict: dict[str, torch.Tensor],
    baseline_state_dict: dict[str, torch.Tensor],
    seed: str,
    device: torch.device,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    compute_per_source: bool = False,
) -> SeedResult:
    """Score a single (miner, baseline) checkpoint pair against a single seed.

    Replicates the validator's per-miner flow (`evaluate_one_miner_sync` →
    `_evaluate_on_fresh_loader_sync` → `evaluate_model`):

    1. Build a streaming dataloader seeded by `seed` (mixes C4 + Nemotron
       with the 50_000-row shuffle + 50_000-row skip).
    2. Load the baseline state_dict into a deep-copy of `base_model` and
       evaluate. (Validator evaluates the global / baseline model.)
    3. Load the miner state_dict into another deep-copy and evaluate
       against the SAME dataloader seed. (Validator's `combined_seed` is
       reused across baseline + every miner so deltas are comparable.)

    `compute_per_source` runs an additional pass per data source (no
    interleave) — purely diagnostic, the per-source loss is NOT part of
    the validator's score formula.
    """
    logger.info(
        "Scoring seed",
        seed=seed,
        max_eval_batches=max_eval_batches,
        device=str(device),
    )

    # --- Baseline eval ---
    baseline_model = copy.deepcopy(base_model)
    incompatible = baseline_model.load_state_dict(baseline_state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.debug(
            "baseline state_dict mismatch (strict=False)",
            missing=len(incompatible.missing_keys),
            unexpected=len(incompatible.unexpected_keys),
        )
    baseline_model = baseline_model.to(device)

    baseline_loader = get_dataloader(
        config=config,
        tokenizer=tokenizer,
        seed=seed,
        rank=0,
        world_size=config.dataloader.world_size,
    )
    baseline_metrics = evaluate_model(
        0, baseline_model, baseline_loader, device, max_eval_batches, rank=None,
    )
    baseline_loss = float(baseline_metrics.get("val_loss", float("inf")))
    del baseline_loader, baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Miner eval ---
    miner_model = copy.deepcopy(base_model)
    incompatible = miner_model.load_state_dict(miner_state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.debug(
            "miner state_dict mismatch (strict=False)",
            missing=len(incompatible.missing_keys),
            unexpected=len(incompatible.unexpected_keys),
        )
    miner_model = miner_model.to(device)

    miner_loader = get_dataloader(
        config=config,
        tokenizer=tokenizer,
        seed=seed,
        rank=0,
        world_size=config.dataloader.world_size,
    )
    val_loss, scored_batches, nan_batches, per_batch_loss = _make_per_batch_eval(
        model=miner_model,
        dataloader=miner_loader,
        device=device,
        max_eval_batches=max_eval_batches,
    )

    per_source_loss: dict[str, float] | None = None
    if compute_per_source:
        try:
            per_source_loss = _per_source_losses(
                config=config,
                tokenizer=tokenizer,
                model=miner_model,
                seed=seed,
                device=device,
                max_eval_batches=max_eval_batches,
            )
        except Exception as e:
            logger.warning("per-source eval failed", error=str(e))
            per_source_loss = None

    del miner_loader, miner_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Mirror the validator's clamp + power exponent. `max(0, ...)`
    # collapses negative deltas (miner worse than baseline) to 0 score.
    delta = max(0.0, baseline_loss - val_loss)
    score = delta ** 1.2

    return SeedResult(
        seed=seed,
        baseline_loss=baseline_loss,
        miner_val_loss=val_loss,
        delta=delta,
        score=score,
        scored_batches=scored_batches,
        nan_batches=nan_batches,
        per_batch_loss=per_batch_loss,
        per_source_loss=per_source_loss,
    )


# -----------------------------------------------------------------------------
# Multi-seed driver
# -----------------------------------------------------------------------------
def score_checkpoint(
    *,
    config: Any,
    tokenizer: Any,
    base_model: torch.nn.Module,
    miner_state_dict: dict[str, torch.Tensor],
    baseline_state_dict: dict[str, torch.Tensor],
    seeds: list[str],
    device: torch.device,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    compute_per_source: bool = False,
) -> dict[str, Any]:
    """Score `seeds` (one or many) and return a JSON-serializable report.

    For a single seed the report has the top-level fields (`baseline_loss`,
    `miner_val_loss`, `delta`, `score`, ...) populated from that one seed.

    For >1 seed the top-level fields are populated from the FIRST seed
    (for callers who want a single representative number), and
    `multi_seed_summary` aggregates score statistics across all seeds.
    `seeds_used` always lists every seed evaluated.
    """
    if not seeds:
        raise ValueError("score_checkpoint: at least one seed required")

    results = [
        score_checkpoint_for_seed(
            config=config,
            tokenizer=tokenizer,
            base_model=base_model,
            miner_state_dict=miner_state_dict,
            baseline_state_dict=baseline_state_dict,
            seed=s,
            device=device,
            max_eval_batches=max_eval_batches,
            compute_per_source=compute_per_source,
        )
        for s in seeds
    ]

    primary = results[0]
    report: dict[str, Any] = {
        "baseline_loss": primary.baseline_loss,
        "miner_val_loss": primary.miner_val_loss,
        "delta": primary.delta,
        "score": primary.score,
        "scored_batches": primary.scored_batches,
        "nan_batches": primary.nan_batches,
        "per_batch_loss": primary.per_batch_loss,
        "per_source_loss": primary.per_source_loss,
        "seeds_used": [r.seed for r in results],
        "multi_seed_summary": None,
    }

    if len(results) > 1:
        scores = [r.score for r in results]
        report["multi_seed_summary"] = {
            "mean_score": statistics.fmean(scores),
            # `pstdev` so a single seed run wouldn't crash on stdev; with
            # >1 seed it's the unbiased-style population stdev which is
            # what we want for "how variable is this miner's score across
            # seeds the validator might pick" — not a sample estimate.
            "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores),
            "max_score": max(scores),
            "per_seed": [
                {
                    "seed": r.seed,
                    "val_loss": r.miner_val_loss,
                    "delta": r.delta,
                    "score": r.score,
                }
                for r in results
            ],
        }
    return report


# -----------------------------------------------------------------------------
# Checkpoint and model loading
# -----------------------------------------------------------------------------
def _load_base_model_from_config(config: Any) -> tuple[torch.nn.Module, Any]:
    """Build the base (untrained) model + tokenizer from a `WorkerConfig`.

    Reuses `connito.shared.modeling.mycelia.get_base_model` so the
    architecture matches exactly what the validator instantiates. The
    expert_manager is constructed from the config's `task.path`
    expert-assignment file.

    Tokenizer is HF AutoTokenizer for `config.model.model_path` — the
    validator uses the same one to feed the dataloader.
    """
    from transformers import AutoTokenizer

    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.mycelia import get_base_model

    expert_manager = ExpertManager.from_path(config.task.path)
    base_model = get_base_model(
        config=config,
        expert_manager=expert_manager,
        group_ids=[config.task.exp.group_id],
        partial=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return base_model, tokenizer


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench.local_scorer",
        description=(
            "Validator-replica scorer. Computes "
            "`score = max(0, baseline_loss - miner_val_loss) ** 1.2` "
            "for a miner checkpoint against a baseline, using the same "
            "dataloader + evaluator the on-chain validator uses."
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to miner checkpoint (.safetensors or .pt) to score.",
    )
    parser.add_argument(
        "--baseline-checkpoint", required=True, type=Path,
        help="Path to baseline (validator global model) checkpoint to score against.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Optional YAML config (validator/miner). If omitted, a default "
            "`MinerConfig()` is built with the `exp_math` expert group."
        ),
    )
    parser.add_argument(
        "--expert-group", default=None,
        help="Override `task.expert_group_name` (e.g. 'exp_math', 'exp_dummy').",
    )
    parser.add_argument(
        "--seed", default=None,
        help="Eval seed (hex string, e.g. sha256 of a validator combined-seed). "
             "If omitted and --multi-seed/--validator-seeds also omitted, a random seed is used.",
    )
    parser.add_argument(
        "--validator-seeds", default=None,
        help="Comma-separated list of seeds to evaluate across.",
    )
    parser.add_argument(
        "--multi-seed", type=int, default=0,
        help="Sample N random seeds and report mean/std/min/max score.",
    )
    parser.add_argument(
        "--max-batches", type=int, default=EVAL_MAX_BATCHES,
        help=f"Max batches per eval (default: {EVAL_MAX_BATCHES}, matches validator).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (e.g. 'cuda:0', 'cpu'). Default: cuda:0 if available, else cpu.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write JSON report. Always also printed to stdout.",
    )
    parser.add_argument(
        "--per-source", action="store_true",
        help="Compute per-source loss (C4 vs Nemotron). Doubles eval cost.",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> Any:
    """Build a `MinerConfig` from CLI args.

    If `--config` is provided, load from disk. Otherwise instantiate the
    default `MinerConfig()` — which constructs paths from CWD and reads
    `expert_groups/<expert_group_name>/config.yaml` for the dataset
    sources. The validator path uses `ValidatorConfig`; we use
    `MinerConfig` because it's cheaper (no `evaluation`/`dht` sections)
    and the dataloader code paths read the same fields off either.
    """
    from connito.shared.config import MinerConfig

    if args.config is not None:
        cfg = MinerConfig.from_path(args.config, auto_update_config=False)
    else:
        cfg = MinerConfig()
    if args.expert_group is not None:
        cfg._update_by_task(expert_group_name=args.expert_group)  # type: ignore[attr-defined]
    return cfg


def _resolve_seeds(args: argparse.Namespace) -> list[str]:
    """Resolve seeds from CLI args.

    Precedence (highest first):
      1. `--validator-seeds a,b,c` (comma-separated explicit list)
      2. `--multi-seed N` (N random seeds; if --seed is also passed,
         that's the first one and the remaining N-1 are random)
      3. `--seed S` (single explicit seed)
      4. (no flags) — one random seed
    """
    if args.validator_seeds:
        return [_normalize_seed(s) for s in args.validator_seeds.split(",") if s.strip()]
    if args.multi_seed and args.multi_seed > 0:
        first = [_normalize_seed(args.seed)] if args.seed else []
        n_random = max(0, args.multi_seed - len(first))
        return first + [_random_seed() for _ in range(n_random)]
    if args.seed:
        return [_normalize_seed(args.seed)]
    return [_random_seed()]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = _resolve_device(args.device)
    seeds = _resolve_seeds(args)

    logger.info(
        "local_scorer starting",
        checkpoint=str(args.checkpoint),
        baseline=str(args.baseline_checkpoint),
        device=str(device),
        num_seeds=len(seeds),
        max_batches=args.max_batches,
        per_source=args.per_source,
    )

    if not args.checkpoint.exists():
        logger.error("miner checkpoint not found", path=str(args.checkpoint))
        return 2
    if not args.baseline_checkpoint.exists():
        logger.error("baseline checkpoint not found", path=str(args.baseline_checkpoint))
        return 2

    config = _build_config(args)
    base_model, tokenizer = _load_base_model_from_config(config)

    miner_sd = load_state_dict_from_path(args.checkpoint)
    baseline_sd = load_state_dict_from_path(args.baseline_checkpoint)

    report = score_checkpoint(
        config=config,
        tokenizer=tokenizer,
        base_model=base_model,
        miner_state_dict=miner_sd,
        baseline_state_dict=baseline_sd,
        seeds=seeds,
        device=device,
        max_eval_batches=args.max_batches,
        compute_per_source=args.per_source,
    )

    # Stable JSON for diff-friendly comparison across runs. Floats stay
    # as-is — no rounding; downstream consumers may rely on full precision.
    serialized = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(serialized)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n")
        logger.info("wrote report", path=str(args.out))

    return 0


if __name__ == "__main__":
    sys.exit(main())

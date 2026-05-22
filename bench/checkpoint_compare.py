"""Multi-checkpoint comparator with multi-seed scoring.

Given N local checkpoints, score each against the same baseline checkpoint
using the same set of evaluation seeds and produce a ranked leaderboard
matching the validator's scoring formula:

    score = max(0, baseline_loss - miner_val_loss) ** 1.2

Per-validator val_loss variance is large in practice (a checkpoint that
scores 0.37 weight from one validator can score 0.0066 from another for
the same round). Multi-seed robustness is therefore a much better signal
than a single-seed peak — this comparator averages per-seed scores so a
checkpoint that wins only on a lucky seed does not outrank one that is
broadly good.

Two execution paths:

  * Production: real DeepSeek-V2-Lite model + the validator's eval
    dataloader (``connito.shared.dataloader.get_dataloader``) +
    ``connito.shared.evaluate.evaluate_model``. Wired up only when
    ``main()`` is invoked from the CLI.

  * Tests / smoke: caller injects an ``eval_fn`` callable
    ``(checkpoint_path, seed) -> val_loss``. This keeps the ranking,
    aggregation, and CLI plumbing decoupled from heavy HF / GPU
    dependencies so the test suite can exercise the comparator with
    synthetic state-dicts in seconds.

Critical input to Unit 15 (best-checkpoint selection) which picks the
right snapshot to submit at MinerCommit2 phase.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# Suffixes accepted when auto-discovering checkpoints in a directory.
# Mirrors ``connito.shared.helper.MINER_CHECKPOINT_SUFFIXES`` — ``.safetensors``
# is the preferred path (no pickle, no code-execution surface),
# ``.pt`` / ``.bin`` are still accepted for backwards compatibility with
# checkpoints saved by older miner / HuggingFace tooling.
CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".pt", ".bin")

# Validator scoring constants — keep in sync with
# ``connito/validator/evaluator.py``. The exponent comes from the same
# call site (`score = delta ** 1.2`).
SCORE_EXPONENT: float = 1.2

# Default eval seed count when none is provided. Five is the smallest
# count that keeps the per-checkpoint std meaningful while staying cheap
# enough to run on a single GPU in a reasonable wall time.
DEFAULT_NUM_SEEDS: int = 5
# Default tokenizer / base-model identifier — matches the production
# miner's choice (see ``docs/`` for the migration history).
DEFAULT_TOKENIZER: str = "deepseek-ai/DeepSeek-V2-Lite"
# Default eval batch cap per (checkpoint, seed). Mirrors
# ``connito/validator/evaluator.py:EVAL_MAX_BATCHES`` so a local score
# matches a validator round eval reasonably closely.
DEFAULT_MAX_BATCHES: int = 50


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    """Per-seed eval result for a single checkpoint."""

    seed: int
    val_loss: float
    delta: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "seed": int(self.seed),
            "val_loss": float(self.val_loss),
            "delta": float(self.delta),
            "score": float(self.score),
        }


@dataclass
class CheckpointResult:
    """Aggregate eval result for a single checkpoint across all seeds."""

    path: str
    name: str
    per_seed: list[SeedResult] = field(default_factory=list)

    # Aggregate fields are computed once all seeds have been scored, via
    # `finalize()`. They are exposed as plain attributes (not @property)
    # so callers can mutate them in tests / inject precomputed values.
    mean_val_loss: float = 0.0
    std_val_loss: float = 0.0
    min_val_loss: float = 0.0
    max_val_loss: float = 0.0
    mean_delta: float = 0.0
    mean_score: float = 0.0

    def finalize(self) -> None:
        """Populate aggregate fields from `per_seed`.

        Mean / std use sample statistics when n>=2 so a single-seed eval
        does not raise on ``statistics.stdev``. Inf / NaN val_loss values
        flow through to the aggregates unchanged (the validator scoring
        formula already clamps delta to 0 in that case via ``max(0,
        ...)``), which keeps a clearly broken checkpoint from spuriously
        outranking a working one.
        """
        if not self.per_seed:
            self.mean_val_loss = float("inf")
            self.std_val_loss = 0.0
            self.min_val_loss = float("inf")
            self.max_val_loss = float("inf")
            self.mean_delta = 0.0
            self.mean_score = 0.0
            return

        losses = [r.val_loss for r in self.per_seed]
        deltas = [r.delta for r in self.per_seed]
        scores = [r.score for r in self.per_seed]

        self.mean_val_loss = float(statistics.fmean(losses))
        self.std_val_loss = float(statistics.stdev(losses)) if len(losses) >= 2 else 0.0
        self.min_val_loss = float(min(losses))
        self.max_val_loss = float(max(losses))
        self.mean_delta = float(statistics.fmean(deltas))
        self.mean_score = float(statistics.fmean(scores))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": str(self.name),
            "mean_val_loss": float(self.mean_val_loss),
            "std_val_loss": float(self.std_val_loss),
            "min_val_loss": float(self.min_val_loss),
            "max_val_loss": float(self.max_val_loss),
            "mean_delta": float(self.mean_delta),
            "mean_score": float(self.mean_score),
            "per_seed": [r.to_dict() for r in self.per_seed],
        }


# ---------------------------------------------------------------------------
# Pure scoring / ranking helpers (no HF / GPU dependency)
# ---------------------------------------------------------------------------


def score_from_loss(baseline_loss: float, val_loss: float) -> tuple[float, float]:
    """Return ``(delta, score)`` for a single (baseline, val_loss) pair.

    Mirrors the validator's exact formula:

        delta = max(0, baseline_loss - val_loss)
        score = delta ** SCORE_EXPONENT

    Non-finite ``val_loss`` (e.g. a checkpoint that always NaNs in eval)
    naturally collapses ``delta`` to 0 because ``baseline - inf == -inf``
    and ``max(0, -inf) == 0``. Returning a 0 score here matches what
    the validator records on-chain for the same input — no special
    casing needed.
    """
    # `baseline_loss - val_loss` can be inf/-inf when val_loss is inf,
    # so wrap the subtraction in `max(0.0, ...)` explicitly so a NaN
    # val_loss (rare but possible from a malformed eval path) does not
    # propagate as NaN into the leaderboard. `max(0.0, float('nan'))` is
    # NaN on Python, so we filter NaN to 0 explicitly.
    diff = baseline_loss - val_loss
    if diff != diff:  # NaN check (NaN != NaN by definition)
        delta = 0.0
    else:
        delta = max(0.0, float(diff))
    score = float(delta) ** SCORE_EXPONENT
    return delta, score


def discover_checkpoints(
    *, paths: Sequence[str] | None = None, directory: str | None = None
) -> list[Path]:
    """Return a deterministically-ordered list of checkpoint paths.

    Either ``paths`` (explicit comma-separated list) or ``directory``
    (auto-discover all ``*.safetensors`` / ``*.pt`` / ``*.bin``) must be
    provided. ``directory`` is walked one level deep (``glob`` not
    ``rglob``) — auto-discovery is meant to find a flat directory of
    sibling checkpoints, not crawl an entire training tree.

    Duplicates (same resolved path) are de-duplicated while preserving
    discovery order. Returned in lexicographic order within each source
    group so the leaderboard ordering is stable across runs.
    """
    if not paths and not directory:
        raise ValueError("Either paths or directory must be provided")

    collected: list[Path] = []
    seen: set[Path] = set()

    if paths:
        for p in paths:
            candidate = Path(p).expanduser()
            if candidate.is_dir():
                # Allow ``--checkpoints dir1,dir2`` to expand each dir;
                # not strictly required by the CLI spec, but it makes
                # the helper more robust to user-supplied mixed inputs.
                for f in sorted(candidate.iterdir()):
                    if f.is_file() and f.suffix.lower() in CHECKPOINT_SUFFIXES:
                        resolved = f.resolve()
                        if resolved not in seen:
                            collected.append(f)
                            seen.add(resolved)
            elif candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    collected.append(candidate)
                    seen.add(resolved)
            else:
                raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")

    if directory:
        d = Path(directory).expanduser()
        if not d.is_dir():
            raise NotADirectoryError(f"Checkpoint dir is not a directory: {d}")
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in CHECKPOINT_SUFFIXES:
                resolved = f.resolve()
                if resolved not in seen:
                    collected.append(f)
                    seen.add(resolved)

    return collected


def parse_seeds(raw: str | None, default_count: int = DEFAULT_NUM_SEEDS) -> list[int]:
    """Parse ``--seeds 1,2,3`` into ``[1, 2, 3]``.

    When ``raw`` is empty / None, a deterministic list of length
    ``default_count`` is drawn from ``random.Random(0)`` so two runs
    with the default seeds line up exactly — eliminates a confounding
    variable when comparing two leaderboards back-to-back.
    """
    if raw:
        seeds: list[int] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                seeds.append(int(tok))
            except ValueError as exc:
                raise ValueError(f"Invalid seed value: {tok!r}") from exc
        if not seeds:
            raise ValueError("--seeds was provided but parsed to an empty list")
        return seeds

    rng = random.Random(0)
    # `randrange(2**31)` keeps seeds inside the int32 range that most HF
    # / torch RNGs accept. Sorted for stable display in the markdown.
    return sorted(rng.randrange(2**31) for _ in range(default_count))


# ---------------------------------------------------------------------------
# Comparator core
# ---------------------------------------------------------------------------


EvalFn = Callable[[Path, int], float]
"""Type alias for the per-(checkpoint, seed) eval callable.

A production implementation loads the state_dict into the base model and
runs ``connito.shared.evaluate.evaluate_model`` against the validator's
eval dataloader for that seed. A test implementation can return a
synthetic val_loss directly.
"""


def compare_checkpoints(
    *,
    checkpoint_paths: Sequence[Path],
    baseline_path: Path,
    seeds: Sequence[int],
    eval_fn: EvalFn,
) -> dict[str, Any]:
    """Score each checkpoint against the baseline across all seeds.

    `eval_fn` is called once per (path, seed) pair — including for the
    baseline path itself, so the baseline's per-seed val_loss is reused
    as the subtrahend for that exact seed. The validator's eval is
    seed-dependent (the dataloader's batches change with the
    ``combined_seed``), so subtracting a baseline computed on a
    different seed would mix different batch distributions and produce
    a noisy delta.

    Returns a dict matching the schema in the Unit 4 spec:

        {
            "baseline_loss_per_seed": {seed: loss, ...},
            "checkpoints": [CheckpointResult.to_dict(), ...],
            "winner": str,
            "rank_by_score": [name, ...],
        }
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints to compare")
    if not seeds:
        raise ValueError("No seeds provided")

    seeds = list(seeds)

    # Baseline eval — once per seed. Results are reused across every
    # checkpoint, which is the whole point of fixing the baseline.
    baseline_loss_per_seed: dict[int, float] = {}
    for seed in seeds:
        logger.info(
            "Evaluating baseline",
            baseline=str(baseline_path),
            seed=seed,
        )
        loss = float(eval_fn(Path(baseline_path), int(seed)))
        baseline_loss_per_seed[int(seed)] = loss
        logger.info(
            "Baseline eval complete",
            seed=seed,
            val_loss=round(loss, 4),
        )

    results: list[CheckpointResult] = []
    for path in checkpoint_paths:
        ckpt = CheckpointResult(path=str(path), name=Path(path).stem)
        for seed in seeds:
            logger.info(
                "Evaluating checkpoint",
                checkpoint=ckpt.name,
                seed=seed,
            )
            val_loss = float(eval_fn(Path(path), int(seed)))
            baseline = baseline_loss_per_seed[int(seed)]
            delta, score = score_from_loss(baseline, val_loss)
            ckpt.per_seed.append(
                SeedResult(seed=int(seed), val_loss=val_loss, delta=delta, score=score)
            )
            logger.info(
                "Checkpoint seed eval complete",
                checkpoint=ckpt.name,
                seed=seed,
                val_loss=round(val_loss, 4),
                delta=round(delta, 4),
                score=round(score, 6),
            )
        ckpt.finalize()
        results.append(ckpt)

    # Sort by mean_score desc; break ties by lower mean_val_loss
    # (deterministic when several broken checkpoints all collapse to
    # mean_score = 0.0). Stable sort so input order survives a full tie.
    results.sort(key=lambda r: (-r.mean_score, r.mean_val_loss, r.name))

    payload: dict[str, Any] = {
        "baseline_loss_per_seed": {
            str(seed): float(loss) for seed, loss in baseline_loss_per_seed.items()
        },
        "checkpoints": [r.to_dict() for r in results],
        "winner": results[0].name if results else "",
        "rank_by_score": [r.name for r in results],
    }
    return payload


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a markdown leaderboard table from ``compare_checkpoints``."""
    lines: list[str] = []
    lines.append("# Checkpoint comparator leaderboard\n")

    baseline_per_seed = payload.get("baseline_loss_per_seed", {})
    if baseline_per_seed:
        lines.append("## Baseline per seed\n")
        lines.append("| Seed | Baseline val_loss |")
        lines.append("|---:|---:|")
        for seed, loss in sorted(baseline_per_seed.items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {seed} | {float(loss):.4f} |")
        lines.append("")

    lines.append("## Ranked checkpoints\n")
    lines.append("| Rank | Name | Mean val_loss | Mean Δ | Mean score |")
    lines.append("|---:|:---|:---:|---:|---:|")
    for rank, ckpt in enumerate(payload.get("checkpoints", []), start=1):
        mean = float(ckpt["mean_val_loss"])
        std = float(ckpt["std_val_loss"])
        delta = float(ckpt["mean_delta"])
        score = float(ckpt["mean_score"])
        lines.append(
            f"| {rank} | {ckpt['name']} | {mean:.4f} ± {std:.4f} "
            f"| {delta:.4f} | {score:.6f} |"
        )

    winner = payload.get("winner") or "(none)"
    lines.append("")
    lines.append(f"**Winner**: `{winner}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real eval-fn factory (CLI path)
# ---------------------------------------------------------------------------


def _build_real_eval_fn(
    *,
    tokenizer_name: str,
    max_batches: int,
    device_str: str | None,
) -> tuple[EvalFn, Callable[[], None]]:
    """Build a real (checkpoint_path, seed) -> val_loss callable.

    Imports heavy dependencies (torch, transformers, HF datasets) lazily
    so ``--help`` and ``import bench.checkpoint_compare`` stay fast and
    do not require GPU / HF credentials. Returns ``(eval_fn, cleanup)``
    where ``cleanup`` releases the base model + tokenizer after all
    evaluations are done.
    """
    # Lazy imports — heavy and not needed for the test path.
    import copy

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from connito.shared.dataloader import get_dataloader
    from connito.shared.evaluate import evaluate_model
    from connito.shared.helper import load_state_dict_from_path

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    logger.info("Loading tokenizer", name=tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # `DataCollatorForLanguageModeling` (used by `get_dataloader`)
        # requires a pad token even with `mlm=False` because it still
        # right-pads to the longest sequence in a batch.
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model", name=tokenizer_name)
    base_model = AutoModelForCausalLM.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Minimal config shim mirroring the fields read by
    # `connito.shared.dataloader.get_dataloader`. Wrapped in tiny
    # SimpleNamespace-like dataclasses so the dotted access (`config.
    # task.exp.data.sequence_length`) used by the dataloader still works
    # without instantiating the full pydantic `ValidatorConfig`.
    from types import SimpleNamespace

    data_cfg = SimpleNamespace(
        dataset_sources=None,
        dataset_name="allenai/c4",
        data_dir="en",
        sequence_length=512,
        per_device_train_batch_size=1,
        world_size=1,
        rank=0,
        dataset_class=None,
        vali_fraction=0.1,
        eval_source_shuffle_buffer=50_000,
        eval_source_skip_max=50_000,
        num_workers=0,
        batch_size=1,
    )
    cfg = SimpleNamespace(
        task=SimpleNamespace(exp=SimpleNamespace(data=data_cfg)),
        dataloader=SimpleNamespace(world_size=1),
    )

    def _eval_fn(path: Path, seed: int) -> float:
        state_dict = load_state_dict_from_path(path)
        if not state_dict:
            raise ValueError(f"Empty state_dict at {path}")

        model = copy.deepcopy(base_model)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        try:
            dataloader = get_dataloader(
                config=cfg,
                tokenizer=tokenizer,
                seed=seed,
                rank=0,
                world_size=1,
            )
            metrics = evaluate_model(
                step=0,
                model=model,
                eval_dataloader=dataloader,
                device=device,
                max_eval_batches=max_batches,
            )
            return float(metrics.get("val_loss", float("inf")))
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _cleanup() -> None:
        nonlocal base_model, tokenizer
        del base_model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _eval_fn, _cleanup


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.checkpoint_compare",
        description=(
            "Score N local checkpoints against a baseline using the "
            "validator's scoring formula and produce a ranked leaderboard."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated list of checkpoint paths to score.",
    )
    group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory to auto-discover .safetensors / .pt / .bin "
            "checkpoints in (non-recursive)."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to the baseline checkpoint that delta is measured against.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            f"Comma-separated list of integer seeds (default: {DEFAULT_NUM_SEEDS} "
            "deterministic seeds drawn from random.Random(0))."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=DEFAULT_MAX_BATCHES,
        help=f"Eval batches per (checkpoint, seed). Default: {DEFAULT_MAX_BATCHES}.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer / base model identifier. Default: {DEFAULT_TOKENIZER}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run eval on. Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write the leaderboard JSON to.",
    )
    parser.add_argument(
        "--md-out",
        type=str,
        default=None,
        help="Path to write the markdown leaderboard table to.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    paths_arg: list[str] | None = None
    if args.checkpoints:
        paths_arg = [tok.strip() for tok in args.checkpoints.split(",") if tok.strip()]

    checkpoint_paths = discover_checkpoints(
        paths=paths_arg,
        directory=args.checkpoint_dir,
    )
    if not checkpoint_paths:
        logger.error("No checkpoints discovered — refusing to run")
        return 2

    baseline_path = Path(args.baseline).expanduser()
    if not baseline_path.is_file():
        logger.error("Baseline checkpoint does not exist", baseline=str(baseline_path))
        return 2

    seeds = parse_seeds(args.seeds)
    logger.info(
        "Running comparator",
        checkpoints=len(checkpoint_paths),
        seeds=seeds,
        baseline=str(baseline_path),
        max_batches=args.max_batches,
    )

    eval_fn, cleanup = _build_real_eval_fn(
        tokenizer_name=args.tokenizer,
        max_batches=args.max_batches,
        device_str=args.device,
    )
    try:
        payload = compare_checkpoints(
            checkpoint_paths=checkpoint_paths,
            baseline_path=baseline_path,
            seeds=seeds,
            eval_fn=eval_fn,
        )
    finally:
        cleanup()

    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote leaderboard JSON", path=str(out_path))

    if args.md_out:
        md_path = Path(args.md_out).expanduser()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        logger.info("Wrote leaderboard markdown", path=str(md_path))

    logger.info(
        "Comparator finished",
        winner=payload.get("winner"),
        rank=payload.get("rank_by_score"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

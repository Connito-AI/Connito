"""Pure helpers for the duplicate-submission ("entropy") filter.

Miners can capture reward weight by near-duplicating a top miner's
submission (copy + noise, or a brief finetune on top of it). The exact-tie
penalty in ``finalize_round_scores`` (two miners with bit-identical
positive scores both get 0) only catches byte-level copies; this module
provides the measurement primitives for the *near*-duplicate signal:

- merge-loss: ``loss(avg(A, B))`` vs ``loss(A)`` / ``loss(B)``. Averaging
  a model with its own near-copy is a no-op (``loss_avg ≈ loss_A``),
  while averaging two genuinely different solutions moves the loss —
  down if the pair is linearly mode-connected (the protocol's own Merge
  premise), up if there is a loss barrier. Which direction dominates for
  honest pairs is an empirical question, so ``shadow_report`` logs the
  raw ``merge_penalty`` and flag verdicts under both one-sided
  predicates rather than pre-committing to either hypothesis.
- delta-cosine: cosine similarity of the *deltas* from the round's base
  snapshot. All miners train from the same round checkpoint, so raw
  weights are near-identical across the roster; the miner's actual work
  product is ``submission − base``. Copies (even noise-perturbed) have
  cosine ≈ 1; independent SGD runs decorrelate.

Everything here is side-effect-free (no GPU, no round mutation) so the
shadow pass in the background eval worker stays trivially auditable as
read-only with respect to scoring.
"""

from __future__ import annotations

from pathlib import Path

import torch

from connito.shared.app_logging import structlog
from connito.shared.helper import MINER_CHECKPOINT_SUFFIXES, parse_dynamic_filename

logger = structlog.get_logger(__name__)

# Thresholds (in val_loss units) the shadow report pre-computes flag
# verdicts for. The what-if analysis works from the raw merge_penalty, so
# these are a convenience for log-grepping, not a commitment.
SHADOW_THRESHOLDS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1)

# Threshold `enforce` mode acts on. 0.0 makes the decision a pure sign
# test; see `is_redundant`. Every larger value in SHADOW_THRESHOLDS was
# measured to flag 100% of live pairs and is unusable for enforcement.
DEFAULT_DEDUP_THRESHOLD: float = 0.0


def average_state_dicts(
    sd_a: dict[str, torch.Tensor], sd_b: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Tensor-wise mean of two submissions over their key intersection.

    Same-expert-group submissions normally share the exact key set;
    asymmetric keys (returned as a count for logging) are dropped from
    the merged dict so the overlay stays a strict subset of both models.
    Math in fp32, result cast back to the input dtype.
    """
    shared = sd_a.keys() & sd_b.keys()
    asymmetric = (len(sd_a) - len(shared)) + (len(sd_b) - len(shared))
    merged: dict[str, torch.Tensor] = {}
    for key in shared:
        a, b = sd_a[key], sd_b[key]
        if a.shape != b.shape:
            raise ValueError(
                f"shape mismatch for key {key!r}: {tuple(a.shape)} vs {tuple(b.shape)}"
            )
        merged[key] = ((a.float() + b.float()) / 2.0).to(a.dtype)
    return merged, asymmetric


def delta_cosine(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    base_sd: dict[str, torch.Tensor],
) -> tuple[float, int]:
    """Global cosine similarity between the two submissions' deltas from
    the round base, accumulated per-key in fp32 on CPU.

    Only keys present in both submissions AND the base participate (the
    base is the full model snapshot; submissions are expert shards).
    Returns ``(cosine, n_keys)``; cosine is 0.0 when either delta is a
    zero vector or no keys overlap.
    """
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    n_keys = 0
    for key in sd_a.keys() & sd_b.keys() & base_sd.keys():
        base = base_sd[key].detach().to("cpu", torch.float32)
        da = sd_a[key].detach().to("cpu", torch.float32) - base
        db = sd_b[key].detach().to("cpu", torch.float32) - base
        dot += float(torch.sum(da * db))
        norm_a_sq += float(torch.sum(da * da))
        norm_b_sq += float(torch.sum(db * db))
        n_keys += 1
    if n_keys == 0 or norm_a_sq == 0.0 or norm_b_sq == 0.0:
        return 0.0, n_keys
    return dot / ((norm_a_sq**0.5) * (norm_b_sq**0.5)), n_keys


def select_pairs(
    scores: dict[int, float],
    *,
    top_k: int,
    max_pairs: int,
    exclude: set[frozenset[int]] | None = None,
) -> list[tuple[int, int]]:
    """Rank-ordered pairs of the top-``top_k`` positive-score miners.

    Only strictly positive, finite scores participate (score 0 means "did
    not beat baseline" and is never reward-relevant). Pairs are emitted
    best-ranked-first as ``(uid_lo, uid_hi)`` tuples, minus ``exclude``
    (already-evaluated pairs from earlier idle ticks of the same round),
    capped at ``max_pairs``.
    """
    import math

    exclude = exclude or set()
    ranked = sorted(
        ((uid, s) for uid, s in scores.items() if s > 0.0 and math.isfinite(s)),
        key=lambda kv: (-kv[1], kv[0]),
    )[: max(0, top_k)]
    pairs: list[tuple[int, int]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            uid_a, uid_b = ranked[i][0], ranked[j][0]
            pair = (min(uid_a, uid_b), max(uid_a, uid_b))
            if frozenset(pair) in exclude:
                continue
            pairs.append(pair)
            if len(pairs) >= max(0, max_pairs):
                return pairs
    return pairs


def compute_merge_penalty(loss_a: float, loss_b: float, loss_avg: float) -> float:
    """``loss_avg − min(loss_a, loss_b)`` — the merge-loss statistic.

    Negative means averaging beat the better of the two sides outright:
    each model held information the other lacked. Zero or positive means
    the merge gained nothing, which is what a near-duplicate looks like.

    Factored out so the enforcing predicate and the shadow log cannot
    drift apart, and so enforcement can act on the UNROUNDED value —
    `shadow_report` rounds to 6 dp, and a tiny negative penalty rounded
    to `-0.0` would satisfy `>= 0` and flag an honest pair.
    """
    return loss_avg - min(loss_a, loss_b)


def is_redundant(
    merge_penalty: float, threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> bool:
    """Enforcement predicate: ``merge_penalty >= -threshold``.

    The `would_flag_not_better` criterion promoted from a shadow
    convenience map to the single decision `enforce` mode acts on.

    At the default ``threshold = 0.0`` this is a pure sign test. Measured
    on 7 live submissions from round 8814586: pairs containing a miner
    with a genuine per-cycle training history came out negative (−4.8e-4,
    −4.9e-4) and were not flagged, while near-copy and noise-injected
    pairs came out >= 0 (+1.2e-5 … +5.8e-4) and were. The sign carries
    the signal; the magnitude does not.

    Caller beware: the run-to-run noise floor of `merge_penalty` has NOT
    been measured. If repeat evaluations of one pair vary by more than
    ~5e-4, the sign is not stable and this predicate will zero honest
    miners at random. Measure before enabling `enforce` in production.
    """
    return merge_penalty >= -threshold


def shadow_report(
    *,
    loss_a: float,
    loss_b: float,
    loss_avg: float,
    baseline: float,
    cosine: float,
    thresholds: tuple[float, ...] = SHADOW_THRESHOLDS,
) -> dict:
    """Assemble the per-pair shadow log payload.

    ``merge_penalty = loss_avg − min(loss_a, loss_b)``: near-duplicates
    cluster at ≈ 0; genuinely distinct pairs land clearly negative
    (averaging helps — same-basin) or clearly positive (loss barrier).
    Two convenience flag maps per threshold τ:

    - ``would_flag_not_better``: ``merge_penalty ≥ −τ`` — the original
      one-sided criterion ("the merge is NOT strictly better than the
      better side by at least τ"); an exact duplicate (penalty == 0)
      flags even at τ = 0.
    - ``would_flag_band``: ``|merge_penalty| ≤ τ`` — flags only the
      near-zero band, tolerant of loss-barrier pairs.
    """
    merge_penalty = compute_merge_penalty(loss_a, loss_b, loss_avg)
    return {
        "loss_a": round(loss_a, 6),
        "loss_b": round(loss_b, 6),
        "loss_avg": round(loss_avg, 6),
        "baseline": round(baseline, 6),
        "merge_penalty": round(merge_penalty, 6),
        "delta_cosine": round(cosine, 6),
        "would_flag_not_better": {
            str(t): bool(merge_penalty >= -t) for t in thresholds
        },
        "would_flag_band": {
            str(t): bool(abs(merge_penalty) <= t) for t in thresholds
        },
    }


def recover_val_loss(score: float, baseline: float) -> float:
    """Invert ``score = (baseline − val_loss) ** 1.2`` for positive scores.

    Exact up to float roundoff for ``score > 0`` (the map is strictly
    monotonic there); callers must not use this for score == 0, where the
    clamp in ``evaluate_one_miner_sync`` destroys the information.
    """
    return baseline - score ** (1.0 / 1.2)


def find_submission_path(
    submission_dir: Path,
    hotkey: str,
    submission_block_range: tuple[int, int] | None,
) -> Path | None:
    """Locate `hotkey`'s on-disk submission for the round.

    Fallback for miners scored by the foreground pass, whose paths never
    transit the background worker (`pop_downloaded` only covers bg-scored
    miners). Same matching rules as
    ``BackgroundDownloadWorker._existing_submission``: filename embeds
    hotkey + block; the block must fall inside the round's submission
    window when one is known.
    """
    candidates = [
        p for suffix in MINER_CHECKPOINT_SUFFIXES
        for p in submission_dir.glob(f"*{suffix}")
    ]
    for path in candidates:
        if path.name.startswith(".tmp"):
            continue
        meta = parse_dynamic_filename(path.name)
        if not meta or meta.get("hotkey") != hotkey:
            continue
        if submission_block_range is not None:
            block = meta.get("block")
            if not isinstance(block, int):
                continue
            start, end = submission_block_range
            if not (start <= block <= end):
                continue
        return path
    return None

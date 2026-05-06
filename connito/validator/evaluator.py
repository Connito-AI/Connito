from __future__ import annotations

import asyncio
import copy
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import bittensor
import torch
import torch.nn as nn

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model
from connito.shared.helper import parse_dynamic_filename
from connito.shared.telemetry import track_eval_latency, track_model_load_latency

logger = structlog.get_logger(__name__)


def cleanup_non_top_submissions(
    *,
    round_obj,  # connito.validator.round.Round
    submission_dir: Path,
    top_k: int,
) -> list[str]:
    """Delete miner submission files for UIDs that have been *processed*
    this round but are not in the top-`top_k` by *this round's* score.

    Ranking uses `Round.top_scored_uids_this_round`, which reads only
    `round.scores` (populated by `mark_scored`) — the global
    `MinerScoreAggregator` is intentionally not consulted here, so a
    miner's history from prior rounds cannot pull them into the keep
    set this round and the cleanup decision is fully owned by the round
    object.

    A file is deleted iff its hotkey resolves to a UID that:
      - is in `round.failed_uids` (validation/timeout/exception, score=0
        — never top), or
      - is in `round.scored_uids` AND is not in the per-round top-k.

    Files for UIDs that have *not* yet been processed are explicitly
    skipped — this is the safety guarantee the eval workers rely on:
    bg-download has already written the shard to disk, but no eval has
    happened, so the file MUST stay until the worker can read it.
    Files belonging to hotkeys outside the round's roster (stale from a
    previous cycle, etc.) are also skipped here; the cycle-tail prune
    catches those.
    """
    submission_dir = Path(submission_dir)
    if not submission_dir.exists():
        return []

    scored, failed = round_obj.processed_uids_snapshot()
    processed = scored | failed
    if not processed:
        return []

    top_uids = round_obj.top_scored_uids_this_round(top_k)
    delete_uids = failed | (scored - top_uids)
    if not delete_uids:
        return []

    # Map UID → hotkey for the deletion target set, and collect the
    # hotkeys of every roster UID that has *not* been processed yet so
    # we can refuse to touch their files even by accident (defense in
    # depth: a hotkey clash would already be impossible, but the explicit
    # filter makes the invariant readable at the deletion site).
    hotkeys_to_delete = {
        round_obj.uid_to_hotkey[uid]
        for uid in delete_uids
        if uid in round_obj.uid_to_hotkey
    }
    unprocessed_hotkeys = {
        hotkey
        for uid, hotkey in round_obj.uid_to_hotkey.items()
        if uid not in processed
    }
    if not hotkeys_to_delete:
        return []

    deleted: list[str] = []
    for file_path in submission_dir.glob("*.pt"):
        if file_path.name.startswith(".tmp"):
            continue
        meta = parse_dynamic_filename(file_path.name)
        if not meta:
            continue
        hotkey = meta.get("hotkey")
        if hotkey in unprocessed_hotkeys:
            # Explicit safety: never delete a file for a miner whose
            # checkpoint has not been evaluated yet.
            continue
        if hotkey in hotkeys_to_delete:
            try:
                file_path.unlink(missing_ok=True)
                deleted.append(file_path.name)
            except Exception as e:
                logger.warning(
                    "cleanup_non_top_submissions: failed to delete file",
                    file=file_path.name, error=str(e),
                )
    return deleted


# Rank → score mapping used by `finalize_round_scores`. Geometric
# progression with ratio 1.5: top-1 in the round's delta ranking gets
# 2.25, runner-up 1.5, third 1.0; everyone else (and every failed /
# missing miner) gets 0.0. The geometric spacing concentrates more
# reward weight at the top — `top1 / top3 = 2.25` vs. the previous
# arithmetic mapping's `3 / 1 = 3` — while keeping the second-place
# miner closer to first than to third (`top2 / top1 = 0.667` vs.
# `top3 / top2 = 0.667`, equal ratios across tiers). Hard-coded rather
# than parameterized off `top_k_miners_to_reward` (which governs disk
# retention, not reward weight) because these values are part of the
# scoring contract — see PR #93.
_RANK_TO_SCORE: tuple[float, ...] = (2.25, 1.5, 1.0)


def finalize_round_scores(
    *,
    round_obj,  # connito.validator.round.Round
    score_aggregator,
    score_path=None,
) -> dict[int, float]:
    """Replace this round's per-miner aggregator entries with rank-based
    scores derived from `round.scores` (the delta-based per-round signal
    recorded by `mark_scored`).

    Drops every aggregator point tagged with `round.round_id` first so
    intermediate eval-time scores do not stack with the rank-based ones,
    then re-adds:

      - Top-1 by `round.scores` (delta desc): score 2.25.
      - Top-2: score 1.5.
      - Top-3: score 1.0.
      - Other scored UIDs (incl. delta=0): score 0.
      - Any UIDs whose `round.scores` value exactly equals another
        scored miner's: score 0 — a tied val_loss is evidence of a
        duplicated submission, so both sides are penalized regardless
        of where they would have ranked.
      - `validation_failed_uids` (hash/sig/expert_group/NaN-Inf): score 0.
      - `freeze_zero_uids` (no/invalid chain commit at freeze): score 0.

    Operational failures (download timeout, eval timeout, OOM, unexpected
    exception) live in `failed_uids` but NOT in `validation_failed_uids`,
    so finalize deliberately writes nothing for them — their prior EMA
    is preserved. The validator's lack of compute / bandwidth must not
    dock a miner's reward.

    Likewise, miners we never reached (submission never landed, or
    bg-eval ran out of time before claiming) are absent from every set
    and receive no entry. They keep their prior EMA and the next
    round's stalest-first prioritization gives them another shot.

    Miners whose `round.scores` value is 0.0 are explicitly excluded
    from the top-3 ranking so a "best of a bad bunch" miner cannot
    collect reward weight without actually improving over baseline.

    Returns ``{uid: rank_score}`` for the UIDs the function wrote, for
    logging.
    """
    # Snapshot all sets under the round's lock so the worker threads
    # cannot race a mark_scored / mark_failed against the read.
    scored, _failed = round_obj.processed_uids_snapshot()
    # `round.scores` is mutated under the same lock; copy it explicitly
    # rather than alias.
    with round_obj._lock:  # noqa: SLF001 — same module family
        round_scores = dict(round_obj.scores)
        validation_failed = set(round_obj.validation_failed_uids)
    freeze_zero = set(round_obj.freeze_zero_uids)
    freeze_hotkeys = dict(round_obj.freeze_zero_hotkeys)

    score_aggregator.drop_round(round_obj.round_id)

    # Rank only positive-delta miners — see the docstring's "best of a
    # bad bunch" clause.
    positive = [
        (uid, score) for uid, score in round_scores.items()
        if uid in scored and score > 0.0
    ]
    # Group by exact score value: any miner whose val_loss matches
    # another miner's gets 0 regardless of where they would have ranked.
    # `score = (baseline_loss - val_loss) ** 1.2` with float64 math —
    # exact equality between two miners is overwhelmingly evidence of a
    # duplicated submission, not legitimate parallel improvement, so
    # penalize both sides. Unique-score miners are then ranked normally
    # and slot into the 3/2/1 mapping by position.
    score_counts: dict[float, int] = {}
    for _, s in positive:
        score_counts[s] = score_counts.get(s, 0) + 1
    tied_uids = {uid for uid, s in positive if score_counts[s] > 1}
    unique_positive = [(uid, s) for uid, s in positive if score_counts[s] == 1]
    unique_positive.sort(key=lambda kv: (-kv[1], kv[0]))

    written: dict[int, float] = {}
    top_uids: set[int] = set()
    for rank, (uid, _) in enumerate(unique_positive):
        rank_score = _RANK_TO_SCORE[rank] if rank < len(_RANK_TO_SCORE) else 0.0
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=rank_score, round_id=round_obj.round_id,
        )
        written[uid] = rank_score
        top_uids.add(uid)

    # Tied positive-delta miners — explicit 0 entry per uid.
    for uid in tied_uids:
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0
        top_uids.add(uid)

    # Remaining scored UIDs (delta == 0 or beyond top-3): score 0.
    for uid in scored - top_uids:
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0

    # Explicit validation failures — submission was off-spec.
    for uid in validation_failed:
        hotkey = round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0

    # Freeze-time invalid-checkpoint penalties. Skip any UID that ended
    # up in scored/validation_failed (cannot happen today, but keep the
    # override explicit if the freeze logic ever shifts).
    for uid in freeze_zero - scored - validation_failed:
        hotkey = freeze_hotkeys.get(uid) or round_obj.uid_to_hotkey.get(uid)
        if hotkey is None:
            continue
        score_aggregator.add_score(
            uid=uid, hotkey=hotkey, score=0.0, round_id=round_obj.round_id,
        )
        written[uid] = 0.0

    if score_path is not None:
        try:
            score_aggregator.persist_atomic(score_path)
        except Exception as e:
            logger.warning(
                "finalize_round_scores: persist_atomic failed",
                round_id=round_obj.round_id, error=str(e),
            )

    logger.info(
        "finalize_round_scores: round scored by rank",
        round_id=round_obj.round_id,
        top3={
            int(u): _RANK_TO_SCORE[r]
            for r, (u, _) in enumerate(unique_positive[:3])
        },
        scored_count=len(scored),
        tied_count=len(tied_uids),
        validation_failed_count=len(validation_failed),
        freeze_zero_count=len(freeze_zero - scored - validation_failed),
    )
    return written


@dataclass(frozen=True)
class WeightSubmissionPayload:
    """Structured payload returned by `build_submission_uid_weights`.

    `uid_weights` is what the chain submitter consumes; the
    `weight_group_*` fields are populated only when the cohort-style
    emission was used (otherwise empty), and exist purely so the caller
    can log them without recomputing the selection.
    `g1_redirected_to_uid_zero` is set when the empty-G1 guard fires —
    the caller logs that case under its own info line.
    """
    uid_weights: dict[int, float]
    weight_group_1: tuple[int, ...] = ()
    weight_group_2: tuple[int, ...] = ()
    cohort_emission: bool = False
    g1_redirected_to_uid_zero: bool = False


def build_submission_uid_weights(
    *,
    score_aggregator,
    pending_round=None,
    eval_cfg=None,
    cycle_length: int | None = None,
) -> WeightSubmissionPayload:
    """Build the `{uid: weight}` payload for a single chain submission.

    Two callers share this:

    * **Restart replay** (no `Round`) — `pending_round=None`. Falls
      back to the score-aggregator avg directly so on-chain weights
      are recovered immediately after a restart instead of waiting a
      full cycle for the end-of-step-3 path.
    * **End-of-round step (3 → 4)** — `pending_round` is the live
      `Round`. When the round was frozen under the cohort scheme,
      applies the production emission rule:
        - Group 1 (`cfg.weight_group_1_share`): top-`weight_group_1_size`
          of A∪B by aggregator avg, restricted to UIDs with
          `record_count >= 3` AND a score recorded in BOTH the current
          and previous rounds (`round_id = cur` and `cur - cycle_length`).
          Empty-G1 guard: if no UID clears, redirect to `uid = 0`
          (subnet owner) so the validator stays at full emission.
        - Group 2 (`cfg.weight_group_2_share`): top-`weight_group_2_size`
          of A∪B∪C \\ G1 by aggregator avg, restricted to UIDs with
          `record_count >= 2` (no recency gate).
      A non-cohort round (`cohort_state is None`) falls back to the
      aggregator avg.

    `eval_cfg` and `cycle_length` are required for the cohort branch
    (the recency gate reads `cycle_length`, the share/size knobs come
    from `eval_cfg`). Missing either falls back to avg.
    """
    avg_scores = score_aggregator.uid_score_pairs(how="avg")
    cohort_state = getattr(pending_round, "cohort_state", None) if pending_round is not None else None
    if cohort_state is None or eval_cfg is None or cycle_length is None:
        return WeightSubmissionPayload(uid_weights=avg_scores)

    from connito.validator import round_groups as _rg

    ab_uids = list(pending_round.validation_group_a) + list(pending_round.validation_group_b)
    abc_uids = ab_uids + list(pending_round.validation_group_c)
    cur_rid = int(pending_round.round_id)
    g1_required_rids = (cur_rid, cur_rid - int(cycle_length))

    ab_qualified = [
        u for u in ab_uids
        if score_aggregator.record_count(u) >= 3
        and score_aggregator.has_round_ids(u, g1_required_rids)
    ]
    g1 = _rg.select_top_n_by_local_score(
        ab_qualified,
        avg_scores,
        n=eval_cfg.weight_group_1_size,
    )
    g1_redirected = False
    if not g1:
        g1 = (0,)
        g1_redirected = True
    g1_set = set(g1)
    g2_pool = [
        u for u in abc_uids
        if u not in g1_set
        and score_aggregator.record_count(u) >= 2
    ]
    g2 = _rg.select_top_n_by_local_score(
        g2_pool,
        avg_scores,
        n=eval_cfg.weight_group_2_size,
    )
    uid_weights = _rg.compute_uid_weights(
        weight_group_1=g1,
        weight_group_2=g2,
        local_scores=avg_scores,
        group_1_share=eval_cfg.weight_group_1_share,
        group_2_share=eval_cfg.weight_group_2_share,
    )
    return WeightSubmissionPayload(
        uid_weights=uid_weights,
        weight_group_1=g1,
        weight_group_2=g2,
        cohort_emission=True,
        g1_redirected_to_uid_zero=g1_redirected,
    )


def _prune_non_top_after_eval(
    *,
    config,
    round_obj,
) -> None:
    """Wrapper around `cleanup_non_top_submissions` that swallows errors
    so a cleanup failure can never abort eval flow.
    """
    try:
        deleted = cleanup_non_top_submissions(
            round_obj=round_obj,
            submission_dir=Path(config.ckpt.miner_submission_path),
            top_k=int(config.evaluation.top_k_miners_to_reward),
        )
    except Exception as e:
        logger.warning("foreground eval: post-eval cleanup failed", error=str(e))
        return
    if deleted:
        logger.info(
            "foreground eval: pruned non-top miner submissions",
            round_id=round_obj.round_id,
            deleted=len(deleted),
            files=deleted,
        )


# -----------------------------------------------------------------------------
def validate_miner_submission(
    *,
    round_obj,  # connito.validator.round.Round
    uid: int,
    model_path: str | Path,
    expert_group_assignment,
) -> str | None:
    """Run the existing `ChainCheckpoint.validate(...)` against a miner's
    on-disk submission before it is fed to `evaluate_one_miner`.

    Returns ``None`` on success. On failure returns a short reason string —
    one of ``no_chain_commit | signature | hash | expert_group | nan_inf``,
    or a generic ``"unknown"`` if the helper raised. The reason is intended
    to be plumbed into telemetry labels and log lines.

    The chain checkpoint is read from `round_obj.uid_to_chain_checkpoint`
    so this never re-fetches anything from the chain. The check itself is
    `ChainCheckpoint.validate(expert_group_assignment=…)`, which runs:

    - `_verify_signature` — the chain hotkey signed `model_hash`.
    - `_verify_hash` — the on-disk shard's hash matches the chain commit.
    - `_verify_expert_group` — every routed-expert key in the state dict
      belongs to the miner's assigned group, and no tensor contains NaN/Inf.
    """
    chain_checkpoint = round_obj.uid_to_chain_checkpoint.get(int(uid))
    if chain_checkpoint is None:
        return "no_chain_commit"

    # `validate()` reads the state dict from `chain_checkpoint.path`; point
    # it at the on-disk submission for this round.
    chain_checkpoint.path = Path(model_path)

    try:
        ok = chain_checkpoint.validate(expert_group_assignment=expert_group_assignment)
    except Exception as e:
        logger.warning(
            "validate_miner_submission: validate() raised",
            uid=int(uid), error=str(e), exc_info=True,
        )
        return "unknown"

    if ok:
        return None

    # `validate()` already logged a structured warning per failed sub-check.
    # Map the per-check booleans to a single short reason for telemetry.
    if not getattr(chain_checkpoint, "signature_verified", False):
        return "signature"
    if not getattr(chain_checkpoint, "hash_verified", False):
        return "hash"
    if not getattr(chain_checkpoint, "expert_group_verified", False):
        # _verify_expert_group folds the NaN/Inf scan in with the routing
        # check, so we cannot tell them apart from the booleans alone. The
        # underlying logger.warning at the failure site distinguishes them.
        return "expert_group_or_nan"
    return "unknown"


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinerEvalJob:
    uid: int
    hotkey: str
    model_path: str
    step: int
    score: float = 0.0


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------

# def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
#     sd = torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
#     model = copy.deepcopy(base_model)
#     model.load_state_dict(sd, strict=False)
#     return model.to(device)

@track_model_load_latency()
def load_model_from_path(path: str, base_model: nn.Module, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}: {type(ckpt).__name__}")

    if len(sd) == 0:
        raise ValueError(f"Checkpoint at {path} has empty model_state_dict")

    model = copy.deepcopy(base_model)

    # Keys in each state_dict (before loading)
    base_sd = base_model.state_dict()
    base_keys = set(base_sd.keys())
    ckpt_keys = set(sd.keys())

    # 1) Params that are the same across both dicts (intersection).
    #    (Optional: filter to ones with matching shapes too.)
    common_keys = base_keys & ckpt_keys
    common_same_shape = {k for k in common_keys if base_sd[k].shape == sd[k].shape}

    # 2) Keys containing 'expert' that exist in the checkpoint but NOT in the base model
    expert_not_in_base = {k for k in ckpt_keys - base_keys if "expert" in k}

    # 3) "expert" keys in base_model but NOT in checkpoint/common_keys
    expert_in_base_not_common = {k for k in (base_keys - common_keys) if "expert" in k}

    if len(common_same_shape) == 0:
        logger.warning(
            "No compatible keys between checkpoint and base model — "
            "checkpoint is likely from a different architecture or naming convention",
            ckpt_key_count=len(ckpt_keys),
            base_key_count=len(base_keys),
            sample_ckpt_keys=sorted(k for k in ckpt_keys if "expert" in k)[:5],
            sample_base_keys=sorted(k for k in base_keys if "expert" in k)[:5],
        )
    elif expert_not_in_base:
        logger.warning(
            "Expert keys in checkpoint not found in base model",
            expert_not_in_base=len(expert_not_in_base),
            sample_keys=sorted(expert_not_in_base)[:5],
        )
    elif common_same_shape != common_keys:
        logger.warning(
            "Some common keys have mismatched shapes",
            common_keys=len(common_keys),
            common_same_shape=len(common_same_shape),
            shape_mismatch=len(common_keys - common_same_shape),
            sample_mismatched=sorted(common_keys - common_same_shape)[:5],
        )
    else:
        logger.debug(
            "Key summary",
            common_keys=len(common_keys),
            common_same_shape=len(common_same_shape),
            expert_in_base_not_common=len(expert_in_base_not_common),
        )

    # Load weights (strict=False so missing/unexpected are allowed)
    incompatible = model.load_state_dict(sd, strict=False)

    # # Extra helpful debug (optional)
    # if incompatible.missing_keys:
    #     print(f"[load_model] missing keys (first 50): {incompatible.missing_keys[:50]}")
    # if incompatible.unexpected_keys:
    #     print(f"[load_model] unexpected keys (first 50): {incompatible.unexpected_keys[:50]}")

    return model.to(device)


async def _evaluate_on_fresh_loader(
    *,
    config,
    tokenizer,
    combinded_seed: str,
    step: int,
    model: nn.Module,
    device: torch.device,
    max_eval_batches: int,
    rank: int | None = None,
) -> dict:
    """Build a fresh eval dataloader and run evaluate_model on the given model.

    Every caller shares the same `combinded_seed`, so the baseline and all
    miner evals see the same batches — the deltas are comparable.
    """
    dataloader = await asyncio.to_thread(
        get_dataloader,
        config=config,
        tokenizer=tokenizer,
        seed=combinded_seed,
        rank=0,
        world_size=config.dataloader.world_size,
    )

    @track_eval_latency()
    def _run():
        return evaluate_model(step, model, dataloader, device, max_eval_batches, rank)

    try:
        return await asyncio.to_thread(_run)
    finally:
        del dataloader


async def evaluate_one_miner(
    *,
    config,
    model_path: str | Path,
    uid: int,
    hotkey: str,
    base_model: nn.Module,
    tokenizer,
    combined_seed: str,
    device: torch.device,
    baseline_loss: float,
    step: int,
    round_id: int | None = None,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
) -> "MinerEvalJob | None":
    """Evaluate a single miner and return the per-round delta-based score.

    Shared between foreground (`evaluate_foreground_round`) and the
    `BackgroundEvalWorker`. `base_model` is treated as read-only — caller
    is responsible for not mutating it across calls so successive miners
    see an identical baseline.

    The returned `MinerEvalJob.score` is the raw `(baseline_loss - val_loss)
    ** 1.2` signal; the caller stores it in `round.scores` via
    `mark_scored`. The actual reward score sent to the chain is rank-based
    and written by `finalize_round_scores` at end of round — this function
    does not touch the global `MinerScoreAggregator`.

    Returns a `MinerEvalJob` on success (so the caller can later use
    `model_path` for gradient aggregation), or None on failure.
    """
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        miner_model = await asyncio.to_thread(load_model_from_path, str(model_path), base_model, device)

        try:
            metrics = await _evaluate_on_fresh_loader(
                config=config,
                tokenizer=tokenizer,
                combinded_seed=combined_seed,
                step=step,
                model=miner_model,
                device=device,
                max_eval_batches=max_eval_batches,
                rank=rank,
            )
        finally:
            del miner_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = float(metrics.get("val_loss", 100))
        delta = max(0.0, baseline_loss - val_loss)
        # `score` is the per-round delta-based signal stored on
        # `MinerEvalJob` and recorded in `round.scores` by the caller via
        # `mark_scored`. The aggregator is intentionally NOT updated here
        # — `finalize_round_scores` is the sole writer for this round's
        # aggregator entries (see PR #93 introducing rank-based scoring).
        score = delta ** 1.2
        logger.info(
            "evaluate_one_miner: complete",
            uid=int(uid),
            hotkey=hotkey[:6],
            val_loss=round(val_loss, 4),
            baseline_loss=round(baseline_loss, 4),
            delta=round(delta, 4),
            score=round(score, 6),
            round_id=round_id,
        )
        return MinerEvalJob(
            uid=int(uid),
            hotkey=hotkey,
            model_path=str(model_path),
            step=int(step),
            score=float(score),
        )
    except torch.cuda.OutOfMemoryError:
        logger.error("evaluate_one_miner: OOM", uid=int(uid))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.exception("evaluate_one_miner: failed", uid=int(uid), error=str(e))
        return None


async def evaluate_foreground_round(
    *,
    config,
    round_obj,  # connito.validator.round.Round
    subtensor: bittensor.Subtensor,
    step: int,
    device: torch.device,
    base_model: nn.Module,
    tokenizer,
    end_block: int,
    expert_group_assignment,
    poll_interval_sec: float = 6.0,
    per_miner_eval_timeout_sec: float | None = None,
    completed_out: list[MinerEvalJob] | None = None,
) -> list[MinerEvalJob]:
    """Foreground (step 2): evaluate the round's top-N miners during
    Submission + Validate.

    Walks `round_obj.foreground_uids` only and calls `evaluate_one_miner`
    for each. Miner checkpoints are made available locally by the
    `BackgroundDownloadWorker` (HF); this function does not pull from HF
    itself. UIDs that exceed the per-miner budget or fail to land by
    `end_block` are left unclaimed so the `BackgroundEvalWorker` can pick
    them up in step 3.

    If `completed_out` is provided, finished `MinerEvalJob`s are appended
    to that list as they complete (in addition to being returned at the
    end). Callers wrap this coroutine with `asyncio.wait_for` to enforce
    a wall-clock cap; on cancellation the local return value is lost,
    but `completed_out` retains every miner that finished scoring before
    the deadline. `list.append` is atomic, so there is no partial-append
    race during cancellation.
    """
    # Lazy imports — connito.shared.cycle imports this module, so a top-
    # level import would create a cycle.
    from connito.shared.cycle import BITTENSOR_BLOCK_TIME_SECONDS, gather_validation_job

    # Baseline once against the round's input model (= live `base_model`,
    # which equals round.model_snapshot_cpu since the foreground runs
    # before Merge(K)).
    baseline_metrics = await _evaluate_on_fresh_loader(
        config=config,
        tokenizer=tokenizer,
        combinded_seed=round_obj.seed,
        step=step,
        model=base_model,
        device=device,
        max_eval_batches=EVAL_MAX_BATCHES,
    )
    baseline_loss = float(baseline_metrics.get("val_loss", 100))
    del baseline_metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    foreground_set = set(round_obj.foreground_uids)
    completed: list[MinerEvalJob] = completed_out if completed_out is not None else []

    logger.info(
        "foreground eval: starting",
        round_id=round_obj.round_id,
        foreground_uids=list(round_obj.foreground_uids),
        end_block=end_block,
        current_block=subtensor.block,
        baseline_loss=round(baseline_loss, 4),
        per_miner_eval_timeout_sec=per_miner_eval_timeout_sec,
    )

    poll_idx = 0
    while subtensor.block <= end_block:
        try:
            discovered = gather_validation_job(
                config,
                subtensor,
                step=step,
                validator_miner_assignment=round_obj.validator_miner_assignment,
            )
        except Exception as e:
            logger.warning("foreground eval: gather_validation_job failed", error=str(e))
            discovered = []

        # Walk foreground UIDs in incentive order; pick up any whose
        # checkpoint has landed and is not yet claimed/scored.
        by_uid: dict[int, MinerEvalJob] = {j.uid: j for j in discovered if j.uid in foreground_set}
        scored_count = sum(1 for u in foreground_set if u in round_obj.scored_uids)
        current_block = subtensor.block
        logger.info(
            "foreground eval: poll",
            round_id=round_obj.round_id,
            poll_idx=poll_idx,
            current_block=current_block,
            blocks_remaining=max(0, end_block - current_block),
            discovered_total=len(discovered),
            discovered_in_foreground=len(by_uid),
            ready_uids=sorted(by_uid.keys()),
            scored=scored_count,
            foreground_total=len(foreground_set),
        )
        poll_idx += 1
        progressed = False
        phase_deadline_crossed = False
        for uid in round_obj.foreground_uids:
            if uid not in by_uid:
                continue
            # Hard-stop before claiming if Validate has ended. Without
            # this, the inner for-loop walks every foreground UID before
            # the outer `subtensor.block > end_block` check fires, so a
            # 5-miner round can spill ~5 × per_miner_eval_timeout_sec
            # past end_block.
            block_now = subtensor.block
            if block_now > end_block:
                phase_deadline_crossed = True
                break
            if not round_obj.claim_for_foreground(uid):
                continue
            job = by_uid[uid]
            hotkey = round_obj.uid_to_hotkey[uid]
            progressed = True

            # Verify the on-disk submission against the chain commit (signed
            # hash, hash, expert-group ownership, NaN/Inf scan) BEFORE the
            # GPU eval. A failure here means the submission is off-spec —
            # mark the miner failed so the missed-submission penalty pass
            # zeroes their score for the round.
            fail_reason = await asyncio.to_thread(
                validate_miner_submission,
                round_obj=round_obj,
                uid=uid,
                model_path=job.model_path,
                expert_group_assignment=expert_group_assignment,
            )
            if fail_reason is not None:
                # Invalid checkpoint (no chain commit / signature / hash /
                # expert_group / NaN-Inf): mark validation-failed so
                # `finalize_round_scores` records score=0 at end of round.
                # Operational failures below (timeout / OOM / unexpected
                # exception) use plain `mark_failed`, which leaves the
                # miner's prior EMA untouched — those failures are not
                # the miner's fault.
                logger.warning(
                    "foreground eval: submission failed validation — will record score=0 at finalize",
                    uid=uid, hotkey=hotkey[:6],
                    round_id=round_obj.round_id,
                    reason=fail_reason,
                )
                from connito.shared.telemetry import inc_error
                inc_error(component="foreground_eval", kind="validation")
                round_obj.mark_validation_failed(uid)
                _prune_non_top_after_eval(
                    config=config,
                    round_obj=round_obj,
                )
                continue

            # Cap the per-miner eval at min(configured_timeout, time_to_end_block)
            # so a long-running eval can't itself overrun the phase boundary.
            sec_to_end_block = max(0.0, (end_block - block_now) * BITTENSOR_BLOCK_TIME_SECONDS)
            effective_timeout: float | None = sec_to_end_block
            if per_miner_eval_timeout_sec is not None:
                effective_timeout = min(per_miner_eval_timeout_sec, sec_to_end_block)
            if effective_timeout <= 0:
                round_obj.release_claim(uid)
                phase_deadline_crossed = True
                break

            eval_coro = evaluate_one_miner(
                config=config,
                model_path=job.model_path,
                uid=uid,
                hotkey=hotkey,
                base_model=base_model,
                tokenizer=tokenizer,
                combined_seed=round_obj.seed,
                device=device,
                baseline_loss=baseline_loss,
                step=step,
                round_id=round_obj.round_id,
            )
            try:
                evaluated = await asyncio.wait_for(eval_coro, timeout=effective_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "foreground eval: per-miner timeout — marking failed",
                    uid=uid, hotkey=hotkey[:6],
                    timeout_sec=round(effective_timeout, 2),
                )
                round_obj.mark_failed(uid)
                _prune_non_top_after_eval(
                    config=config,
                    round_obj=round_obj,
                )
                continue
            except Exception as e:
                logger.exception("foreground eval: unexpected failure", uid=uid, error=str(e))
                round_obj.mark_failed(uid)
                _prune_non_top_after_eval(
                    config=config,
                    round_obj=round_obj,
                )
                continue

            if evaluated is None:
                round_obj.mark_failed(uid)
                _prune_non_top_after_eval(
                    config=config,
                    round_obj=round_obj,
                )
                continue
            round_obj.mark_scored(uid, evaluated.score)
            completed.append(evaluated)
            _prune_non_top_after_eval(
                config=config,
                round_obj=round_obj,
            )

        # Stop once every top-N UID is scored or the phase boundary hits.
        scored_top_n = sum(1 for u in foreground_set if u in round_obj.scored_uids)
        if scored_top_n >= len(foreground_set):
            break

        if phase_deadline_crossed or subtensor.block > end_block:
            logger.info(
                "foreground eval: validate phase ended — stopping",
                round_id=round_obj.round_id,
                end_block=end_block,
                current_block=subtensor.block,
                scored=scored_top_n,
                foreground_total=len(foreground_set),
            )
            break
        if not progressed:
            await asyncio.sleep(poll_interval_sec)

    logger.info(
        "foreground eval: complete",
        round_id=round_obj.round_id,
        top_n=len(foreground_set),
        scored=len(completed),
        spilled=len(foreground_set) - len(completed),
    )
    return completed

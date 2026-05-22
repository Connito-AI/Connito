"""Multi-seed best-checkpoint selection before MinerCommit2 (Unit 15 of the
SN102 miner optimization).

Validators score miners via
``score = max(0, baseline_loss - miner_val_loss) ** 1.2``
(``connito/validator/evaluator.py:787``). Per-validator weight variance
is huge — the same submission can score 0.37 from one validator and
0.0066 from another — so the marginal win comes from picking a
submission whose val_loss is robust across multiple seeds, not the
single-seed peak.

This module runs a local replica of the validator's eval pass over the
last N saved checkpoints (plus optional EMA snapshot) against multiple
seeds locally, and returns the checkpoint with the lowest **mean**
val_loss. The miner's commit_worker then signs and uploads that path
instead of always taking the most recent train step.

Defaults are tuned to be cheap — 3 candidates, 3 seeds, 20 batches each
— so the selection adds a few minutes at most before MinerCommit2. The
feature is opt-in via ``config.commit.selection.enabled`` so existing
miner runs are bit-for-bit unaffected when disabled.

Note about local_scorer (Unit 1): when ``bench/local_scorer`` is
available, its scoring logic is reused. When not (parallel Unit 1
build), the scoring is inlined via ``connito/shared/evaluate.py``
directly — the same code path the validator uses for the canonical
score.
"""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from connito.shared.config import MinerConfig

logger = structlog.get_logger(__name__)


def _seed_to_hex(seed: int) -> str:
    """Convert an integer seed to the hex digest shape ``get_dataloader`` expects.

    Mirrors ``connito.miner.local_eval._seed_to_hex``: the dataloader
    reads ``int(seed[:8], 16)`` to derive ``int_seed`` for HF shuffle /
    skip. SHA-256 of the integer's textual form gives a stable,
    well-distributed hex string.
    """
    return hashlib.sha256(str(int(seed)).encode("utf-8")).hexdigest()


class BestCheckpointSelector:
    """Pick the lowest mean-val-loss checkpoint across multiple seeds.

    Constructed once per commit job. ``select(...)`` iterates the
    supplied candidate paths, loads each into a clone of ``base_model``,
    runs ``evaluate_model`` against a seed-derived dataloader for every
    configured seed, and returns the path with the lowest mean
    val_loss along with a structured selection report.

    Parameters
    ----------
    config:
        :class:`MinerConfig`. Reads ``commit.selection.max_eval_batches``
        and ``task.exp.data.world_size`` for sharding. The dataloader is
        built with ``train=False`` to mirror the validator's eval split
        semantics.
    tokenizer:
        Tokenizer used to build the eval dataloader.
    base_model:
        Model to load each candidate's state_dict into. Each call uses
        ``copy.deepcopy(base_model)`` so candidates are evaluated
        independently and the supplied ``base_model`` is never mutated.
    device:
        Device the model should live on during evaluation.
    seeds:
        List of integer seeds. The eval is run once per seed per
        candidate; the lowest **mean** val_loss across seeds wins.
        Empty list → :meth:`select` raises ``ValueError``.
    max_batches:
        Per-seed batch cap forwarded to ``evaluate_model``. Defaults
        align with ``CommitSelectionCfg.max_eval_batches``.
    rank:
        DP rank, forwarded to ``get_dataloader`` for sharding.
    """

    def __init__(
        self,
        config: "MinerConfig",
        tokenizer: "PreTrainedTokenizerBase",
        base_model: nn.Module,
        device: torch.device,
        seeds: list[int],
        max_batches: int = 20,
        rank: int = 0,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        # Defensive copy — caller may mutate the supplied list later.
        self.seeds = [int(s) for s in seeds]
        self.max_batches = int(max_batches)
        self.rank = int(rank)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select(self, candidate_paths: list[Path]) -> tuple[Path, dict]:
        """Score each candidate across all seeds; return the winner + report.

        Returns
        -------
        tuple[Path, dict]
            ``(winner_path, report)`` where ``report`` is a dict of
            shape::

                {
                    "results": [
                        {
                            "path": str,
                            "mean_val_loss": float,
                            "per_seed": [{"seed": int, "val_loss": float}, ...],
                        },
                        ...
                    ],
                    "winner": str,
                    "winner_mean_val_loss": float,
                    "seeds": list[int],
                    "max_batches": int,
                }

        Raises
        ------
        ValueError
            When ``candidate_paths`` is empty or ``self.seeds`` is empty.
            Both are caller-side configuration errors — the commit
            worker should fall back to the legacy "latest checkpoint"
            behavior in those cases.
        """
        if not candidate_paths:
            raise ValueError(
                "BestCheckpointSelector.select: candidate_paths is empty; "
                "cannot run selection."
            )
        if not self.seeds:
            raise ValueError(
                "BestCheckpointSelector.select: no seeds configured; "
                "cannot evaluate candidates."
            )

        results: list[dict] = []
        for path in candidate_paths:
            per_seed: list[dict] = []
            try:
                miner_model = self._load_into_base(path)
            except Exception as e:
                # A candidate that fails to load (corrupt shard, missing
                # file, etc.) is recorded as +inf so it cannot win the
                # selection but the round still completes against the
                # remaining candidates. Surfacing the error in logs is
                # important so operators see why their checkpoint was
                # skipped.
                logger.warning(
                    "best_ckpt_selector: failed to load candidate",
                    path=str(path),
                    error=str(e),
                )
                results.append({
                    "path": str(path),
                    "mean_val_loss": float("inf"),
                    "per_seed": [
                        {"seed": int(seed), "val_loss": float("inf"), "load_error": True}
                        for seed in self.seeds
                    ],
                    "load_error": True,
                })
                continue

            try:
                for seed in self.seeds:
                    val_loss = self._score_one_seed(model=miner_model, seed=int(seed))
                    per_seed.append({"seed": int(seed), "val_loss": float(val_loss)})
            finally:
                del miner_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Mean over finite val_losses; if any seed returned +inf
            # (every eval batch was NaN/Inf for that seed), the mean is
            # +inf and that candidate cannot win — same clamp as the
            # validator-side `max(0, baseline - val_loss)`.
            mean_loss = sum(p["val_loss"] for p in per_seed) / len(per_seed)
            results.append({
                "path": str(path),
                "mean_val_loss": float(mean_loss),
                "per_seed": per_seed,
            })

            logger.info(
                "best_ckpt_selector: scored candidate",
                path=str(path),
                mean_val_loss=round(mean_loss, 4) if math.isfinite(mean_loss) else mean_loss,
                per_seed_val_loss=[
                    round(p["val_loss"], 4) if math.isfinite(p["val_loss"]) else p["val_loss"]
                    for p in per_seed
                ],
            )

        best = min(results, key=lambda r: r["mean_val_loss"])
        # All candidates failed — every mean is +inf. Surface this
        # explicitly so the commit worker can decide to fall back to the
        # legacy "use latest" path rather than uploading a known-broken
        # checkpoint.
        if not math.isfinite(best["mean_val_loss"]):
            logger.warning(
                "best_ckpt_selector: all candidates produced non-finite mean val_loss; "
                "winner is the first by enumeration order",
                candidate_count=len(results),
            )

        winner_path = Path(best["path"])
        report = {
            "results": results,
            "winner": best["path"],
            "winner_mean_val_loss": float(best["mean_val_loss"]),
            "seeds": list(self.seeds),
            "max_batches": int(self.max_batches),
        }

        logger.info(
            "best_ckpt_selector: selection complete",
            winner=str(winner_path),
            winner_mean_val_loss=round(best["mean_val_loss"], 4)
            if math.isfinite(best["mean_val_loss"])
            else best["mean_val_loss"],
            candidate_count=len(results),
            seed_count=len(self.seeds),
        )
        return winner_path, report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _score_one_seed(self, *, model: nn.Module, seed: int) -> float:
        """Build a seed-derived dataloader and run a single eval pass.

        Mirrors ``LocalEvaluator._run_one_seed``: uses ``train=False``
        so the eval pulls from the validation split (validator-replica
        semantics for canonical scoring), and forwards the integer
        seed as a hex digest so the dataloader's HF shuffle/skip lands
        at a different per-source offset for each seed.
        """
        seed_hex = _seed_to_hex(int(seed))
        loader = get_dataloader(
            config=self.config,
            tokenizer=self.tokenizer,
            seed=seed_hex,
            rank=self.rank,
            world_size=self.config.task.exp.data.world_size,
            train=False,
        )
        try:
            with torch.no_grad():
                metrics = evaluate_model(
                    step=0,
                    model=model,
                    eval_dataloader=loader,
                    device=self.device,
                    max_eval_batches=int(self.max_batches),
                    rank=int(self.rank),
                )
        finally:
            del loader

        return float(metrics.get("val_loss", float("inf")))

    def _load_into_base(self, path: Path) -> nn.Module:
        """Load ``path``'s state_dict into a clone of ``self.base_model``.

        Reuses the validator's ``load_model_from_path`` helper when
        ``path`` is a single checkpoint file (``.safetensors`` / ``.pt``)
        so the safety gates (no pickle execution, key-compatibility
        warnings) are identical to the canonical validator code path.

        Directory inputs (e.g. ``globalver_X_inneropt_Y/`` containing
        a ``model_expgroup_{gid}.safetensors`` shard) are loaded via
        ``compile_full_state_dict_from_path`` and copied into a deep
        copy of ``base_model`` — same approach the trainer uses to
        resume from a sharded checkpoint, but without touching the
        optimizer / scheduler / dataloader state.
        """
        # Local imports keep this module light at import time; the model
        # / checkpoint stack pulls in CUDA, safetensors, etc. which are
        # heavy enough that we don't want them on every miner-side
        # import path.
        import copy as _copy

        from connito.shared.checkpoint_helper import compile_full_state_dict_from_path

        path = Path(path)
        if path.is_file():
            # Single-file path → reuse the validator helper for parity.
            from connito.validator.evaluator import load_model_from_path

            return load_model_from_path(str(path), self.base_model, self.device)

        # Directory path → assemble the state_dict from expert-group
        # shards in the directory. The trainer saves to a directory
        # whose name encodes ``globalver_X_inneropt_Y`` and writes
        # ``model_expgroup_{gid}.safetensors`` inside it.
        expert_group_id = self.config.task.exp.group_id
        state_dict = compile_full_state_dict_from_path(
            path, expert_groups=[expert_group_id],
        )
        if not state_dict:
            raise ValueError(
                f"Checkpoint at {path} produced an empty state_dict "
                f"(expert_group={expert_group_id})"
            )

        model = _copy.deepcopy(self.base_model)
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device)


def discover_candidate_checkpoints(
    checkpoint_dir: Path,
    candidate_count: int,
    *,
    include_ema: bool = True,
    ema_filename: str = "ema.safetensors",
) -> list[Path]:
    """Return up to ``candidate_count`` most-recent checkpoints under
    ``checkpoint_dir``, plus the EMA snapshot if requested.

    The miner trainer saves to directories named
    ``globalver_X_inneropt_Y/`` (see ``connito/miner/train.py`` around
    line 640); this helper reuses ``build_local_checkpoints`` so the
    sort order matches ``select_best_checkpoint`` exactly.

    Parameters
    ----------
    checkpoint_dir:
        The miner's local checkpoint root (``config.ckpt.checkpoint_path``).
    candidate_count:
        Maximum number of recent training checkpoints to consider. The
        EMA snapshot (when present and ``include_ema=True``) is added
        on top of this count.
    include_ema:
        When True, look for ``ema_filename`` (or a directory of that
        name) inside ``checkpoint_dir`` and prepend it to the
        candidates if it exists. The EMA artifact is produced by Unit
        13 (parallel build); this selector tolerates its absence so a
        miner running Unit 15 without Unit 13 still benefits from the
        multi-seed selection over training checkpoints.
    ema_filename:
        Filename or directory name of the EMA snapshot artifact.

    Returns
    -------
    list[Path]
        Ordered list of candidate paths, EMA snapshot first (if
        present), then most-recent training checkpoints. Empty list if
        no candidates are available.
    """
    # Local import avoids a circular dependency at module load — the
    # checkpoints module imports from shared.config which imports the
    # full pydantic stack; keeping the import here keeps this module's
    # import time low for callers that don't end up running selection.
    from connito.shared.checkpoints import build_local_checkpoints

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.debug(
            "best_ckpt_selector: checkpoint directory does not exist; no candidates",
            checkpoint_dir=str(checkpoint_dir),
        )
        return []

    candidates: list[Path] = []

    if include_ema:
        ema_path = checkpoint_dir / ema_filename
        if ema_path.exists():
            candidates.append(ema_path)
            logger.debug(
                "best_ckpt_selector: EMA snapshot found",
                ema_path=str(ema_path),
            )

    local = build_local_checkpoints(checkpoint_dir, role="miner").ordered()
    for ckpt in local[: int(candidate_count)]:
        if ckpt.path is not None:
            candidates.append(Path(ckpt.path))

    logger.info(
        "best_ckpt_selector: discovered candidates",
        candidate_count=len(candidates),
        checkpoint_dir=str(checkpoint_dir),
        requested_training_count=int(candidate_count),
        include_ema=bool(include_ema),
        candidates=[str(p) for p in candidates],
    )
    return candidates

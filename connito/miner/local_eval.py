"""Mid-training validator-replica mini-eval (Unit 8 of the SN102 miner optimization).

The validator scores a miner once per round (~58 min) using
``connito/shared/evaluate.py:evaluate_model`` over 50 batches against a
seed-derived deterministic dataloader. Score is
``max(0, baseline_loss - miner_val_loss) ** 1.2`` per
``connito/validator/evaluator.py:787``.

``LocalEvaluator`` runs a much cheaper replica of that pass inside the
training loop every N optimizer steps so a regression surfaces in
real time — operators see ``miner_local_val_loss`` climb in Grafana
within seconds of the bad step, instead of waiting for the next
validator round to score the submission.

Design notes:

* The eval is purely observational. It never touches gradients, the
  optimizer, the scheduler, or the dataloader iterator the training
  loop is consuming. The model is briefly switched to ``.eval()`` and
  restored via a ``try/finally`` so a crash inside ``evaluate_model``
  can never leave the model in eval mode.
* The dataloader is built per-call with the same seeding semantics
  as the validator (``get_dataloader(..., seed=<hex>)``), but with a
  small ``max_batches`` so we don't burn through the per-cycle training
  budget. Default is 10 batches vs the validator's 50.
* Baseline comparison is optional. When enabled and a baseline model
  is supplied at construction, we evaluate the baseline ONCE per
  ``LocalEvaluator`` instance (lazy, on the first ``run`` call) per
  seed and cache the result. Re-evaluating the baseline every step
  would more than double the eval cost for no signal — the baseline
  by definition does not change.
* All errors are swallowed by the caller in ``train.py`` (this module
  raises so the caller can decide). Telemetry must not be allowed to
  crash training.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from connito.shared.app_logging import structlog
from connito.shared.dataloader import get_dataloader
from connito.shared.evaluate import evaluate_model
from connito.shared.telemetry import (
    MINER_LOCAL_DELTA,
    MINER_LOCAL_EVAL_STEPS_TOTAL,
    MINER_LOCAL_VAL_LOSS,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from connito.shared.config import MinerConfig

logger = structlog.get_logger(__name__)


def _seed_to_hex(seed: int) -> str:
    """Convert an integer seed to the hex digest shape ``get_dataloader``
    expects.

    The dataloader reads ``int(seed[:8], 16)`` to derive ``int_seed`` for
    HF shuffle/skip. SHA-256 of the integer's textual form gives us a
    stable, well-distributed hex string and lets two seeds that happen
    to differ only in their low bytes still produce very different
    ``int_seed`` values after the truncation.
    """
    return hashlib.sha256(str(int(seed)).encode("utf-8")).hexdigest()


class LocalEvaluator:
    """Validator-replica mini-eval owned by the miner training loop.

    Constructed once at training-loop start. Each call to :meth:`run`
    performs one short eval pass per configured seed, writes results to
    Prometheus, and returns an aggregate dict for the caller to log via
    :class:`connito.shared.metrics.MetricLogger`.

    Parameters
    ----------
    config:
        :class:`MinerConfig`. Reads ``local_eval.max_batches``,
        ``local_eval.seeds``, ``local_eval.use_baseline_compare``, and
        ``task.exp.data.world_size`` for sharding.
    tokenizer:
        Tokenizer to build the eval dataloader with. Should be the
        same tokenizer instance the trainer uses; the dataloader does
        not own the tokenizer's vocabulary.
    device:
        Device the miner model lives on. Eval runs on this device with
        autocast matching the trainer's precision.
    rank:
        DP rank of this worker. Surfaced as a Prometheus label and
        forwarded to ``get_dataloader`` for sharding.
    baseline_model:
        Optional frozen model to compare against. When ``None`` (or
        when ``config.local_eval.use_baseline_compare=False``) the
        delta gauge is not written.
    expert_group_name:
        Free-form label used on the Prometheus gauges; typically
        ``config.task.expert_group_name``.
    """

    def __init__(
        self,
        config: "MinerConfig",
        tokenizer: "PreTrainedTokenizerBase",
        device: torch.device,
        rank: int = 0,
        baseline_model: nn.Module | None = None,
        expert_group_name: str | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.rank = int(rank)
        self.baseline_model = baseline_model
        self.expert_group_name = expert_group_name or str(
            getattr(config.task, "expert_group_name", "unknown_group")
        )

        cfg = config.local_eval
        self.enabled: bool = bool(cfg.enabled)
        self.interval_steps: int = int(cfg.interval_steps)
        self.max_batches: int = int(cfg.max_batches)
        # Defensive copy — caller may mutate the config's list later.
        self.seeds: list[int] = list(cfg.seeds)
        self.use_baseline_compare: bool = bool(cfg.use_baseline_compare) and (
            baseline_model is not None
        )

        # Cache the baseline's val_loss per seed. The baseline never
        # changes during training so evaluating it on every miner
        # invocation would more than double the eval cost.
        self._baseline_loss_by_seed: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def should_run(self, step: int) -> bool:
        """Return True iff a mini-eval is due at ``step``.

        Step gating lives here so the training loop is a one-liner:
        ``if local_evaluator.should_run(step): local_evaluator.run(...)``.
        Step 0 never triggers — the loop hasn't taken any optimizer
        steps yet and the result would be the baseline value, which is
        already covered by the optional baseline-compare path.
        """
        if not self.enabled:
            return False
        if step <= 0:
            return False
        return step % self.interval_steps == 0

    def run(self, model: nn.Module, step: int) -> dict[str, float | int]:
        """Run the mini-eval for every configured seed.

        Iterates ``self.seeds`` and runs one eval per seed. Per-seed
        results are pushed to Prometheus immediately; the returned dict
        aggregates the per-seed metrics (mean val_loss across seeds,
        sum of scored / nan batches, etc.) so the caller can log a
        single row to the metric CSV.

        The caller is responsible for switching ``model`` in and out of
        eval mode via ``try/finally`` — see ``train.py``. Doing it here
        would entangle the lifecycle with the trainer's mode tracking,
        and any exception inside this function still unwinds cleanly
        because :func:`evaluate_model` calls ``model.eval()`` itself.
        """
        if not self.seeds:
            logger.debug("local_eval: no seeds configured; skipping run", step=step)
            # Return the same schema as the normal path so the
            # caller's MetricLogger schema stays rectangular across CSV
            # rows. NaN sentinel for losses; 0 for counts.
            return {
                "local_val_loss": float("nan"),
                "local_delta": float("nan"),
                "local_perplexity": float("inf"),
                "local_scored_batches": 0,
                "local_nan_batches": 0,
                "local_eval_step": int(step),
            }

        per_seed_val_loss: list[float] = []
        per_seed_delta: list[float] = []
        scored_batches_total = 0
        nan_batches_total = 0

        # Track whether the model started in train mode so we can
        # restore it. evaluate_model calls model.eval() internally —
        # without this restoration the training loop would silently
        # keep running with dropout/BN disabled.
        was_training = model.training

        try:
            for seed in self.seeds:
                metrics = self._run_one_seed(model=model, step=step, seed=int(seed))
                val_loss = float(metrics.get("val_loss", float("nan")))
                scored_batches = int(metrics.get("scored_batches", 0))
                nan_batches = int(metrics.get("nan_batches", 0))

                # Skip per-seed publication when we got nothing finite.
                # math.isnan(inf-style) val_loss still publishes — the
                # gauge accepts inf and that's the validator-side signal
                # too. Only None / non-numeric values are filtered.
                if not isinstance(val_loss, (int, float)):
                    continue

                # Publish per-seed val_loss.
                try:
                    MINER_LOCAL_VAL_LOSS.labels(
                        expert_group=self.expert_group_name,
                        rank=str(self.rank),
                        seed=str(int(seed)),
                    ).set(float(val_loss))
                except Exception as e:
                    logger.debug("local_eval: failed to publish val_loss gauge", error=str(e))

                # Optional baseline comparison.
                delta: float | None = None
                if self.use_baseline_compare:
                    baseline_val_loss = self._get_baseline_loss(seed=int(seed), step=step)
                    if baseline_val_loss is not None and math.isfinite(baseline_val_loss):
                        delta = baseline_val_loss - val_loss
                        try:
                            MINER_LOCAL_DELTA.labels(
                                expert_group=self.expert_group_name,
                                rank=str(self.rank),
                                seed=str(int(seed)),
                            ).set(float(delta))
                        except Exception as e:
                            logger.debug(
                                "local_eval: failed to publish delta gauge", error=str(e)
                            )

                per_seed_val_loss.append(float(val_loss))
                if delta is not None:
                    per_seed_delta.append(float(delta))
                scored_batches_total += scored_batches
                nan_batches_total += nan_batches

                logger.info(
                    "local_eval: per-seed result",
                    step=step,
                    seed=int(seed),
                    val_loss=round(val_loss, 4) if math.isfinite(val_loss) else val_loss,
                    delta=round(delta, 4) if (delta is not None and math.isfinite(delta)) else delta,
                    scored_batches=scored_batches,
                    nan_batches=nan_batches,
                )

            # Bump the Prometheus counter once per call (not per seed) so
            # rate(...) gives a clean "evals per minute" signal.
            try:
                MINER_LOCAL_EVAL_STEPS_TOTAL.inc()
            except Exception as e:
                logger.debug("local_eval: failed to bump steps counter", error=str(e))

        finally:
            # Restore train mode regardless of how we exited the loop.
            # evaluate_model unconditionally calls model.eval(); the
            # caller wants the model back the way they handed it over.
            if was_training:
                model.train()

        agg_val_loss = (
            sum(per_seed_val_loss) / len(per_seed_val_loss) if per_seed_val_loss else float("nan")
        )
        agg_delta = (
            sum(per_seed_delta) / len(per_seed_delta) if per_seed_delta else float("nan")
        )
        perplexity = (
            float(math.exp(agg_val_loss))
            if math.isfinite(agg_val_loss) and agg_val_loss < 50.0
            else float("inf")
        )
        return {
            "local_val_loss": float(agg_val_loss),
            "local_delta": float(agg_delta),
            "local_perplexity": float(perplexity),
            "local_scored_batches": int(scored_batches_total),
            "local_nan_batches": int(nan_batches_total),
            "local_eval_step": int(step),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_one_seed(
        self, *, model: nn.Module, step: int, seed: int
    ) -> dict[str, Any]:
        """Build a seed-derived dataloader and run a single eval pass.

        Wraps the call in ``torch.no_grad()`` to guarantee no gradient
        buffers are created, even though :func:`evaluate_model` already
        uses ``no_grad`` internally — defense in depth against a future
        refactor that forgets the inner context manager.
        """
        seed_hex = _seed_to_hex(seed)
        # Build a tiny dataloader using the same machinery the validator
        # uses. We pass ``train=True`` because the validator does the
        # same — production eval pulls from the train split of the
        # configured sources via the fractional-index filter.
        loader = get_dataloader(
            config=self.config,
            tokenizer=self.tokenizer,
            seed=seed_hex,
            rank=self.rank,
            world_size=self.config.task.exp.data.world_size,
            train=True,
        )
        with torch.no_grad():
            return evaluate_model(
                step=int(step),
                model=model,
                eval_dataloader=loader,
                device=self.device,
                max_eval_batches=int(self.max_batches),
                rank=int(self.rank),
            )

    def _get_baseline_loss(self, *, seed: int, step: int) -> float | None:
        """Evaluate the baseline model once per seed and cache the result.

        Returns ``None`` when no baseline model is available — the
        constructor already flips ``use_baseline_compare`` off in that
        case, so this only fires if the model attribute is mutated at
        runtime (which we don't do, but the defensive check costs nothing).
        """
        if self.baseline_model is None:
            return None
        if seed in self._baseline_loss_by_seed:
            return self._baseline_loss_by_seed[seed]

        try:
            metrics = self._run_one_seed(
                model=self.baseline_model, step=step, seed=seed
            )
        except Exception as e:
            logger.warning(
                "local_eval: baseline eval failed; disabling baseline compare for this seed",
                seed=seed,
                error=str(e),
            )
            # Cache an inf sentinel so we don't keep retrying.
            self._baseline_loss_by_seed[seed] = float("inf")
            return float("inf")

        baseline_val_loss = float(metrics.get("val_loss", float("inf")))
        self._baseline_loss_by_seed[seed] = baseline_val_loss
        logger.info(
            "local_eval: cached baseline val_loss",
            seed=seed,
            baseline_val_loss=round(baseline_val_loss, 4)
            if math.isfinite(baseline_val_loss)
            else baseline_val_loss,
        )
        return baseline_val_loss

"""Rolling z-score outlier-batch filter for the miner training loop.

Why this exists
---------------
C4 (and to a lesser degree the math corpus) contains "garbage" rows — broken
HTML, boilerplate-stripped husks, encoding artifacts, code dumps tokenized as
prose. These rows produce loss values multiple sigma above the rolling mean.
Back-propagating through them contaminates gradients with noise that the
optimizer then has to *unlearn* on subsequent in-distribution batches, so
steady-state loss is higher than it would be if we simply skipped the
contaminated batches.

`LossSpikeFilter` maintains a fixed-size FIFO window of recent batch losses
and flags a new batch as "skip" when its loss exceeds ``mu + z * sigma`` of
that window. At the default ``z=4`` this corresponds to ~0.006% of a Gaussian
tail, so well-behaved batches are essentially never flagged — only the
genuine outliers that dominate the right tail of the empirical distribution.

Filter is opt-in via ``MinerConfig.training.batch_filter.enabled``. Default
is OFF so existing miners are unaffected until they explicitly switch it on.
"""

from __future__ import annotations

import statistics
from collections import deque

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


class LossSpikeFilter:
    """Rolling z-score loss-spike filter.

    Each call to :meth:`should_skip` examines a new per-batch loss against
    the mean/std of the last ``window`` accepted losses. Returns ``True``
    when the loss is more than ``z_threshold`` standard deviations above
    the rolling mean, indicating the batch is likely a contaminated sample
    and should be dropped (skip backward + optimizer step).

    During warmup (until at least ``warmup`` losses have been collected)
    nothing is skipped — the filter has no statistics to compare against.

    Skipped batches are *not* appended to the window, so a sustained burst
    of contamination cannot drag the rolling mean up and accidentally
    "normalize" the corrupted distribution.
    """

    def __init__(
        self,
        window: int = 200,
        z_threshold: float = 4.0,
        warmup: int = 50,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if z_threshold <= 0.0:
            raise ValueError(f"z_threshold must be > 0, got {z_threshold}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        if warmup > window:
            raise ValueError(
                f"warmup ({warmup}) must be <= window ({window}); "
                "warmup batches all land in the same window."
            )

        self.window = window
        self.z_threshold = z_threshold
        self.warmup = warmup
        self._losses: deque[float] = deque(maxlen=window)
        self.skipped_count: int = 0
        self.total_seen: int = 0

    def should_skip(self, loss: float) -> bool:
        """Return True iff ``loss`` is a statistical outlier vs. the window.

        Side effects:
        - Increments ``total_seen`` regardless of outcome.
        - Increments ``skipped_count`` and emits a debug log when skipping.
        - Appends ``loss`` to the rolling window only when *not* skipping;
          skipped (outlier) losses are excluded so they cannot poison
          future thresholds.

        Non-finite losses (NaN/inf) are *not* skipped here — the training
        loop already has dedicated non-finite handling further upstream
        (see ``connito/miner/train.py`` consecutive_non_finite_batches
        logic). This filter is strictly for finite outliers.
        """
        self.total_seen += 1

        # Warmup: collect baseline; never skip.
        if len(self._losses) < self.warmup:
            self._losses.append(float(loss))
            return False

        # Need at least 2 samples for a meaningful stdev.
        if len(self._losses) < 2:
            self._losses.append(float(loss))
            return False

        mu = statistics.fmean(self._losses)
        sigma = statistics.pstdev(self._losses)

        # If sigma collapses to 0 (all identical losses, pathological but
        # possible in synthetic tests), don't skip — any difference would
        # otherwise look infinitely many sigma away.
        if sigma <= 0.0:
            self._losses.append(float(loss))
            return False

        threshold = mu + self.z_threshold * sigma
        if loss > threshold:
            self.skipped_count += 1
            logger.debug(
                "LossSpikeFilter: skipping outlier batch",
                loss=float(loss),
                threshold=float(threshold),
                mu=float(mu),
                sigma=float(sigma),
                z_threshold=self.z_threshold,
            )
            return True

        self._losses.append(float(loss))
        return False

    def stats(self) -> dict[str, float | int]:
        """Return a small dict snapshot for telemetry / debug logs."""
        return {
            "window_size": len(self._losses),
            "skipped": self.skipped_count,
            "total_seen": self.total_seen,
            "skip_rate": self.skipped_count / max(self.total_seen, 1),
        }

"""Step (4) helper: submit a round's weights to the chain in the background.

The validator's main loop computes `uid_weights` at the top of each cycle
(after the (3) eval window for the previous round has closed) and hands
them off to this worker via `submit(...)`. The worker schedules the
async chain RPC on the shared `AsyncRunner` (so it runs on the same loop
as the main loop's other lite-subtensor calls and never opens a second
WebSocket).

`submit(...)` is fire-and-forget from the main loop's perspective — the
returned `concurrent.futures.Future` can be ignored. Failures are logged
and the persistent `weights_submitted` flag on the round is set when the
chain accepts the call.
"""

from __future__ import annotations

from concurrent.futures import Future

import bittensor

from connito.shared.app_logging import structlog
from connito.shared.async_runner import AsyncRunner
from connito.shared.chain import submit_weights_async
from connito.validator.round import Round

logger = structlog.get_logger(__name__)


class BackgroundWeightSubmitter:
    """Serial weight submitter driven by a shared `AsyncRunner`."""

    def __init__(
        self,
        *,
        config,
        wallet: bittensor.Wallet,
        async_subtensor: "bittensor.AsyncSubtensor",
        runner: AsyncRunner,
        normalize: bool = True,
        top_k: int | None = None,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> None:
        self.config = config
        self.wallet = wallet
        self.async_subtensor = async_subtensor
        self.runner = runner
        self.normalize = normalize
        self.top_k = top_k
        self.wait_for_inclusion = wait_for_inclusion
        self.wait_for_finalization = wait_for_finalization

    def submit(
        self,
        round_obj: Round,
        uid_weights: dict[int | str, float],
    ) -> Future:
        """Schedule a weight submission for `round_obj`. Returns a Future."""
        coro = self._submit_one(round_obj, uid_weights)
        return self.runner.submit(coro)

    async def _submit_one(
        self,
        round_obj: Round,
        uid_weights: dict[int | str, float],
    ) -> bool:
        logger.info(
            "BackgroundWeightSubmitter: submitting weights",
            round_id=round_obj.round_id,
            top_weights={
                str(k): round(v, 4)
                for k, v in sorted(
                    uid_weights.items(), key=lambda item: item[1], reverse=True,
                )[:5]
            },
        )

        try:
            success = await submit_weights_async(
                config=self.config,
                wallet=self.wallet,
                async_subtensor=self.async_subtensor,
                uid_weights=uid_weights,
                normalize=self.normalize,
                top_k=self.top_k,
                wait_for_inclusion=self.wait_for_inclusion,
                wait_for_finalization=self.wait_for_finalization,
            )
        except Exception as e:
            logger.error(
                "BackgroundWeightSubmitter: submit_weights_async raised",
                round_id=round_obj.round_id, error=str(e), exc_info=True,
            )
            return False

        if success:
            round_obj.weights_submitted = True
            logger.info(
                "BackgroundWeightSubmitter: submission succeeded",
                round_id=round_obj.round_id,
            )
        else:
            logger.warning(
                "BackgroundWeightSubmitter: submission failed",
                round_id=round_obj.round_id,
            )
        return success

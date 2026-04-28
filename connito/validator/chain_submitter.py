"""Background worker for all validator chain submissions.

Owns a dedicated `AsyncSubtensor` (lite endpoint) and the `AsyncRunner`
that drives it. Every submission — `set_commitment` and `set_weights` —
is queued on that single loop, so they share one WebSocket connection
and never race each other.

The validator's main loop stays sync; it only calls:

    chain_submitter.async_commit(status)
    chain_submitter.async_submit_weight(round_obj, uid_weights)
    chain_submitter.async_submit_fallback_weights()

Each method is fire-and-forget — the returned `Future` can be ignored.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future

import bittensor

from connito.shared.app_logging import structlog
from connito.shared.async_runner import AsyncRunner
from connito.shared.chain import (
    MinerChainCommit,
    SignedModelHashChainCommit,
    ValidatorChainCommit,
    _asubmit_fallback_weights,
    acommit_status,
    submit_weights_async,
)
from connito.shared.telemetry import (
    CHAIN_WEIGHT_SET_FAILURE,
    CHAIN_WEIGHT_SET_SUCCESS,
    VALIDATOR_MINER_CHAIN_WEIGHT,
    inc_error,
)
from connito.validator.round import Round

logger = structlog.get_logger(__name__)


class ChainSubmitter:
    """Single background submitter for commits and weight sets."""

    def __init__(
        self,
        config,
        wallet: bittensor.Wallet,
        *,
        normalize: bool = True,
        top_k: int | None = None,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        post_submit_delay_s: float = 14.0,
        runner_name: str = "connito-validator-chain-submitter",
    ) -> None:
        self.config = config
        self.wallet = wallet
        self.normalize = normalize
        self.top_k = top_k
        self.wait_for_inclusion = wait_for_inclusion
        self.wait_for_finalization = wait_for_finalization
        # Held inside the lock after each extrinsic so the wallet's nonce on
        # chain has time to increment before the next caller submits. Without
        # this, back-to-back submissions land in the substrate tx pool with
        # the same nonce and collide ("Priority is too low (1 vs 1)").
        self.post_submit_delay_s = post_submit_delay_s

        self._runner = AsyncRunner(name=runner_name)
        self._async_subtensor = bittensor.AsyncSubtensor(
            network=config.chain.lite_network or config.chain.network
        )
        # Serializes every chain extrinsic this submitter issues.
        # Why: same wallet → substrate tx pool collisions ("Priority is too
        # low", "Invalid Transaction") when set_weights and set_commitment
        # land concurrently. Lazy-init inside the runner loop.
        self._submit_lock: asyncio.Lock | None = None
        init = getattr(self._async_subtensor, "initialize", None)
        if init is not None:
            try:
                self._runner.run(init())
            except Exception as e:
                logger.warning("ChainSubmitter: AsyncSubtensor.initialize() failed", error=str(e))

    def _get_lock(self) -> asyncio.Lock:
        if self._submit_lock is None:
            self._submit_lock = asyncio.Lock()
        return self._submit_lock

    async def _hold_for_nonce_advance(self) -> None:
        """Held inside the lock after a successful chain extrinsic so the
        wallet's on-chain nonce has time to advance before the next caller
        submits — defending against substrate tx-pool collisions."""
        if self.post_submit_delay_s > 0:
            await asyncio.sleep(self.post_submit_delay_s)

    async def _commit_locked(
        self,
        status: ValidatorChainCommit | MinerChainCommit | SignedModelHashChainCommit,
    ):
        async with self._get_lock():
            try:
                return await acommit_status(self.config, self.wallet, self._async_subtensor, status)
            finally:
                await self._hold_for_nonce_advance()

    async def _submit_fallback_locked(self):
        async with self._get_lock():
            try:
                return await _asubmit_fallback_weights(
                    self.config,
                    self.wallet,
                    self._async_subtensor,
                    wait_for_inclusion=self.wait_for_inclusion,
                    wait_for_finalization=self.wait_for_finalization,
                )
            finally:
                await self._hold_for_nonce_advance()

    def async_commit(
        self,
        status: ValidatorChainCommit | MinerChainCommit | SignedModelHashChainCommit,
    ) -> Future:
        return self._runner.submit(self._commit_locked(status))

    def async_submit_weight(
        self,
        round_obj: Round,
        uid_weights: dict[int | str, float],
    ) -> Future:
        nonzero = sum(1 for v in uid_weights.values() if v > 0)
        logger.info(
            "ChainSubmitter: scheduling weight submission",
            round_id=round_obj.round_id,
            total_uids=len(uid_weights),
            nonzero_uids=nonzero,
            top_k=self.top_k,
            normalize=self.normalize,
        )
        coro = self._submit_weight_one(round_obj, uid_weights)
        return self._runner.submit(coro)

    def async_submit_fallback_weights(self) -> Future:
        logger.info(
            "ChainSubmitter: scheduling fallback weight submission",
            wait_for_inclusion=self.wait_for_inclusion,
            wait_for_finalization=self.wait_for_finalization,
        )
        return self._runner.submit(self._submit_fallback_locked())

    def stop(self) -> None:
        self._runner.stop()

    async def _submit_weight_one(
        self,
        round_obj: Round,
        uid_weights: dict[int | str, float],
    ) -> bool:
        nonzero = sum(1 for v in uid_weights.values() if v > 0)
        logger.info(
            "ChainSubmitter: submitting weights to chain (RPC starting)",
            round_id=round_obj.round_id,
            total_uids=len(uid_weights),
            nonzero_uids=nonzero,
            top_k=self.top_k,
            normalize=self.normalize,
            wait_for_inclusion=self.wait_for_inclusion,
            wait_for_finalization=self.wait_for_finalization,
            top_weights={
                str(k): round(v, 4)
                for k, v in sorted(
                    uid_weights.items(), key=lambda item: item[1], reverse=True,
                )[:5]
            },
        )

        try:
            async with self._get_lock():
                try:
                    success = await submit_weights_async(
                        config=self.config,
                        wallet=self.wallet,
                        async_subtensor=self._async_subtensor,
                        uid_weights=uid_weights,
                        normalize=self.normalize,
                        top_k=self.top_k,
                        wait_for_inclusion=self.wait_for_inclusion,
                        wait_for_finalization=self.wait_for_finalization,
                    )
                finally:
                    await self._hold_for_nonce_advance()
        except Exception as e:
            logger.error(
                "ChainSubmitter: submit_weights_async raised",
                round_id=round_obj.round_id, error=str(e), exc_info=True,
            )
            CHAIN_WEIGHT_SET_FAILURE.inc()
            inc_error(component="weight_submit", kind="unknown")
            return False

        if success:
            round_obj.weights_submitted = True
            CHAIN_WEIGHT_SET_SUCCESS.inc()
            # Publish the per-uid weights actually submitted so dashboards can
            # graph the consensus signal alongside the rolling-avg score.
            for uid, weight in uid_weights.items():
                try:
                    VALIDATOR_MINER_CHAIN_WEIGHT.labels(miner_uid=str(uid)).set(float(weight))
                except Exception:
                    pass
            logger.info(
                "ChainSubmitter: submission succeeded",
                round_id=round_obj.round_id,
            )
        else:
            CHAIN_WEIGHT_SET_FAILURE.inc()
            inc_error(component="weight_submit", kind="network")
            logger.warning(
                "ChainSubmitter: submission failed",
                round_id=round_obj.round_id,
            )
        return success

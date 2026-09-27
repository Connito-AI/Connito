"""The baseline a round's submissions trained from — fetched to measure against.

Measurement only. `select_baseline_uid` decides what is published, and nothing
here changes that. This module gives the round a second reference point next
to pretrained: the chain-advertised baseline at freeze, which is exactly the
file those submissions started from (advertised at the previous
ValidatorCommit2, downloaded by miners at Distribute). With its loss on the
round's batches, `baseline_selection_report` can say whether the baseline the
validators went on to publish was better or worse than the one it replaced.

Runs on a daemon thread so the ~2-3 GB download never sits in the main loop,
with its own archive connection because a `Subtensor` is not safe to share
across threads. The fetch is the miner's own (`fetch_model_from_chain_validator`)
so the file, the majority-hash rule and the hash check are the ones miners get.
"""

from __future__ import annotations

from connito.shared.app_logging import structlog
from connito.shared.helper import expert_group_shard_name

logger = structlog.get_logger(__name__)


def fetch_round_incumbent(round_obj, config, expert_group_assignment) -> None:
    """Set `round_obj.incumbent_path`, or leave it None and log why. Never raises."""
    rid = getattr(round_obj, "round_id", None)
    try:
        import bittensor

        from connito.shared.model import fetch_model_from_chain_validator

        group_id = config.task.exp.group_id
        subtensor = bittensor.Subtensor(network=config.chain.network)
        ckpt = fetch_model_from_chain_validator(
            current_model_meta=None,
            config=config,
            subtensor=subtensor,
            wallet=None,
            expert_group_ids=[group_id],
            expert_group_assignment=expert_group_assignment,
        )
        if ckpt is None or ckpt.path is None:
            logger.info("incumbent: none advertised or download failed", round_id=rid)
            return
        path = ckpt.path / expert_group_shard_name(group_id)
        if not path.is_file():
            logger.warning("incumbent: downloaded folder lacks the shard", round_id=rid, path=str(path))
            return
        round_obj.incumbent_path = path
        logger.info(
            "incumbent: fetched",
            round_id=rid,
            path=str(path),
            global_ver=ckpt.global_ver,
            hf_revision=ckpt.hf_revision,
            model_hash=ckpt.model_hash,
        )
    except Exception as e:  # measurement must never disturb the round
        logger.warning("incumbent: fetch failed", round_id=rid, error=str(e), exc_info=True)

"""Publishing the round's result outward.

`evaluator.py` decides how good each miner is and `round.py` decides which one
wins; this module is what leaves the box. Kept separate so the evaluator does
not grow an upload path — it had no HuggingFace dependency before the baseline
feature, and the transport primitives already live in
`connito.shared.hf_distribute`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from connito.shared.app_logging import structlog
from connito.shared.helper import (
    find_submission_for_hotkey,
    get_model_hash,
    load_state_dict_from_path,
)
from connito.shared.hf_distribute import (
    resolve_hf_repo_ids,
    upload_checkpoint_to_hf_subprocess,
)

logger = structlog.get_logger(__name__)


def publish_round_baseline(*, round_obj, config, out: dict | None = None) -> None:
    """Upload the round's best-averaged submission to HF as the next baseline.

    On success `out` receives `path` — the file the Merge window loads to
    advance this validator's own model — plus the coordinates the next
    ValidatorCommit advertises. `path` is written first and separately: the
    local model must still advance in the rounds where we cannot advertise.

    Never raises: a failed publish must not touch scoring, and the caller runs
    this on a daemon thread where an exception would be invisible.
    """
    from connito.validator.round import select_baseline_uid

    rid = getattr(round_obj, "round_id", None)
    try:
        val_losses = dict(getattr(round_obj, "val_losses", None) or {})
        avg = dict(getattr(round_obj, "prior_avg_scores", None) or {})
        # Same call cleanup retains against, so the winner's file is still on
        # disk by construction.
        uid = select_baseline_uid(val_losses, avg)
        if uid is None:
            logger.info("publish_baseline: no scored miners", round_id=rid)
            return
        val_loss = val_losses[uid]
        hotkey = round_obj.uid_to_hotkey.get(uid)
        submission_dir = Path(config.ckpt.miner_submission_path)
        src = find_submission_for_hotkey(
            submission_dir, hotkey, round_obj.submission_block_range,
        ) if hotkey else None
        # A `.pt` submission republished under a `.safetensors` name would
        # download fine and then fail to load.
        if src is None or src.suffix != ".safetensors":
            logger.warning(
                "publish_baseline: winner submission unavailable",
                round_id=rid, uid=uid, path=str(src),
            )
            return

        repo_id, _ = resolve_hf_repo_ids(config.hf)
        # Hardlink, not copy: same filesystem, so staging a ~3 GB shard costs
        # an inode — and the link keeps the bytes alive if the round's prune
        # deletes the source out from under an in-flight upload.
        stage = Path(tempfile.mkdtemp(dir=submission_dir, prefix=".tmp_baseline_"))
        size_bytes, started = src.stat().st_size, time.monotonic()
        try:
            # The name miners already fetch.
            os.link(src, stage / f"model_expgroup_{config.task.exp.group_id}.safetensors")
            revision = upload_checkpoint_to_hf_subprocess(
                ckpt_dir=stage, repo_id=repo_id,
                token_env_var=config.hf.token_env_var,
                commit_message=f"baseline round_id={rid} uid={uid}",
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        # Recorded before the hash so a hashing failure still leaves the local
        # model able to advance — only the advertisement is lost.
        if out is not None:
            out.update(path=str(src), round_id=rid, uid=uid)
        # Prefer the winner's committed hash — we republish its bytes unchanged
        # via hardlink, so nothing needs re-hashing. Fall back to hashing what
        # we uploaded: a miner can be scored and still have no chain commit,
        # and without a hash miners reject the download.
        model_hash = getattr(round_obj.uid_to_chain_checkpoint.get(uid), "model_hash", None)
        hash_source = "chain"
        if not model_hash:
            model_hash = get_model_hash(load_state_dict_from_path(src), hex=True)
            hash_source = "recomputed"
        if out is not None:
            out.update(revision=revision, model_hash=model_hash)
        logger.info(
            "publish_baseline: published", round_id=rid, uid=uid, val_loss=val_loss,
            avg_score=round(avg.get(uid, 0.0), 4),
            repo_id=repo_id, revision=revision, size_bytes=size_bytes,
            model_hash=model_hash[:6] if model_hash else None, hash_source=hash_source,
            elapsed_s=round(time.monotonic() - started, 1),
        )
    except Exception as e:
        logger.warning("publish_baseline: failed", round_id=rid, error=str(e), exc_info=True)

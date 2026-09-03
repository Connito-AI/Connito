"""Publishing the round's result outward.

`evaluator.py` decides how good each miner is and `round.py` decides which one
wins; this module is what leaves the box. Kept separate so the evaluator does
not grow an upload path — it had no HuggingFace dependency before the baseline
feature, and the transport primitives already live in
`connito.shared.hf_distribute`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
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
    resolve_hf_token,
    upload_checkpoint_to_hf_subprocess,
)

logger = structlog.get_logger(__name__)


def _retain_baseline(src: Path, baseline_dir: Path, round_id) -> Path:
    """Hardlink `src` into `baseline_dir` and drop any earlier baseline.

    A sibling of the submission dir, so the same filesystem and the link costs
    an inode rather than a ~3 GB copy — but it does pin one shard the cycle
    prune used to free, which is why the previous one goes first.
    """
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for stale in baseline_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    dest = baseline_dir / f"round_{round_id}{src.suffix}"
    os.link(src, dest)
    return dest


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
        # `src` sits in the submission dir, which the end-of-cycle prune empties
        # at MinerCommit1 — seconds after this runs, and four phases before Merge
        # loads it. Keep a second name for the same bytes outside that dir so
        # unlinking the original frees nothing; that also covers the archive
        # step, which `shutil.move`s the top-k out from under us.
        retained = _retain_baseline(src, submission_dir.parent / "baseline", rid)
        # Recorded before the hash so a hashing failure still leaves the local
        # model able to advance — only the advertisement is lost.
        if out is not None:
            out.update(path=str(retained), round_id=rid, uid=uid)
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


# One upload at a time. A round's podium is ~9 GB and a cycle is ~105 min, so
# overlap should not happen — but if an upload stalls, stacking a second one
# behind it would compete for the same bandwidth and double the staged bytes
# the cycle prune is being held off from.
_podium_lock = threading.Lock()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Streaming sha256 — submissions are ~3 GB, so never read one whole."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _prune_archive_repo(repo_id: str, keep_rounds: int, token: str | None, squash: bool) -> None:
    """Drop `round_*` folders beyond the newest `keep_rounds`, then squash.

    Deleting a folder only rewrites the tree; the LFS bytes stay in history and
    keep counting against the repo. `super_squash_history` is what actually
    reclaims them, at the cost of every prior revision — which is why the
    archive repo must stay separate from the one miners pin.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    rounds: list[tuple[int, str]] = []
    for entry in api.list_repo_tree(repo_id, recursive=False):
        name = Path(entry.path).name
        if not name.startswith("round_"):
            continue
        try:
            rounds.append((int(name.removeprefix("round_")), entry.path))
        except ValueError:
            # Leave anything we cannot identify alone rather than deleting blind.
            logger.warning("publish_podium: unrecognized archive folder", path=entry.path)
    rounds.sort(reverse=True)

    dropped = [path for _, path in rounds[keep_rounds:]]
    for path in dropped:
        api.delete_folder(path_in_repo=path, repo_id=repo_id, commit_message=f"prune {path}")
    if dropped:
        logger.info("publish_podium: pruned archive rounds", repo_id=repo_id, dropped=dropped)
    if dropped and squash:
        api.super_squash_history(repo_id=repo_id, commit_message="reclaim pruned podium bytes")


def publish_round_podium(*, round_obj, config) -> None:
    """Upload this round's top-`top_k` submissions to the archive repo.

    The end-of-cycle prune empties the submission dir at MinerCommit1 regardless
    of rank. `publish_round_baseline` rescues one file, but it picks the best
    *proven* miner by track record — so when this cycle's top scorer is not a
    proven miner, not even rank 1 survives. Miners routinely delete the
    evaluated revision from their own repo, so this is the only chance.

    Ranked by *this round's* scores rather than rolling averages: a miner with a
    strong average may not have submitted this cycle, and its file would not be
    on disk to upload. That also means rank 1 here and the published baseline —
    which selects on the average — legitimately differ some rounds.

    Never raises: runs on a daemon thread where an exception would be invisible,
    and a lost archive must never disturb scoring.
    """
    rid = getattr(round_obj, "round_id", None)
    repo_id = getattr(config.hf, "archive_repo", None)
    if not repo_id:
        return
    if not _podium_lock.acquire(blocking=False):
        logger.warning("publish_podium: previous upload still running", round_id=rid)
        return
    stage: Path | None = None
    try:
        podium = round_obj.top_scored_ranked_this_round(int(config.evaluation.top_k_miners_to_reward))
        if not podium:
            logger.info("publish_podium: nothing scored this round", round_id=rid)
            return

        submission_dir = Path(config.ckpt.miner_submission_path)
        # Hardlink, not copy: same filesystem, so staging ~9 GB costs three
        # inodes — and the links keep the bytes alive when the prune unlinks
        # the originals out from under an upload still in flight.
        stage = Path(tempfile.mkdtemp(dir=submission_dir, prefix=".tmp_podium_"))
        entries: list[dict] = []
        for rank, (uid, score) in enumerate(podium, start=1):
            hotkey = round_obj.uid_to_hotkey.get(uid)
            src = find_submission_for_hotkey(
                submission_dir, hotkey, round_obj.submission_block_range,
            ) if hotkey else None
            if src is None or src.suffix != ".safetensors":
                logger.warning("publish_podium: submission unavailable",
                               round_id=rid, rank=rank, uid=uid, path=str(src))
                continue
            dest = stage / f"rank{rank}_uid{uid}.safetensors"
            os.link(src, dest)
            entries.append({
                "rank": rank, "uid": int(uid), "hotkey": hotkey,
                "score": round(float(score), 6), "filename": dest.name,
                "size_bytes": dest.stat().st_size, "sha256": _sha256_file(dest),
            })

        if not entries:
            logger.warning("publish_podium: no podium file was available", round_id=rid)
            return

        (stage / "manifest.json").write_text(json.dumps({
            "round_id": int(rid),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": entries,
        }, indent=2))

        started = time.monotonic()
        token_env_var = config.hf.token_env_var
        revision = upload_checkpoint_to_hf_subprocess(
            ckpt_dir=stage, repo_id=repo_id, token_env_var=token_env_var,
            commit_message=f"podium round_id={rid}",
            # The default patterns match `model_expgroup_*` only, which would
            # upload an empty commit here.
            allow_patterns=["rank*.safetensors", "manifest.json"],
            path_in_repo=f"round_{rid}",
        )
        logger.info(
            "publish_podium: published", round_id=rid, repo_id=repo_id, revision=revision,
            uids=[e["uid"] for e in entries],
            size_bytes=sum(e["size_bytes"] for e in entries),
            elapsed_s=round(time.monotonic() - started, 1),
        )
        # After the upload, so a retention failure cannot cost us the round.
        _prune_archive_repo(
            repo_id, int(config.hf.archive_keep_rounds),
            resolve_hf_token(token_env_var=token_env_var), bool(config.hf.archive_squash),
        )
    except Exception as e:
        logger.warning("publish_podium: failed", round_id=rid, error=str(e), exc_info=True)
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        _podium_lock.release()

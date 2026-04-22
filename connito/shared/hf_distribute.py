from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


def _resolve_token(token: str | None, env_var: str) -> str | None:
    if token:
        return token
    return os.environ.get(env_var) or None


def upload_checkpoint_to_hf(
    ckpt_dir: Path,
    repo_id: str,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    commit_message: str | None = None,
) -> str:
    """Upload a checkpoint directory to HF and return the commit revision SHA.

    The revision SHA is what validators write to the Bittensor chain so miners
    can pull exactly this snapshot even if the repo advances later. `main` is
    updated to point at the new commit as a side effect, but miners should
    pin to the SHA from the chain, not the branch name.
    """
    resolved_token = _resolve_token(token, token_env_var)
    if resolved_token is None:
        raise RuntimeError(
            f"HF token missing — set {token_env_var} or pass token= explicitly"
        )
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    api = HfApi(token=resolved_token)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except HfHubHTTPError as e:
        # 409 on already-exists races is fine; anything else is real.
        if getattr(e.response, "status_code", None) not in (409,):
            raise

    commit_info = api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        commit_message=commit_message or f"checkpoint upload from {ckpt_dir.name}",
    )
    revision = commit_info.oid
    logger.info(
        "Uploaded checkpoint to HF",
        repo_id=repo_id,
        revision=revision,
        src_dir=str(ckpt_dir),
    )
    return revision


def download_checkpoint_from_hf(
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
) -> Path:
    """Download specific files from a HF repo revision into dest_dir.

    We download only the shards the caller needs (e.g. `model_expgroup_3.pt`
    + `model_shared.pt`) rather than the whole repo, since a validator may
    publish every expert group and a given miner only needs one or two.
    """
    resolved_token = _resolve_token(token, token_env_var)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fname in filenames:
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=fname,
                local_dir=str(dest_dir),
                token=resolved_token,
            )
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"HF repo not found or unauthorized: {repo_id}@{revision}"
        ) from e

    logger.info(
        "Downloaded checkpoint from HF",
        repo_id=repo_id,
        revision=revision,
        files=filenames,
        dest_dir=str(dest_dir),
    )
    return dest_dir

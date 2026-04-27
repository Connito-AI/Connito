from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


def _resolve_token(token: str | None, env_var: str) -> str | None:
    if token:
        return token
    return os.environ.get(env_var) or None


def resolve_hf_token(token: str | None = None, token_env_var: str = "HF_TOKEN") -> str | None:
    return _resolve_token(token, token_env_var)


@lru_cache(maxsize=8)
def _resolve_default_checkpoint_repo_from_token(resolved_token: str, default_repo_name: str) -> str | None:
    api = HfApi(token=resolved_token)
    whoami = api.whoami()
    namespace = str(whoami.get("name") or "").strip()
    if not namespace:
        logger.warning("HF whoami did not return a namespace for default checkpoint repo derivation")
        return None
    return f"{namespace}/{default_repo_name}"


def resolve_default_checkpoint_repo(
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    default_repo_name: str = "co",
) -> str | None:
    repo_name = default_repo_name.strip()
    if not repo_name:
        raise ValueError("default_repo_name cannot be empty")
    if "/" in repo_name:
        raise ValueError("default_repo_name must be a repo name, not '<namespace>/<repo>'")

    resolved_token = _resolve_token(token, token_env_var)
    if resolved_token is None:
        return None

    try:
        return _resolve_default_checkpoint_repo_from_token(resolved_token, repo_name)
    except Exception as exc:
        logger.warning(
            "Failed to derive default HF checkpoint repo from authenticated user",
            default_repo_name=repo_name,
            error=str(exc),
        )
        return None


def get_hf_upload_readiness(
    repo_id: str | None,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
) -> tuple[bool, str]:
    if not repo_id:
        return False, "HF checkpoint repo not configured"
    if _resolve_token(token, token_env_var) is None:
        return False, f"HF token missing — set {token_env_var} or pass token= explicitly"
    return True, "ready"


def resolve_hf_repo_ids(
    hf_cfg,
    max_chain_repo_chars: int | None = None,
) -> tuple[str | None, str | None]:
    """Resolve (upload_repo_id, chain_repo_id) from an HfCfg-shaped object.

    Shared between validator and miner so the derivation rules are identical:
    honor the explicit ``checkpoint_repo`` if set, otherwise derive
    ``{authenticated_hf_user}/{default_repo_name}`` from the HF token.
    ``max_chain_repo_chars`` enforces the role-specific chain payload budget
    (validators have a tighter one than miners).
    """
    derived_repo = resolve_default_checkpoint_repo(
        token_env_var=hf_cfg.token_env_var,
        default_repo_name=hf_cfg.default_repo_name,
    )
    upload_repo_id = hf_cfg.resolve_upload_repo(derived_repo)
    chain_repo_id = hf_cfg.advertised_repo_id(upload_repo_id)

    if chain_repo_id and max_chain_repo_chars is not None and len(chain_repo_id) > max_chain_repo_chars:
        raise ValueError(
            "HF chain repo id is too long for the chain payload budget: "
            f"{len(chain_repo_id)} > {max_chain_repo_chars}"
        )

    return upload_repo_id, chain_repo_id


def upload_checkpoint_to_hf(
    ckpt_dir: Path,
    repo_id: str,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    commit_message: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Upload a checkpoint directory to HF and return a short commit revision.

    The returned value is the first 12 chars of the commit SHA so callers can
    use a shorter immutable revision token downstream. `main` is updated to
    point at the new commit as a side effect, but miners should pin to the
    revision from the chain, not the branch name.
    """
    ready, reason = get_hf_upload_readiness(repo_id=repo_id, token=token, token_env_var=token_env_var)
    if not ready:
        raise RuntimeError(reason)
    resolved_token = _resolve_token(token, token_env_var)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    api = HfApi(token=resolved_token)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except HfHubHTTPError as e:
        # 409 on already-exists races is fine; anything else is real.
        if getattr(e.response, "status_code", None) not in (409,):
            raise

    # Default uploads model_expgroup_N.pt + model_shared.pt (validator publish);
    # miners override to drop model_shared.pt since validators only fetch the
    # expert-group shard from miner submissions.
    commit_info = api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        commit_message=commit_message or f"checkpoint upload from {ckpt_dir.name}",
        allow_patterns=allow_patterns if allow_patterns is not None else ["model*.pt"],
    )
    revision = commit_info.oid[:12]
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

    logger.debug(
        "Downloaded checkpoint from HF",
        repo_id=repo_id,
        revision=revision,
        files=filenames,
        dest_dir=str(dest_dir),
    )
    return dest_dir

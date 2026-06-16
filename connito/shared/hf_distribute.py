from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# Metadata (HEAD) request timeout passed explicitly to `hf_hub_download` so a
# stuck etag lookup can't extend our wall-clock budget. HF's chunk-fetch
# timeout is controlled separately via the `HF_HUB_DOWNLOAD_TIMEOUT` env var
# (default 10s) — we don't override that here so operators can tune it.
_HF_ETAG_TIMEOUT_SEC = 10.0


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

    # Default uploads expert-group shards only (both `.pt` and `.safetensors`
    # during the migration window). `model_shared.*` is intentionally excluded
    # from the default: it's no longer persisted or distributed; every
    # participant reconstructs backbone state from `config.model.model_path`
    # at startup. Callers that need different behavior can override
    # `allow_patterns` explicitly.
    default_allow_patterns = [
        "model_expgroup_*.pt",
        "model_expgroup_*.safetensors",
    ]
    commit_info = api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        commit_message=commit_message or f"checkpoint upload from {ckpt_dir.name}",
        allow_patterns=allow_patterns if allow_patterns is not None else default_allow_patterns,
    )
    revision = commit_info.oid[:12]
    logger.info(
        "Uploaded checkpoint to HF",
        repo_id=repo_id,
        revision=revision,
        src_dir=str(ckpt_dir),
    )
    return revision


def _install_queue_listener_eof_suppressor() -> None:
    """Silence the QueueListener teardown EOFError that HF up/download
    children spill onto the parent's inherited stderr.

    `huggingface_hub`'s xet backend (and a few related HF subsystems)
    start a `logging.handlers.QueueListener` inside the spawned child
    to forward log records from its internal worker pool. When the
    child exits, the workers tear down their multiprocessing queue
    before the listener's `_monitor` thread is joined. `_monitor`
    blocks in `queue.get()`, the queue closes underneath it, and
    `_recv()` raises `EOFError`. Python's default `threading.excepthook`
    then prints a ~20-line traceback to the child's stderr. The child
    inherits stderr from the validator, so every spawn dumps that
    traceback into the validator log — hundreds per cycle, with no
    operational signal in it.

    Suppression is intentionally narrow: only `EOFError` raised in a
    thread whose name ends with `"(_monitor)"` (the QueueListener's
    target function) is swallowed. Anything else flows through
    `threading.__excepthook__` unchanged, so a real crash in any
    other child thread still surfaces in full.
    """
    import threading

    original = threading.__excepthook__

    def _hook(args: threading.ExceptHookArgs) -> None:
        if (
            args.exc_type is EOFError
            and args.thread is not None
            and args.thread.name.endswith("(_monitor)")
        ):
            return
        original(args)

    threading.excepthook = _hook


def _disable_hf_progress_bars_in_child() -> None:
    """Turn off `huggingface_hub`'s tqdm progress bars inside the child.

    The xet backend writes per-chunk tqdm bars to stderr during both
    `hf_hub_download` and `upload_folder`. A single 3.6 GB upload spills
    ~55 lines like `el_expgroup_0.safetensors: 1% | ... / 3.60GB` plus
    `Processing Files (N / M)`, all repainted via ANSI cursor moves. In
    a non-tty log capture those control sequences flatten to a stream
    of duplicated percentage lines that bury real log entries.

    `huggingface_hub.utils.disable_progress_bars()` toggles the in-process
    state — required, not just env-var-based, because HF caches the
    progress-bar state after first import. Called from each child entry
    point so the suppression follows the spawn boundary (parent-side
    progress bars, if any operator wants them, are unaffected).
    """
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()


def _upload_child(
    conn,
    ckpt_dir: str,
    repo_id: str,
    token: str | None,
    token_env_var: str,
    commit_message: str | None,
    allow_patterns: list[str] | None,
) -> None:
    # Entry point in the spawned child. Pickle boundary keeps args
    # simple types (str/list); reconstruct Path inside the child.
    _install_queue_listener_eof_suppressor()
    _disable_hf_progress_bars_in_child()
    try:
        revision = upload_checkpoint_to_hf(
            ckpt_dir=Path(ckpt_dir),
            repo_id=repo_id,
            token=token,
            token_env_var=token_env_var,
            commit_message=commit_message,
            allow_patterns=allow_patterns,
        )
        conn.send(("ok", revision))
    except BaseException as e:
        import traceback as _tb
        conn.send(("err", (type(e).__name__, str(e), _tb.format_exc())))
    finally:
        conn.close()


def upload_checkpoint_to_hf_subprocess(
    ckpt_dir: Path,
    repo_id: str,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    commit_message: str | None = None,
    allow_patterns: list[str] | None = None,
    timeout_sec: float | None = None,
) -> str:
    """Run `upload_checkpoint_to_hf` in a spawned child process.

    The motivation is isolation, not parallelism: when the parent process
    is a long-running validator with a sync-websocket keepalive thread
    (bittensor's substrate client), huggingface_hub's chunked-upload
    workers + TLS encryption hold the GIL long enough to starve the
    keepalive of CPU. The keepalive thread then misses its pong window
    and the websocket library raises a `ConnectionClosedError` whose
    traceback lands on stderr — the parent then has to reconnect and
    rebroadcast its next extrinsic (substrate replies "Invalid
    Transaction / Transaction is outdated" on the stale nonce).

    Spawning a fresh interpreter for the upload removes that GIL pressure
    from the parent entirely. NIC bandwidth and the kernel TCP send queue
    are still shared, so under a true bandwidth wall this is not enough —
    apply OS-level shaping (cgroup net_cls, tc, trickle) to the child
    process if that becomes the limit.

    Use `spawn` rather than `fork`: the validator has CUDA initialized
    in the parent, and a forked child inherits a half-initialized CUDA
    context that segfaults on first use.
    """
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_upload_child,
        args=(
            child_conn,
            str(ckpt_dir),
            repo_id,
            token,
            token_env_var,
            commit_message,
            allow_patterns,
        ),
        name="hf-upload-worker",
    )
    proc.start()
    # Parent doesn't write — close its handle so the child sees EOF
    # cleanly on its own copy of the read end if it ever inspects.
    child_conn.close()

    try:
        proc.join(timeout=timeout_sec)
        if proc.is_alive():
            # Timed out: terminate, then escalate to kill if needed.
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            raise TimeoutError(
                f"HF upload subprocess exceeded {timeout_sec}s; killed"
            )

        # The child writes exactly one message before exiting; if the
        # pipe is empty, the child crashed before reaching either branch
        # of its try/except (e.g. import failure, OOM-kill).
        if not parent_conn.poll():
            raise RuntimeError(
                f"HF upload subprocess exited without sending a result "
                f"(exitcode={proc.exitcode})"
            )

        tag, payload = parent_conn.recv()
        if tag == "ok":
            return payload
        exc_type, exc_msg, exc_tb = payload
        raise RuntimeError(
            f"HF upload subprocess failed: {exc_type}: {exc_msg}\n{exc_tb}"
        )
    finally:
        parent_conn.close()


def download_checkpoint_from_hf(
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
) -> Path:
    """Download specific files from a HF repo revision into dest_dir.

    We download only the shards the caller needs (e.g. `model_expgroup_3.pt`)
    rather than the whole repo, since a validator may publish every expert
    group and a given miner only needs one. `model_shared.*` is no longer
    distributed; backbone state is reconstructed from
    `config.model.model_path` at instantiation.
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
                etag_timeout=_HF_ETAG_TIMEOUT_SEC,
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


def download_checkpoint_from_hf_with_timeout(
    *,
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    timeout_sec: float | None,
) -> Path:
    """Wall-clock-bounded variant of `download_checkpoint_from_hf`.

    `asyncio.wait_for(asyncio.to_thread(...))` cannot interrupt a thread
    blocked inside `huggingface_hub` (it uses requests/httpx under the hood),
    so a hung download leaks a worker into the asyncio default executor pool
    indefinitely. Once enough leak, every subsequent `asyncio.to_thread`
    call queues behind a zombie and times out without ever sending bytes.

    Instead, run each attempt in its own one-shot ThreadPoolExecutor. On
    timeout we `shutdown(wait=False, cancel_futures=True)` so the zombie
    detaches from this caller; the underlying thread continues until its OS
    socket eventually closes, but it no longer blocks the asyncio loop and
    no longer starves unrelated `to_thread` callers.
    """
    if timeout_sec is None:
        return download_checkpoint_from_hf(
            repo_id=repo_id,
            revision=revision,
            filenames=filenames,
            dest_dir=dest_dir,
            token=token,
            token_env_var=token_env_var,
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-dl")
    future = executor.submit(
        download_checkpoint_from_hf,
        repo_id=repo_id,
        revision=revision,
        filenames=filenames,
        dest_dir=dest_dir,
        token=token,
        token_env_var=token_env_var,
    )
    try:
        return future.result(timeout=timeout_sec)
    finally:
        # wait=False: a hung thread won't be joined — we accept the leak so
        # the caller can move on. cancel_futures=True clears anything queued
        # (no-op here since max_workers=1 and we submitted one task).
        executor.shutdown(wait=False, cancel_futures=True)


def _download_child(
    conn,
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: str,
    token: str | None,
    token_env_var: str,
) -> None:
    # Entry point in the spawned child. Pickle boundary keeps args
    # simple types (str/list); reconstruct Path inside the child.
    _install_queue_listener_eof_suppressor()
    _disable_hf_progress_bars_in_child()
    try:
        result = download_checkpoint_from_hf(
            repo_id=repo_id,
            revision=revision,
            filenames=filenames,
            dest_dir=Path(dest_dir),
            token=token,
            token_env_var=token_env_var,
        )
        conn.send(("ok", str(result)))
    except BaseException as e:
        import traceback as _tb
        conn.send(("err", (type(e).__name__, str(e), _tb.format_exc())))
    finally:
        conn.close()


def download_checkpoint_from_hf_subprocess(
    *,
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    timeout_sec: float | None,
) -> Path:
    """Run `download_checkpoint_from_hf` in a spawned child process.

    Replaces `download_checkpoint_from_hf_with_timeout` for callers that
    invoke many downloads from a long-running process. The thread-based
    variant deliberately leaks the worker thread on timeout
    (`executor.shutdown(wait=False, cancel_futures=True)`); over a
    multi-day validator the leaked threads — plus
    `huggingface_hub`'s xet backend which spins its own thread pool
    per call — accumulate enough that the shared HF session pool
    degrades. The observed symptom is a falling success rate even on
    revisions that are present and downloadable from a fresh Python
    interpreter in the same container with the same token — i.e. the
    failure is in the parent process's accumulated state, not in HF
    or the network. See
    `/connito/test/test_download_checkpoint_subprocess.py` for the
    invariants this helper pins.

    A subprocess can be cleanly terminated on timeout, so this code
    path leaves no residual state in the parent regardless of how
    many timeouts occur. Spawn cost is ~200-500ms once the page
    cache is warm, dominated by `huggingface_hub`'s own imports —
    acceptable amortised over a 3-4 GB download and a per-cycle
    download budget measured in tens of attempts.

    Uses `spawn` (not `fork`) — the validator has CUDA initialized in
    the parent, and a forked child inherits a half-initialized CUDA
    context that segfaults on first use.

    Raises:
        TimeoutError: if the child does not finish within ``timeout_sec``.
        RuntimeError: if the child crashed (no result sent) or raised
            any other exception during the download. The original
            exception's type name and stack trace are preserved in the
            message so the caller can diagnose the cause.
    """
    if timeout_sec is None:
        return download_checkpoint_from_hf(
            repo_id=repo_id,
            revision=revision,
            filenames=filenames,
            dest_dir=dest_dir,
            token=token,
            token_env_var=token_env_var,
        )

    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_download_child,
        args=(
            child_conn,
            repo_id,
            revision,
            list(filenames),
            str(dest_dir),
            token,
            token_env_var,
        ),
        name="hf-download-worker",
    )
    proc.start()
    # Parent doesn't write — close its handle so the child sees EOF
    # cleanly on its own copy of the read end if it ever inspects.
    child_conn.close()

    try:
        proc.join(timeout=timeout_sec)
        if proc.is_alive():
            # Timed out: terminate, then escalate to kill if needed.
            # SIGTERM gives the child a chance to flush its HF connection
            # cleanly; SIGKILL is the fallback if it ignores SIGTERM (e.g.
            # stuck inside a C extension that masks signals).
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            raise TimeoutError(
                f"HF download subprocess exceeded {timeout_sec}s; killed"
            )

        # The child writes exactly one message before exiting; if it
        # crashed before either `conn.send` (e.g. import failure, OOM,
        # SIGSEGV inside a C extension) the pipe is either empty or
        # closed. `poll()` returns False for the empty case but True
        # for the closed-without-write case (the OS surfaces EOF as
        # "data available"), and the subsequent `recv()` then raises
        # `EOFError`. Both shapes mean the same thing: the child died
        # silently. Translate both into a single RuntimeError that
        # carries the exit code so an oncall can grep for it.
        try:
            if not parent_conn.poll():
                raise RuntimeError(
                    f"HF download subprocess exited without sending a result "
                    f"(exitcode={proc.exitcode})"
                )
            tag, payload = parent_conn.recv()
        except EOFError as e:
            raise RuntimeError(
                f"HF download subprocess exited without sending a result "
                f"(exitcode={proc.exitcode})"
            ) from e

        if tag == "ok":
            return Path(payload)
        exc_type, exc_msg, exc_tb = payload
        raise RuntimeError(
            f"HF download subprocess failed: {exc_type}: {exc_msg}\n{exc_tb}"
        )
    finally:
        parent_conn.close()

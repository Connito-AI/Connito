"""Pin the invariants of `download_checkpoint_from_hf_subprocess` and
`_download_child` independently.

The subprocess wrapper has two halves that we test separately because
``multiprocessing`` spawn re-imports the worker module from disk in the
child, so an in-test ``monkeypatch`` of the production-side helper
cannot reach the child:

  - the parent's spawn/supervise/timeout-kill loop — driven by a
    synthetic top-level child target defined here, so we don't need
    to hit HuggingFace to exercise it;
  - the actual ``_download_child`` contract — exercised in-process
    (not through ``spawn``) so we can monkey-patch ``download_checkpoint_from_hf``
    and verify the success/error message shapes the parent expects.

This split keeps the production wrapper API clean (no test-only hooks)
while still asserting all behaviour the bug fix depends on:
``timeout_sec=None`` short-circuits, ``ok`` payloads are passed through,
``err`` payloads surface the original exception type+trace, and the
SIGTERM-then-SIGKILL escalation actually terminates the child.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

import connito.shared.hf_distribute as hf_distribute
from connito.shared.hf_distribute import (
    _download_child,
    download_checkpoint_from_hf_subprocess,
)


# ---------------------------------------------------------------------------
# Synthetic subprocess targets — defined at module scope so they're
# picklable for `spawn`. Each represents a child behaviour we want to
# verify the parent supervisor handles correctly.
# ---------------------------------------------------------------------------
def _synthetic_quick_success(conn) -> None:
    conn.send(("ok", "/fake/path"))
    conn.close()


def _synthetic_hang(conn) -> None:
    time.sleep(60)
    conn.send(("ok", "never"))  # unreachable; parent will SIGTERM us first
    conn.close()


def _synthetic_silent_exit(conn) -> None:
    # Don't send anything; simulates a child that crashes before reaching
    # either branch of its try/except (import failure, OOM-kill, …).
    conn.close()


def _spawn_and_supervise(target, timeout_sec: float):
    """Replicates the parent-side supervisor inside
    `download_checkpoint_from_hf_subprocess` against an arbitrary
    target function. We can't share `download_checkpoint_from_hf_subprocess`
    directly here because it pins `_download_child` as the target — and
    `_download_child` requires HuggingFace.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(child_conn,))
    proc.start()
    child_conn.close()
    try:
        proc.join(timeout=timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            raise TimeoutError("synthetic timeout")
        # Match the production wrapper: handle both empty-pipe and
        # closed-without-write (EOFError on recv) as the same failure
        # mode — the child died without producing a result.
        try:
            if not parent_conn.poll():
                raise RuntimeError(f"silent exit (exitcode={proc.exitcode})")
            return parent_conn.recv()
        except EOFError as e:
            raise RuntimeError(f"silent exit (exitcode={proc.exitcode})") from e
    finally:
        parent_conn.close()


# ---------------------------------------------------------------------------
# `timeout_sec=None` short-circuit: stays in-process, no spawn.
# ---------------------------------------------------------------------------
def test_timeout_none_runs_in_process(monkeypatch, tmp_path):
    calls: list[dict] = []

    def _spy(*, repo_id, revision, filenames, dest_dir, token=None, token_env_var="HF_TOKEN"):
        calls.append({"repo": repo_id, "rev": revision})
        return dest_dir

    monkeypatch.setattr(hf_distribute, "download_checkpoint_from_hf", _spy)
    result = download_checkpoint_from_hf_subprocess(
        repo_id="org/repo",
        revision="abc",
        filenames=["x.safetensors"],
        dest_dir=tmp_path,
        timeout_sec=None,
    )
    assert result == tmp_path
    assert calls == [{"repo": "org/repo", "rev": "abc"}]


# ---------------------------------------------------------------------------
# `_download_child` contract — exercised in-process so monkeypatching works.
# Verifies that the parent's `tag, payload = parent_conn.recv()` unpacking
# will succeed for both branches.
# ---------------------------------------------------------------------------
def test_download_child_sends_ok_on_success(monkeypatch, tmp_path):
    def _ok(*, repo_id, revision, filenames, dest_dir, token=None, token_env_var="HF_TOKEN"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in filenames:
            (dest_dir / f).write_bytes(b"x")
        return dest_dir

    monkeypatch.setattr(hf_distribute, "download_checkpoint_from_hf", _ok)
    parent_conn, child_conn = mp.Pipe(duplex=False)
    _download_child(
        child_conn,
        "org/repo",
        "abc",
        ["x.safetensors"],
        str(tmp_path),
        None,
        "HF_TOKEN",
    )
    tag, payload = parent_conn.recv()
    assert tag == "ok"
    assert Path(payload) == tmp_path


def test_download_child_sends_err_with_type_and_trace(monkeypatch, tmp_path):
    def _boom(*, repo_id, revision, filenames, dest_dir, token=None, token_env_var="HF_TOKEN"):
        raise ValueError("simulated HF error")

    monkeypatch.setattr(hf_distribute, "download_checkpoint_from_hf", _boom)
    parent_conn, child_conn = mp.Pipe(duplex=False)
    _download_child(
        child_conn,
        "org/repo",
        "abc",
        ["x.safetensors"],
        str(tmp_path),
        None,
        "HF_TOKEN",
    )
    tag, payload = parent_conn.recv()
    assert tag == "err"
    exc_type, exc_msg, exc_tb = payload
    assert exc_type == "ValueError"
    assert "simulated HF error" in exc_msg
    # The traceback string must include the inner raise line so the parent
    # can produce a useful diagnostic in its `RuntimeError`.
    assert "ValueError" in exc_tb


# ---------------------------------------------------------------------------
# Parent-side supervisor — driven by synthetic spawn targets so we don't
# need network. Validates the contract the bug fix depends on: a hung
# child is actually terminated, not leaked.
# ---------------------------------------------------------------------------
def test_supervisor_returns_child_payload_on_success():
    tag, payload = _spawn_and_supervise(_synthetic_quick_success, timeout_sec=10.0)
    assert tag == "ok"
    assert payload == "/fake/path"


def test_supervisor_raises_runtime_error_on_silent_exit():
    with pytest.raises(RuntimeError, match="silent exit"):
        _spawn_and_supervise(_synthetic_silent_exit, timeout_sec=10.0)


def test_supervisor_kills_hung_child_within_grace():
    """The bug we're fixing: thread-based timeout leaked workers. This
    test pins the property the subprocess variant adds — a wedged child
    is actually torn down, not silently abandoned."""
    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        _spawn_and_supervise(_synthetic_hang, timeout_sec=1.5)
    elapsed = time.perf_counter() - start
    # 1.5s timeout + 5s SIGTERM grace + 5s SIGKILL grace = 11.5s ceiling.
    # In practice SIGTERM is honoured immediately so we expect under 8s.
    assert elapsed < 12, f"subprocess kill took too long: {elapsed:.1f}s"

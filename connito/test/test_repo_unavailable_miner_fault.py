"""Tests for miner-fault attribution of unretrievable HF checkpoints.

Covers the two layers introduced by the fix:

1. `hf_distribute` — repo/revision/file-level HF errors surface as the
   typed `HFRepoUnavailableError` / `HFFileMissingError` (instead of a
   generic `RuntimeError`), and the unauthenticated public probe maps
   HTTP statuses onto True / False / None.

2. `BackgroundDownloadWorker._download_one` — a typed miss that the
   public probe CONFIRMS lands the uid in `validation_failed_uids`
   (score=0 at finalize), while timeouts, generic network errors, and
   unconfirmed misses keep the legacy EMA-preserving `mark_failed` path.
"""

from __future__ import annotations

import asyncio
import io
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

import connito.shared.hf_distribute as hfd
import connito.validator.background_download_worker as bdw
from connito.shared.hf_distribute import (
    HFFileMissingError,
    HFRepoUnavailableError,
    check_file_publicly_fetchable,
    download_checkpoint_from_hf,
)
from connito.validator.background_download_worker import BackgroundDownloadWorker
from connito.validator.round import Round


# ---------------------------------------------------------------------------
# Layer 1: typed errors out of download_checkpoint_from_hf
# ---------------------------------------------------------------------------

def _hub_error(cls, msg):
    """HfHubHTTPError subclasses require a `response` kwarg in recent
    huggingface_hub versions; EntryNotFoundError does not accept one."""
    if cls is EntryNotFoundError:
        return cls(msg)
    import requests
    resp = requests.Response()
    resp.status_code = 404
    return cls(msg, response=resp)


@pytest.mark.parametrize(
    "hub_error_cls, msg, expected",
    [
        (RepositoryNotFoundError, "repo gone", HFRepoUnavailableError),
        (RevisionNotFoundError, "revision gone", HFRepoUnavailableError),
        (EntryNotFoundError, "file gone", HFFileMissingError),
    ],
)
def test_download_raises_typed_errors(monkeypatch, tmp_path, hub_error_cls, msg, expected):
    def _boom(**kwargs):
        raise _hub_error(hub_error_cls, msg)

    monkeypatch.setattr(hfd, "hf_hub_download", _boom)
    with pytest.raises(expected):
        download_checkpoint_from_hf(
            repo_id="acct/repo", revision="abc123",
            filenames=["model_expgroup_0.safetensors"], dest_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Layer 1: public probe status mapping
# ---------------------------------------------------------------------------

def _probe_with_urlopen(monkeypatch, urlopen):
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return check_file_publicly_fetchable(
        repo_id="acct/repo", revision="abc123",
        filename="model_expgroup_0.safetensors",
    )


def test_probe_public_file_is_true(monkeypatch):
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    assert _probe_with_urlopen(monkeypatch, lambda req, timeout: _Resp()) is True


@pytest.mark.parametrize("code", [401, 403, 404])
def test_probe_gone_private_gated_is_false(monkeypatch, code):
    def _raise(req, timeout):
        raise urllib.error.HTTPError("url", code, "nope", {}, io.BytesIO())

    assert _probe_with_urlopen(monkeypatch, _raise) is False


def test_probe_server_error_is_indeterminate(monkeypatch):
    def _raise(req, timeout):
        raise urllib.error.HTTPError("url", 503, "hf down", {}, io.BytesIO())

    assert _probe_with_urlopen(monkeypatch, _raise) is None


def test_probe_network_error_is_indeterminate(monkeypatch):
    def _raise(req, timeout):
        raise OSError("connection refused")

    assert _probe_with_urlopen(monkeypatch, _raise) is None


# ---------------------------------------------------------------------------
# Layer 2: worker classification
# ---------------------------------------------------------------------------

def _make_worker_and_round(tmp_path: Path, *, miner_fault_enabled: bool = True):
    config = SimpleNamespace(
        evaluation=SimpleNamespace(
            per_miner_download_timeout_sec=5,
            repo_unavailable_is_miner_fault=miner_fault_enabled,
        ),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
        ckpt=SimpleNamespace(miner_submission_path=tmp_path / "subs"),
    )
    worker = BackgroundDownloadWorker(
        config=config,
        round_ref=SimpleNamespace(current=None),
        merge_phase_active=threading.Event(),
    )
    round_obj = Round(
        round_id=1,
        seed="test-seed",
        validator_miner_assignment={},
        foreground_uids=(1,),
        background_uids=(),
        uid_to_hotkey={1: "hk1"},
        model_snapshot_cpu={},
        uid_to_chain_checkpoint={
            1: SimpleNamespace(hf_repo_id="acct/repo", hf_revision="abc123"),
        },
    )
    return worker, round_obj


def _run_download(worker, round_obj):
    asyncio.run(worker._download_one(round_obj, uid=1, hotkey="hk1"))


def test_confirmed_unavailable_repo_is_miner_fault(monkeypatch, tmp_path):
    worker, round_obj = _make_worker_and_round(tmp_path)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(HFRepoUnavailableError("repo gone")),
    )
    probe_calls = []
    monkeypatch.setattr(
        bdw, "check_file_publicly_fetchable",
        lambda **kw: probe_calls.append(kw) or False,
    )

    _run_download(worker, round_obj)

    assert 1 in round_obj.validation_failed_uids
    assert probe_calls and probe_calls[0]["repo_id"] == "acct/repo"


def test_confirmed_missing_file_is_miner_fault(monkeypatch, tmp_path):
    worker, round_obj = _make_worker_and_round(tmp_path)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(HFFileMissingError("file gone")),
    )
    monkeypatch.setattr(bdw, "check_file_publicly_fetchable", lambda **kw: False)

    _run_download(worker, round_obj)

    assert 1 in round_obj.validation_failed_uids


def test_probe_says_public_keeps_ema(monkeypatch, tmp_path):
    """Typed miss but the file IS publicly fetchable — validator-side
    problem (e.g. expired token); the miner must keep its prior average."""
    worker, round_obj = _make_worker_and_round(tmp_path)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(HFRepoUnavailableError("401 via token")),
    )
    monkeypatch.setattr(bdw, "check_file_publicly_fetchable", lambda **kw: True)

    _run_download(worker, round_obj)

    assert 1 in round_obj.failed_uids
    assert 1 not in round_obj.validation_failed_uids


def test_probe_indeterminate_keeps_ema(monkeypatch, tmp_path):
    worker, round_obj = _make_worker_and_round(tmp_path)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(HFRepoUnavailableError("repo gone")),
    )
    monkeypatch.setattr(bdw, "check_file_publicly_fetchable", lambda **kw: None)

    _run_download(worker, round_obj)

    assert 1 in round_obj.failed_uids
    assert 1 not in round_obj.validation_failed_uids


def test_generic_network_error_keeps_ema_and_skips_probe(monkeypatch, tmp_path):
    worker, round_obj = _make_worker_and_round(tmp_path)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("connection reset")),
    )
    probe_calls = []
    monkeypatch.setattr(
        bdw, "check_file_publicly_fetchable",
        lambda **kw: probe_calls.append(kw) or False,
    )

    _run_download(worker, round_obj)

    assert 1 in round_obj.failed_uids
    assert 1 not in round_obj.validation_failed_uids
    assert not probe_calls


def test_timeout_keeps_ema(monkeypatch, tmp_path):
    worker, round_obj = _make_worker_and_round(tmp_path)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(TimeoutError("too slow")),
    )

    _run_download(worker, round_obj)

    assert 1 in round_obj.failed_uids
    assert 1 not in round_obj.validation_failed_uids


def test_flag_off_restores_legacy_behavior(monkeypatch, tmp_path):
    worker, round_obj = _make_worker_and_round(tmp_path, miner_fault_enabled=False)
    monkeypatch.setattr(
        bdw, "download_checkpoint_from_hf_subprocess",
        lambda **kw: (_ for _ in ()).throw(HFRepoUnavailableError("repo gone")),
    )
    probe_calls = []
    monkeypatch.setattr(
        bdw, "check_file_publicly_fetchable",
        lambda **kw: probe_calls.append(kw) or False,
    )

    _run_download(worker, round_obj)

    assert 1 in round_obj.failed_uids
    assert 1 not in round_obj.validation_failed_uids
    assert not probe_calls

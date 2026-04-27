"""Regression tests confirming HTTP submission and distribute paths are gone.

These tests pin the post-removal contract:
- ``connito.validator.server`` no longer imports.
- No live source reference to ``submit_model`` or
  ``_download_checkpoint_from_validator_http`` outside this allow-list.
- The miner's commit cycle never issues an HTTP request, whether HF upload
  succeeds or fails.
- ``fetch_model_from_chain_validator`` never issues an HTTP request, whether
  HF download succeeds or fails.
- An HF upload failure inside the commit cycle results in a chain commit
  with no HF coordinates, so the validator's existing missing-submission
  penalty applies.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from connito.shared.checkpoints import ChainCheckpoint, ChainCheckpoints, ModelCheckpoint


CONNITO_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = Path(__file__).resolve()


def _iter_python_sources():
    for path in CONNITO_ROOT.rglob("*.py"):
        if path == TEST_FILE:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def test_validator_server_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("connito.validator.server")


def test_no_remaining_callers_of_http_helpers():
    forbidden = {
        "submit_model(",
        "_download_checkpoint_from_validator_http",
        "_build_validator_checkpoint_url",
    }
    offenders: list[tuple[Path, str]] = []
    for source_path in _iter_python_sources():
        text = source_path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append((source_path, needle))
    assert not offenders, f"unexpected references to deleted HTTP helpers: {offenders}"


def test_miner_commit_cycle_never_issues_http_request_on_hf_success():
    from connito.miner import model_io

    config = SimpleNamespace(
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        ckpt=SimpleNamespace(checkpoint_path=Path("/tmp"), resume_from_ckpt=False),
        chain=SimpleNamespace(netuid=102, hotkey_ss58="miner-hotkey", network="finney"),
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey"))
    subtensor = SimpleNamespace(block=10)
    shared_state = model_io.SharedState()

    fake_checkpoint = MagicMock(spec=ModelCheckpoint)
    fake_checkpoint.path = Path("/tmp/ckpt")
    fake_checkpoint.global_ver = 5
    fake_checkpoint.model_hash = "abcd"
    fake_checkpoint.expert_group = 0

    with patch.object(model_io, "select_best_checkpoint", return_value=fake_checkpoint), \
         patch.object(model_io, "_commit_signed_model_hash") as commit_signed, \
         patch.object(model_io, "_commit_model_hash") as commit_hash, \
         patch.object(model_io, "_upload_checkpoint_to_hf_safe", return_value=("owner/repo", "abcdef0")) as hf_upload, \
         patch.object(model_io, "check_phase_expired"), \
         patch.object(model_io, "wait_till", return_value=SimpleNamespace()), \
         patch("requests.post", side_effect=AssertionError("HTTP submission must not happen")), \
         patch("requests.get", side_effect=AssertionError("HTTP request must not happen")):
        latest = model_io._prepare_checkpoint_for_commit(config, wallet, shared_state)
        model_io._commit_signed_model_hash(config, wallet, subtensor, latest)
        hf_chain_repo_id, hf_revision = model_io._upload_checkpoint_to_hf_safe(config, latest)
        model_io._commit_model_hash(config, wallet, subtensor, latest, hf_chain_repo_id, hf_revision)

    commit_signed.assert_called_once()
    commit_hash.assert_called_once()
    hf_upload.assert_called_once()


def test_miner_commit_cycle_never_issues_http_request_on_hf_failure(caplog):
    from connito.miner import model_io

    config = SimpleNamespace(
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        ckpt=SimpleNamespace(checkpoint_path=Path("/tmp"), resume_from_ckpt=False),
        chain=SimpleNamespace(netuid=102, hotkey_ss58="miner-hotkey", network="finney"),
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey"))
    subtensor = SimpleNamespace(block=10)

    fake_checkpoint = MagicMock(spec=ModelCheckpoint)
    fake_checkpoint.path = Path("/tmp/ckpt")
    fake_checkpoint.global_ver = 5
    fake_checkpoint.model_hash = "abcd"
    fake_checkpoint.expert_group = 0

    with patch.object(model_io, "resolve_hf_repo_ids", side_effect=RuntimeError("hf misconfigured")), \
         patch.object(model_io, "_commit_model_hash") as commit_hash, \
         patch("requests.post", side_effect=AssertionError("HTTP submission must not happen")), \
         patch("requests.get", side_effect=AssertionError("HTTP request must not happen")), \
         caplog.at_level(logging.ERROR):
        hf_chain_repo_id, hf_revision = model_io._upload_checkpoint_to_hf_safe(config, fake_checkpoint)
        model_io._commit_model_hash(config, wallet, subtensor, fake_checkpoint, hf_chain_repo_id, hf_revision)

    assert hf_chain_repo_id is None
    assert hf_revision is None
    # _commit_model_hash receives None HF coords — chain commit goes out without r/rv,
    # which makes the miner missing for this round.
    args, kwargs = commit_hash.call_args
    assert kwargs == {} and args[-2] is None and args[-1] is None


def test_fetch_model_does_not_issue_http_when_hf_succeeds(tmp_path):
    from connito.shared import model as shared_model

    config = SimpleNamespace(
        chain=SimpleNamespace(netuid=102, hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(
            validator_checkpoint_path=tmp_path / "validator_checkpoint",
            checkpoint_topk=2,
        ),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
    )
    config.ckpt.validator_checkpoint_path.mkdir(parents=True)

    chain_checkpoint = ChainCheckpoint(
        uid=7,
        hotkey="validator-hotkey",
        global_ver=42,
        model_hash="abcd",
        signed_model_hash="signed",
        expert_group=0,
        ip="127.0.0.1",
        port=8000,
        hf_repo_id="owner/repo",
        hf_revision="abcdef0",
    )

    def fake_hf_download(**kwargs):
        dest = Path(kwargs["dest_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        for fname in kwargs["filenames"]:
            (dest / fname).write_bytes(b"hf-bytes")

    with patch.object(shared_model, "build_chain_checkpoints_from_previous_phase",
                      return_value=ChainCheckpoints(checkpoints=[chain_checkpoint])), \
         patch.object(shared_model, "download_checkpoint_from_hf", side_effect=fake_hf_download), \
         patch.object(shared_model, "delete_old_checkpoints"), \
         patch.object(ChainCheckpoint, "validate", return_value=True), \
         patch("requests.get", side_effect=AssertionError("HTTP distribute must not happen")), \
         patch("requests.post", side_effect=AssertionError("HTTP submission must not happen")):
        result = shared_model.fetch_model_from_chain_validator(
            current_model_meta=None,
            config=config,
            subtensor=SimpleNamespace(block=123, get_subnet_owner_hotkey=lambda netuid: "owner"),
            wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey")),
            expert_group_ids=[0],
            expert_group_assignment={},
        )

    assert result is chain_checkpoint


def test_fetch_model_does_not_issue_http_when_hf_fails(tmp_path):
    from connito.shared import model as shared_model

    config = SimpleNamespace(
        chain=SimpleNamespace(netuid=102, hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(
            validator_checkpoint_path=tmp_path / "validator_checkpoint",
            checkpoint_topk=2,
        ),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
    )
    config.ckpt.validator_checkpoint_path.mkdir(parents=True)

    chain_checkpoint = ChainCheckpoint(
        uid=7,
        hotkey="validator-hotkey",
        global_ver=42,
        model_hash="abcd",
        signed_model_hash="signed",
        expert_group=0,
        ip="127.0.0.1",
        port=8000,
        hf_repo_id="owner/repo",
        hf_revision="abcdef0",
    )

    with patch.object(shared_model, "build_chain_checkpoints_from_previous_phase",
                      return_value=ChainCheckpoints(checkpoints=[chain_checkpoint])), \
         patch.object(shared_model, "download_checkpoint_from_hf", side_effect=RuntimeError("hf down")), \
         patch.object(shared_model, "delete_old_checkpoints"), \
         patch.object(shared_model.time, "sleep"), \
         patch.object(ChainCheckpoint, "validate", return_value=True), \
         patch("requests.get", side_effect=AssertionError("HTTP distribute must not happen")), \
         patch("requests.post", side_effect=AssertionError("HTTP submission must not happen")):
        result = shared_model.fetch_model_from_chain_validator(
            current_model_meta=None,
            config=config,
            subtensor=SimpleNamespace(block=123, get_subnet_owner_hotkey=lambda netuid: "owner"),
            wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey")),
            expert_group_ids=[0],
            expert_group_assignment={},
        )

    assert result is None


def test_hf_upload_failure_logs_at_error_level(caplog):
    from connito.miner import model_io

    config = SimpleNamespace(
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
    )
    fake_checkpoint = MagicMock(spec=ModelCheckpoint)
    fake_checkpoint.path = Path("/tmp/ckpt")
    fake_checkpoint.global_ver = 1

    with patch.object(model_io, "resolve_hf_repo_ids", return_value=("owner/repo", "owner/repo")), \
         patch.object(model_io, "get_hf_upload_readiness", return_value=(False, "HF token missing")), \
         caplog.at_level(logging.ERROR):
        hf_chain_repo_id, hf_revision = model_io._upload_checkpoint_to_hf_safe(config, fake_checkpoint)

    assert hf_chain_repo_id is None
    assert hf_revision is None
    assert any("HF upload unavailable" in rec.getMessage() for rec in caplog.records)

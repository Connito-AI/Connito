"""Regression tests for the HTTP-server removal.

After this PR there is no validator HTTP server (no `/submit-checkpoint`,
no `/get-checkpoint`, no health endpoints) and the miner submits only via
Hugging Face. These tests guard against accidentally re-introducing the
old code paths.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

CONNITO_ROOT = Path(__file__).resolve().parents[1]


def test_validator_server_module_is_gone():
    """The validator HTTP server module has been deleted."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("connito.validator.server")


def test_shared_client_module_is_gone():
    """`connito.shared.client` (which only existed for the HTTP
    submit/download helpers) has been deleted."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("connito.shared.client")


def test_no_live_references_to_submit_checkpoint_endpoint():
    """No remaining .py file references the deleted endpoint paths in code,
    comments, or log strings. (Test files are allowed to mention them in
    test-name strings; this test itself is the only allowed source.)
    """
    forbidden = ["/submit-checkpoint", "submit_checkpoint", "/get-checkpoint"]
    self_path = Path(__file__).resolve()
    offending: list[tuple[str, str, int]] = []
    for path in CONNITO_ROOT.rglob("*.py"):
        if path.resolve() == self_path:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in forbidden:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    offending.append((str(path), needle, lineno))
    assert offending == [], (
        "Found live references to deleted HTTP endpoints:\n"
        + "\n".join(f"  {p}:{ln}: {needle}" for p, needle, ln in offending)
    )


def test_no_live_references_to_http_submission_helpers():
    """`submit_model`, `download_model`, `_download_checkpoint_from_validator_http`,
    `_build_validator_checkpoint_url` and `submit_worker` all only existed
    to support the HTTP transports. None of them should be referenced after
    the deletion."""
    forbidden = [
        "submit_model",
        "download_model",
        "_download_checkpoint_from_validator_http",
        "_build_validator_checkpoint_url",
        "submit_worker",
        "search_model_submission_destination",
    ]
    self_path = Path(__file__).resolve()
    # Allow this test file to mention the names so the test itself reads
    # naturally; everything else is fair game.
    offending: list[tuple[str, str, int]] = []
    for path in CONNITO_ROOT.rglob("*.py"):
        if path.resolve() == self_path:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in forbidden:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    offending.append((str(path), needle, lineno))
    assert offending == [], (
        "Found live references to deleted HTTP helpers:\n"
        + "\n".join(f"  {p}:{ln}: {needle}" for p, needle, ln in offending)
    )


def test_validator_checkpoint_cfg_has_no_submission_only_fields():
    """The HTTP-only config knobs are gone from `ValidatorCheckpointCfg`.

    Source-level check (avoids importing the heavy datasets/pandas chain
    that pulls in via the config module on some environments).
    """
    src = (CONNITO_ROOT / "shared" / "config.py").read_text(encoding="utf-8")
    for field in ("max_submission_bytes", "max_submission_bytes_per_expert", "submission_concurrency"):
        assert field not in src, f"{field} should have been removed from config.py"


def test_shared_state_has_no_hf_uploaded_gating_flag():
    """`SharedState.latest_checkpoint_hf_uploaded` only existed to gate the
    HTTP submission attempt."""
    src = (CONNITO_ROOT / "miner" / "model_io.py").read_text(encoding="utf-8")
    assert "latest_checkpoint_hf_uploaded" not in src


def test_miner_jobtype_has_no_submit():
    """The SUBMIT job type was the only consumer of the HTTP submission
    queue; it should be gone."""
    src = (CONNITO_ROOT / "miner" / "model_io.py").read_text(encoding="utf-8")
    assert "SUBMIT = auto()" not in src
    assert "submit_queue" not in src


def test_miner_imports_no_longer_pull_in_http_helpers():
    """`connito.miner.model_io` should not import from the deleted
    `connito.shared.client` module or reference HTTP-only helpers."""
    src = (CONNITO_ROOT / "miner" / "model_io.py").read_text(encoding="utf-8")
    assert "from connito.shared.client" not in src
    assert "submit_model" not in src
    assert "search_model_submission_destination" not in src


def test_docker_compose_has_no_server_service():
    """The docker compose file no longer declares a separate `server`
    container."""
    compose = (CONNITO_ROOT / "validator" / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "connito.validator.server" not in compose
    assert "SERVER_GPU_ID" not in compose

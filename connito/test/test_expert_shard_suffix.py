"""Expert-group shards are fetched as `.safetensors`, not `.pt`.

#118 (2026-05-07) made `.safetensors` the save format. Two fetch sites kept
asking for `.pt` and nothing else, so both 404'd against every checkpoint
published since:

  - `_build_download_targets` — the validator's global-model fetch from chain
  - `hydrate_miner_submissions_from_hf` — recovering miner submissions from HF

Verified against a live checkpoint at the time of the fix: it carried
`model_expgroup_3.safetensors` and `model_expgroup_4.safetensors`, and zero
`.pt` files.

`.pt` survives as a *fallback*, tried only when the preferred suffix is
missing, because `download_checkpoint_from_hf` raises on the first absent
entry — asking for both in one call would fail against any repo carrying only
one, which is every repo.

Run with `python -m pytest connito/test/test_expert_shard_suffix.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from connito.shared.hf_distribute import HFFileMissingError
from connito.shared.helper import MINER_CHECKPOINT_SUFFIXES
from connito.shared.model import (
    EXPERT_SHARD_SUFFIX,
    LEGACY_EXPERT_SHARD_SUFFIX,
    _build_download_targets,
    _download_expert_shards,
)


def test_download_targets_default_to_safetensors():
    assert _build_download_targets([4]) == [(4, "model_expgroup_4.safetensors")]


def test_download_targets_accept_an_explicit_legacy_suffix():
    assert _build_download_targets([4], suffix=LEGACY_EXPERT_SHARD_SUFFIX) == [
        (4, "model_expgroup_4.pt")
    ]


def test_shared_sentinel_and_junk_are_still_dropped():
    assert _build_download_targets(["shared", 2, "nonsense"]) == [
        (2, "model_expgroup_2.safetensors")
    ]


# ── the fallback ladder ──────────────────────────────────────────────────────
class _Recorder:
    """Stands in for the download, recording what was asked for."""

    def __init__(self, available: set[str]) -> None:
        self.available = available
        self.requested: list[list[str]] = []

    def __call__(self, *, repo_id, revision, filenames, dest_dir, token_env_var, timeout_sec):
        self.requested.append(list(filenames))
        missing = [f for f in filenames if f not in self.available]
        if missing:
            raise HFFileMissingError(f"missing: {missing}")


def _run(monkeypatch, available: set[str]) -> _Recorder:
    recorder = _Recorder(available)
    monkeypatch.setattr(
        "connito.shared.model._download_checkpoint_from_hf_with_timeout", recorder
    )
    monkeypatch.setattr("connito.shared.model._clear_download_targets", lambda *a, **k: None)
    return recorder


def test_safetensors_is_tried_first_and_stops_there(monkeypatch):
    recorder = _run(monkeypatch, {"model_expgroup_4.safetensors"})

    written = _download_expert_shards(
        repo_id="org/repo",
        revision="abc123",
        expert_group_ids=[4],
        dest_dir=Path("/tmp/unused"),
        token_env_var="HF_TOKEN",
        timeout_sec=None,
    )

    assert written == ["model_expgroup_4.safetensors"]
    assert recorder.requested == [["model_expgroup_4.safetensors"]]  # no wasted `.pt` call


def test_legacy_pt_is_used_when_safetensors_is_absent(monkeypatch):
    recorder = _run(monkeypatch, {"model_expgroup_4.pt"})

    written = _download_expert_shards(
        repo_id="org/repo",
        revision="abc123",
        expert_group_ids=[4],
        dest_dir=Path("/tmp/unused"),
        token_env_var="HF_TOKEN",
        timeout_sec=None,
    )

    assert written == ["model_expgroup_4.pt"]
    assert recorder.requested == [
        ["model_expgroup_4.safetensors"],
        ["model_expgroup_4.pt"],
    ]


def test_neither_suffix_present_raises(monkeypatch):
    _run(monkeypatch, set())

    with pytest.raises(HFFileMissingError):
        _download_expert_shards(
            repo_id="org/repo",
            revision="abc123",
            expert_group_ids=[4],
            dest_dir=Path("/tmp/unused"),
            token_env_var="HF_TOKEN",
            timeout_sec=None,
        )


def test_a_missing_repo_is_not_retried_under_the_other_suffix(monkeypatch):
    """Only a missing *file* is worth a second attempt.

    A dead repo, an auth failure or a timeout means the candidate checkpoint is
    unusable; retrying doubles the latency of every failure for nothing.
    """
    calls: list[list[str]] = []

    def _boom(*, filenames, **kwargs):
        calls.append(list(filenames))
        raise TimeoutError("network")

    monkeypatch.setattr("connito.shared.model._download_checkpoint_from_hf_with_timeout", _boom)
    monkeypatch.setattr("connito.shared.model._clear_download_targets", lambda *a, **k: None)

    with pytest.raises(TimeoutError):
        _download_expert_shards(
            repo_id="org/repo",
            revision="abc123",
            expert_group_ids=[4],
            dest_dir=Path("/tmp/unused"),
            token_env_var="HF_TOKEN",
            timeout_sec=None,
        )

    assert len(calls) == 1


# ── the suffix contract the hydration path depends on ────────────────────────
def test_preferred_suffix_is_first_in_the_shared_ordering():
    """`hydrate_miner_submissions_from_hf` builds its candidates from
    `MINER_CHECKPOINT_SUFFIXES`, so the preferred format has to lead it."""
    assert MINER_CHECKPOINT_SUFFIXES[0] == EXPERT_SHARD_SUFFIX
    assert LEGACY_EXPERT_SHARD_SUFFIX in MINER_CHECKPOINT_SUFFIXES

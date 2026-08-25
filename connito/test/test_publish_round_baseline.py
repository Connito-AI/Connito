"""`publish_round_baseline` uploads the round's best submission as the next baseline.

It runs on a daemon thread off the main validator loop, so the properties worth
pinning are the ones a thread cannot report: that the winner is picked by
`val_loss` (not by rank score), that the staged file carries the exact name
miners fetch, that staging never copies the ~3 GB shard, and that no input —
missing file, deprecated suffix, HF outage — can escape as an exception.

Run with `python -m pytest connito/test/test_publish_round_baseline.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from connito.validator import evaluator

GROUP_ID = 4
BLOCK_RANGE = (500, 600)
PUBLISHED_NAME = f"model_expgroup_{GROUP_ID}.safetensors"


def _config(submission_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=SimpleNamespace(miner_submission_path=submission_dir),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=GROUP_ID)),
    )


def _round(val_losses: dict[int, float], uid_to_hotkey: dict[int, str]) -> SimpleNamespace:
    return SimpleNamespace(
        round_id=9000, val_losses=val_losses, uid_to_hotkey=uid_to_hotkey,
        submission_block_range=BLOCK_RANGE,
    )


def _submission(submission_dir: Path, hotkey: str, block: int, suffix: str = ".safetensors") -> Path:
    submission_dir.mkdir(parents=True, exist_ok=True)
    path = submission_dir / f"uid_1_hotkey_{hotkey}_block_{block}{suffix}"
    path.write_bytes(b"shard")
    return path


@pytest.fixture
def uploads(monkeypatch) -> list[dict]:
    """Record each upload's staged directory contents, since the real call
    deletes the staging dir before returning."""
    calls: list[dict] = []

    def _fake_upload(*, ckpt_dir, repo_id, token_env_var, commit_message):
        staged = sorted(p.name for p in Path(ckpt_dir).iterdir())
        calls.append({
            "repo_id": repo_id, "commit_message": commit_message, "staged": staged,
            "nlink": (Path(ckpt_dir) / staged[0]).stat().st_nlink,
        })
        return "abc123def456"

    monkeypatch.setattr(evaluator, "upload_checkpoint_to_hf_subprocess", _fake_upload)
    monkeypatch.setattr(evaluator, "resolve_hf_repo_ids", lambda cfg: ("owner/co", "owner/co"))
    return calls


def test_lowest_val_loss_wins_and_is_staged_under_the_miner_facing_name(tmp_path, uploads):
    """Winner is `argmin(val_loss)`. That coincides with the top rank score
    today — every miner in a round shares one baseline — but the rule this
    function commits to is the loss, so pin it independently."""
    sub = tmp_path / "miner_submission"
    _submission(sub, "hkA", 550)
    winner = _submission(sub, "hkB", 550)

    evaluator.publish_round_baseline(
        round_obj=_round({1: 3.9, 2: 3.1}, {1: "hkA", 2: "hkB"}),
        config=_config(sub),
    )

    assert len(uploads) == 1
    assert uploads[0]["staged"] == [PUBLISHED_NAME]
    assert uploads[0]["commit_message"] == "baseline round_id=9000 uid=2"
    # Hardlinked, not copied: the source still exists and shared an inode.
    assert uploads[0]["nlink"] == 2
    assert winner.exists()


def test_exact_tie_resolves_on_uid(tmp_path, uploads):
    sub = tmp_path / "miner_submission"
    _submission(sub, "hkA", 550)
    _submission(sub, "hkB", 550)

    evaluator.publish_round_baseline(
        round_obj=_round({7: 3.5, 2: 3.5}, {7: "hkA", 2: "hkB"}),
        config=_config(sub),
    )
    assert uploads[0]["commit_message"].endswith("uid=2")


@pytest.mark.parametrize("scenario", ["no_scores", "file_missing", "out_of_window", "deprecated_pt"])
def test_nothing_is_published_when_the_winner_shard_is_unusable(tmp_path, uploads, scenario):
    sub = tmp_path / "miner_submission"
    sub.mkdir(parents=True)
    val_losses = {} if scenario == "no_scores" else {2: 3.1}
    if scenario == "out_of_window":
        _submission(sub, "hkB", BLOCK_RANGE[1] + 1)
    elif scenario == "deprecated_pt":
        # A `.pt` republished under a `.safetensors` name downloads fine and
        # then fails to load — the one failure mode with no miner-side signal.
        _submission(sub, "hkB", 550, suffix=".pt")

    evaluator.publish_round_baseline(
        round_obj=_round(val_losses, {2: "hkB"}), config=_config(sub),
    )
    assert uploads == []


def test_upload_failure_neither_raises_nor_leaks_the_staging_dir(tmp_path, monkeypatch, uploads):
    """The caller is a daemon thread — an escaping exception is invisible."""
    sub = tmp_path / "miner_submission"
    _submission(sub, "hkB", 550)

    def _boom(**_):
        raise RuntimeError("HF token missing")

    monkeypatch.setattr(evaluator, "upload_checkpoint_to_hf_subprocess", _boom)
    evaluator.publish_round_baseline(
        round_obj=_round({2: 3.1}, {2: "hkB"}), config=_config(sub),
    )

    assert not [p for p in sub.iterdir() if p.name.startswith(".tmp_baseline_")]
    assert not uploads


def test_staging_dir_is_removed_after_a_successful_upload(tmp_path, uploads):
    sub = tmp_path / "miner_submission"
    src = _submission(sub, "hkB", 550)

    evaluator.publish_round_baseline(
        round_obj=_round({2: 3.1}, {2: "hkB"}), config=_config(sub),
    )

    assert uploads
    assert sorted(p.name for p in sub.iterdir()) == [src.name]
    assert os.stat(src).st_nlink == 1

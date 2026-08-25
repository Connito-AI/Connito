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

from connito.shared.helper import parse_dynamic_filename
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


def _round(val_losses: dict[int, float], uid_to_hotkey: dict[int, str],
           prior_avg_scores: dict[int, float] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        round_id=9000, val_losses=val_losses, uid_to_hotkey=uid_to_hotkey,
        submission_block_range=BLOCK_RANGE,
        # Freeze-time snapshot. Flat by default so tests that are not about
        # selection fall through to the val_loss tie-break.
        prior_avg_scores=prior_avg_scores or {uid: 0.0 for uid in val_losses},
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


def test_best_average_wins_even_when_another_miner_beat_it_this_round(tmp_path, uploads):
    """The rule ranks the miner, not the round. uid 1 has the better val_loss
    here, but uid 2 has the stronger track record and is what gets published —
    one lucky round must not become everyone's baseline."""
    sub = tmp_path / "miner_submission"
    _submission(sub, "hkA", 550)
    winner = _submission(sub, "hkB", 550)

    evaluator.publish_round_baseline(
        round_obj=_round({1: 3.1, 2: 3.9}, {1: "hkA", 2: "hkB"},
                         prior_avg_scores={1: 0.25, 2: 1.75}),
        config=_config(sub),
    )

    assert len(uploads) == 1
    assert uploads[0]["staged"] == [PUBLISHED_NAME]
    assert uploads[0]["commit_message"] == "baseline round_id=9000 uid=2"
    # Hardlinked, not copied: the source still exists and shared an inode.
    assert uploads[0]["nlink"] == 2
    assert winner.exists()


def test_val_loss_breaks_an_average_tie(tmp_path, uploads):
    """Averaged rank scores collide constantly — every miner never in a top-3
    sits at exactly 0.0 — so the round's own result has to settle it."""
    sub = tmp_path / "miner_submission"
    _submission(sub, "hkA", 550)
    _submission(sub, "hkB", 550)

    evaluator.publish_round_baseline(
        round_obj=_round({7: 3.9, 2: 3.1}, {7: "hkA", 2: "hkB"},
                         prior_avg_scores={7: 1.5, 2: 1.5}),
        config=_config(sub),
    )
    assert uploads[0]["commit_message"].endswith("uid=2")


def test_uid_breaks_a_total_tie(tmp_path, uploads):
    """Same average and same val_loss: fall to uid so two validators seeing
    identical numbers never publish different baselines."""
    sub = tmp_path / "miner_submission"
    _submission(sub, "hkA", 550)
    _submission(sub, "hkB", 550)

    evaluator.publish_round_baseline(
        round_obj=_round({7: 3.5, 2: 3.5}, {7: "hkA", 2: "hkB"},
                         prior_avg_scores={7: 1.5, 2: 1.5}),
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


def test_cleanup_keeps_the_miner_the_baseline_will_be_published_from(tmp_path):
    """The guarantee the selection rule depends on. `cleanup_non_top_submissions`
    runs after every eval and drops anyone outside this round's top-3; the
    top-average miner can easily place 4th. Unioning `baseline_winner_uid`
    into the keep set is what stops the winner's file being deleted before
    finalize.

    Uses the real `Round` and the real cleanup — a stub would prove nothing
    about the interaction between the two ranking sets.
    """
    from connito.validator.evaluator import cleanup_non_top_submissions
    from connito.validator.round import Round

    sub = tmp_path / "miner_submission"
    uid_to_hotkey = {n: f"hk{n}" for n in range(1, 6)}
    for hk in uid_to_hotkey.values():
        _submission(sub, hk, 550)

    round_obj = Round(
        round_id=9000, seed="s", validator_miner_assignment={},
        foreground_uids=tuple(uid_to_hotkey), background_uids=(),
        uid_to_hotkey=dict(uid_to_hotkey), model_snapshot_cpu={},
        # uid 5 is the proven miner but has the worst round score.
        prior_avg_scores={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 2.0},
    )
    for uid, score in {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.1}.items():
        round_obj.mark_scored(uid, score=score, val_loss=1.0)

    deleted = cleanup_non_top_submissions(
        round_obj=round_obj, submission_dir=sub, top_k=3,
    )

    survivors = {parse_dynamic_filename(p.name)["hotkey"] for p in sub.glob("*.safetensors")}
    assert "hk5" in survivors, f"publish winner was pruned; kept {sorted(survivors)}"
    # Top-3 by round score still retained — merge takes its top-1 from there.
    assert {"hk1", "hk2", "hk3"} <= survivors
    assert "hk4" in {parse_dynamic_filename(n)["hotkey"] for n in deleted}


def test_cleanup_keeps_the_winner_when_a_better_average_went_unevaluated(tmp_path):
    """Retention and selection must rank over the same population.

    Regression: retention once ranked the whole roster while publish ranked
    only the miners actually evaluated. When the roster's best average never
    gets evaluated — routine, since a round rarely scores everyone — the keep
    set protected a miner that could never be selected, and the miner that
    *would* be selected was pruned as an also-ran. Observed on 2026-08-25,
    round 8923578: publish picked uid 24 and found no file.
    """
    from connito.validator.evaluator import cleanup_non_top_submissions
    from connito.validator.round import Round

    sub = tmp_path / "miner_submission"
    uid_to_hotkey = {n: f"hk{n}" for n in range(1, 7)}
    for hk in uid_to_hotkey.values():
        _submission(sub, hk, 550)

    round_obj = Round(
        round_id=9000, seed="s", validator_miner_assignment={},
        foreground_uids=tuple(uid_to_hotkey), background_uids=(),
        uid_to_hotkey=dict(uid_to_hotkey), model_snapshot_cpu={},
        # uid 6 has the best average but is never evaluated; uid 5 is the best
        # average *among those evaluated*, so it is the one publish will pick.
        prior_avg_scores={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 2.0, 6: 3.0},
    )
    for uid, score in {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.1}.items():
        round_obj.mark_scored(uid, score=score, val_loss=1.0)

    cleanup_non_top_submissions(round_obj=round_obj, submission_dir=sub, top_k=3)

    survivors = {parse_dynamic_filename(p.name)["hotkey"] for p in sub.glob("*.safetensors")}
    assert "hk5" in survivors, f"publish winner was pruned; kept {sorted(survivors)}"


def test_cleanup_keeps_the_winner_when_averages_tie(tmp_path):
    """Tied averages must not split retention from selection.

    Rank scores are 2.25/1.5/1.0/0.0, so averages over the window collide
    often. Retention once broke ties on UID alone while selection broke them
    on val_loss first — so a tie handed the keep slot to one miner and the
    publish to another.
    """
    from connito.validator.evaluator import cleanup_non_top_submissions
    from connito.validator.round import Round

    sub = tmp_path / "miner_submission"
    uid_to_hotkey = {n: f"hk{n}" for n in range(1, 6)}
    for hk in uid_to_hotkey.values():
        _submission(sub, hk, 550)

    round_obj = Round(
        round_id=9000, seed="s", validator_miner_assignment={},
        foreground_uids=tuple(uid_to_hotkey), background_uids=(),
        uid_to_hotkey=dict(uid_to_hotkey), model_snapshot_cpu={},
        # uids 4 and 5 tie on average; 5 has the better val_loss, so publish
        # picks 5, while a uid-only tie-break would keep 4.
        prior_avg_scores={1: 0.5, 2: 0.5, 3: 0.5, 4: 1.5, 5: 1.5},
    )
    for uid, score in {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.2, 5: 0.1}.items():
        round_obj.mark_scored(uid, score=score, val_loss=(2.0 if uid == 4 else 1.0))

    cleanup_non_top_submissions(round_obj=round_obj, submission_dir=sub, top_k=3)

    survivors = {parse_dynamic_filename(p.name)["hotkey"] for p in sub.glob("*.safetensors")}
    assert "hk5" in survivors, f"publish winner was pruned; kept {sorted(survivors)}"

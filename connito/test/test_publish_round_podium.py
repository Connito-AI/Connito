"""`publish_round_podium` uploads the round's top-k submissions to the archive repo.

Ranks 2 and 3 exist nowhere but the submission dir, which the end-of-cycle prune
empties at MinerCommit1 — so the properties worth pinning are the ones whose
failure is silent: that ranking follows *this round's* scores rather than rolling
averages, that staging hardlinks (never copies ~9 GB) and survives the prune,
that the staging dir is always cleaned up, and that no failure escapes a daemon
thread where an exception would be invisible.

Run with `python -m pytest connito/test/test_publish_round_podium.py`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from connito.validator import distribute
from connito.validator.round import Round

BLOCK_RANGE = (500, 600)
ROUND_ID = 8988554
CYCLE = 17153
REPO = "owner/co-archive"


def _config(submission_dir: Path, top_k: int = 3, keep_rounds: int = 3,
            archive_repo: str | None = REPO, squash: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=SimpleNamespace(miner_submission_path=submission_dir),
        hf=SimpleNamespace(token_env_var="HF_TOKEN", archive_repo=archive_repo,
                           archive_keep_rounds=keep_rounds, archive_squash=squash),
        evaluation=SimpleNamespace(top_k_miners_to_reward=top_k),
    )


def _round(scores: dict[int, float], uid_to_hotkey: dict[int, str],
           cycle_index: int | None = CYCLE) -> SimpleNamespace:
    """A stand-in round carrying the *real* ranking method.

    Binding `Round.top_scored_ranked_this_round` rather than reimplementing the
    sort is the point: an inverted comparison in the real code has to fail these
    tests, which a hand-rolled ranking in the fixture would hide.
    """
    obj = SimpleNamespace(
        round_id=ROUND_ID, cycle_index=cycle_index, scores=scores,
        uid_to_hotkey=uid_to_hotkey,
        submission_block_range=BLOCK_RANGE, _lock=threading.Lock(),
    )
    obj.top_scored_ranked_this_round = Round.top_scored_ranked_this_round.__get__(obj)
    return obj


def _submission(submission_dir: Path, hotkey: str, block: int = 550,
                body: bytes = b"shard", suffix: str = ".safetensors") -> Path:
    submission_dir.mkdir(parents=True, exist_ok=True)
    path = submission_dir / f"uid_1_hotkey_{hotkey}_block_{block}{suffix}"
    path.write_bytes(body)
    return path


class _FakeApi:
    """Records the retention calls `_prune_archive_repo` makes."""

    folders: list[str] = []
    deleted: list[str] = []
    squashed: list[str] = []

    def __init__(self, token=None):
        pass

    def list_repo_tree(self, repo_id, recursive=False):
        return [SimpleNamespace(path=p) for p in _FakeApi.folders]

    def delete_folder(self, path_in_repo, repo_id, commit_message=None):
        _FakeApi.deleted.append(path_in_repo)

    def super_squash_history(self, repo_id, commit_message=None):
        _FakeApi.squashed.append(repo_id)


@pytest.fixture
def uploads(monkeypatch) -> list[dict]:
    """Record each upload's staged contents — the real call deletes the
    staging dir before returning, so it cannot be inspected afterwards."""
    calls: list[dict] = []

    def _fake_upload(*, ckpt_dir, repo_id, token_env_var, commit_message,
                     allow_patterns=None, path_in_repo=None):
        staged = sorted(p.name for p in Path(ckpt_dir).iterdir())
        shards = [n for n in staged if n.endswith(".safetensors")]
        calls.append({
            "repo_id": repo_id, "commit_message": commit_message,
            "path_in_repo": path_in_repo, "allow_patterns": allow_patterns,
            "staged": staged, "dir": Path(ckpt_dir),
            "manifest": json.loads((Path(ckpt_dir) / "manifest.json").read_text()),
            "nlink": (Path(ckpt_dir) / shards[0]).stat().st_nlink if shards else None,
            "ino": {n: (Path(ckpt_dir) / n).stat().st_ino for n in shards},
            "bytes": {n: (Path(ckpt_dir) / n).read_bytes() for n in shards},
        })
        return "abc123def456"

    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", _fake_upload)
    monkeypatch.setattr(distribute, "resolve_hf_token", lambda **kw: "tok")
    _FakeApi.folders, _FakeApi.deleted, _FakeApi.squashed = [], [], []
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    return calls


def test_the_highest_scorer_this_round_is_rank1(tmp_path, uploads):
    """Regression on an inverted sort: the predecessor ranked ascending on a
    higher-is-better score and filed the *worst* miner as best."""
    sub = tmp_path / "sub"
    for hk in ("aaa", "bbb", "ccc"):
        _submission(sub, hk)
    rnd = _round({1: 0.10, 2: 0.90, 3: 0.50}, {1: "aaa", 2: "bbb", 3: "ccc"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert [f["uid"] for f in uploads[0]["manifest"]["files"]] == [2, 3, 1]
    assert uploads[0]["staged"] == [
        "manifest.json", "rank1_uid2.safetensors",
        "rank2_uid3.safetensors", "rank3_uid1.safetensors",
    ]


def test_a_miner_unscored_this_round_is_never_selected(tmp_path, uploads):
    """The whole reason for ranking on this round rather than the average: a
    miner with a strong history that did not submit has no file to upload."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    _submission(sub, "bbb")
    # uid 9 is absent from `scores` — it was not evaluated this round.
    rnd = _round({1: 0.10, 2: 0.90}, {1: "aaa", 2: "bbb", 9: "zzz"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert [f["uid"] for f in uploads[0]["manifest"]["files"]] == [2, 1]


def test_top_k_bounds_the_podium(tmp_path, uploads):
    sub = tmp_path / "sub"
    for hk in ("aaa", "bbb", "ccc", "ddd"):
        _submission(sub, hk)
    rnd = _round({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}, {1: "aaa", 2: "bbb", 3: "ccc", 4: "ddd"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub, top_k=2))

    assert [f["uid"] for f in uploads[0]["manifest"]["files"]] == [4, 3]


def test_staging_hardlinks_and_survives_the_end_of_cycle_prune(tmp_path, uploads, monkeypatch):
    """The prune runs ~65s after this and empties the submission dir. A copy
    would cost ~9 GB; a move would strand the baseline upload. Only a hardlink
    both costs nothing and outlives the unlink."""
    sub = tmp_path / "sub"
    src = _submission(sub, "aaa", body=b"the winning weights")
    rnd = _round({1: 0.9}, {1: "aaa"})

    src_ino = src.stat().st_ino          # the prune below removes `src`
    real_upload = distribute.upload_checkpoint_to_hf_subprocess

    def _upload_then_prune(**kwargs):
        result = real_upload(**kwargs)
        # Simulate prune_miner_submission_files deleting every submission.
        for p in sub.glob("*.safetensors"):
            p.unlink()
        staged = Path(kwargs["ckpt_dir"]) / "rank1_uid1.safetensors"
        assert staged.read_bytes() == b"the winning weights"
        return result

    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", _upload_then_prune)
    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert uploads[0]["nlink"] == 2                      # original + staged name
    assert uploads[0]["ino"]["rank1_uid1.safetensors"] == src_ino


def test_manifest_hashes_match_the_staged_bytes(tmp_path, uploads):
    sub = tmp_path / "sub"
    _submission(sub, "aaa", body=b"alpha")
    _submission(sub, "bbb", body=b"beta")
    rnd = _round({1: 0.9, 2: 0.5}, {1: "aaa", 2: "bbb"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    call = uploads[0]
    assert call["manifest"]["round_id"] == ROUND_ID
    assert call["manifest"]["cycle_index"] == CYCLE
    for entry in call["manifest"]["files"]:
        raw = call["bytes"][entry["filename"]]
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["size_bytes"] == len(raw)


def test_the_cycle_folder_is_the_upload_path(tmp_path, uploads):
    """Named by cycle so artifacts line up with the dashboard, which counts
    cycles; `round_id` is a block and the two are not interchangeable.
    Retention also needs one folder per round to delete whole rounds."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert uploads[0]["path_in_repo"] == f"cycle_{CYCLE}"
    assert uploads[0]["repo_id"] == REPO
    # The default patterns match `model_expgroup_*` and would commit nothing.
    assert "manifest.json" in uploads[0]["allow_patterns"]


def test_staging_is_removed_after_a_successful_upload(tmp_path, uploads):
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert not uploads[0]["dir"].exists()
    assert list(sub.glob(".tmp_podium_*")) == []


def test_an_upload_failure_neither_raises_nor_leaks_staging(tmp_path, monkeypatch, uploads):
    """This runs on a daemon thread: an escaping exception is invisible, and a
    leaked staging dir pins ~9 GB the prune can no longer free."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})

    def _boom(**kwargs):
        raise RuntimeError("HF is down")

    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", _boom)
    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert list(sub.glob(".tmp_podium_*")) == []


def test_a_missing_file_does_not_abandon_the_other_ranks(tmp_path, uploads):
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    # uid 2 scored but its file is gone; uid 3 submitted a deprecated suffix.
    _submission(sub, "ccc", suffix=".pt")
    rnd = _round({1: 0.5, 2: 0.9, 3: 0.7}, {1: "aaa", 2: "bbb", 3: "ccc"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert [f["uid"] for f in uploads[0]["manifest"]["files"]] == [1]


def test_nothing_is_uploaded_when_no_podium_file_exists(tmp_path, uploads):
    sub = tmp_path / "sub"
    sub.mkdir(parents=True)
    rnd = _round({1: 0.9}, {1: "aaa"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert uploads == []
    assert list(sub.glob(".tmp_podium_*")) == []


def test_an_unscored_round_uploads_nothing(tmp_path, uploads):
    sub = tmp_path / "sub"
    _submission(sub, "aaa")

    distribute.publish_round_podium(round_obj=_round({}, {1: "aaa"}), config=_config(sub))

    assert uploads == []


def test_the_feature_is_off_without_an_archive_repo(tmp_path, uploads):
    """Default config must not publish other miners' weights anywhere."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub, archive_repo=None))

    assert uploads == []


def test_retention_drops_only_rounds_beyond_keep_rounds(tmp_path, uploads):
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})
    _FakeApi.folders = ["cycle_100", "cycle_300", "cycle_200", "cycle_400"]

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub, keep_rounds=3))

    assert _FakeApi.deleted == ["cycle_100"]
    assert _FakeApi.squashed == [REPO]


def test_retention_leaves_unrecognized_folders_alone(tmp_path, uploads):
    """Deleting something we cannot identify is worse than keeping it."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})
    _FakeApi.folders = ["cycle_old", "cycle_100", "cycle_200"]

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub, keep_rounds=1))

    assert _FakeApi.deleted == ["cycle_100"]


def test_no_squash_when_nothing_was_pruned(tmp_path, uploads):
    """Squashing rewrites history; doing it when no bytes were freed spends
    every prior revision for nothing."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})
    _FakeApi.folders = ["cycle_100"]

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub, keep_rounds=3))

    assert _FakeApi.deleted == []
    assert _FakeApi.squashed == []


def test_a_concurrent_publish_is_skipped_rather_than_stacked(tmp_path, uploads, monkeypatch):
    """Two overlapping uploads would compete for bandwidth and double the
    staged bytes the prune is being held off from."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})
    real_upload = distribute.upload_checkpoint_to_hf_subprocess

    def _reenter(**kwargs):
        # Called while the lock is held — the nested call must decline.
        distribute.publish_round_podium(round_obj=rnd, config=_config(sub))
        return real_upload(**kwargs)

    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", _reenter)
    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert len(uploads) == 1


def test_the_lock_is_released_after_a_failure(tmp_path, uploads, monkeypatch):
    """A lock leaked on the error path would silently disable the archive for
    the lifetime of the process."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"})

    working = distribute.upload_checkpoint_to_hf_subprocess

    def _boom(**kwargs):
        raise RuntimeError("HF is down")

    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", _boom)
    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))
    assert uploads == []

    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", working)
    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))
    assert len(uploads) == 1


def test_a_round_without_a_cycle_number_is_not_published(tmp_path, uploads):
    """The folder is named from the cycle. Inventing a fallback name would put
    a second scheme in the repo for the consumer to parse."""
    sub = tmp_path / "sub"
    _submission(sub, "aaa")
    rnd = _round({1: 0.9}, {1: "aaa"}, cycle_index=None)

    distribute.publish_round_podium(round_obj=rnd, config=_config(sub))

    assert uploads == []
    assert list(sub.glob(".tmp_podium_*")) == []

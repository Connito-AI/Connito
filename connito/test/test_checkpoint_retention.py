from __future__ import annotations

from pathlib import Path

from connito.shared.checkpoint_helper import cleanup_temporary_checkpoint_dirs
from connito.shared.checkpoints import prune_miner_submission_files


def test_cleanup_temporary_checkpoint_dirs_removes_all_tmp_entries(tmp_path: Path):
    tmp_a = tmp_path / ".tmp_globalver_1"
    tmp_b = tmp_path / ".tmp_globalver_2"
    keep = tmp_path / "globalver_3"

    tmp_a.mkdir()
    tmp_b.mkdir()
    keep.mkdir()

    removed = cleanup_temporary_checkpoint_dirs(tmp_path)

    assert sorted(Path(path).name for path in removed) == [".tmp_globalver_1", ".tmp_globalver_2"]
    assert not tmp_a.exists()
    assert not tmp_b.exists()
    assert keep.exists()


def test_prune_miner_submission_files_keeps_latest_per_hotkey(tmp_path: Path):
    files = [
        "hotkey_alpha_block_10.pt",
        "hotkey_alpha_block_20.pt",
        "hotkey_beta_block_15.pt",
        "hotkey_beta_block_30.pt",
    ]
    for name in files:
        (tmp_path / name).write_bytes(b"x")

    deleted = prune_miner_submission_files(tmp_path, current_block=45, cycle_length=20, max_age_cycles=1.5)

    assert sorted(deleted) == ["hotkey_alpha_block_10.pt", "hotkey_beta_block_15.pt"]
    assert (tmp_path / "hotkey_alpha_block_20.pt").exists()
    assert (tmp_path / "hotkey_beta_block_30.pt").exists()


def test_prune_miner_submission_files_keeps_recent_history_within_cycle_window(tmp_path: Path):
    files = [
        "hotkey_alpha_block_10.pt",
        "hotkey_beta_block_20.pt",
        "hotkey_gamma_block_30.pt",
    ]
    for name in files:
        (tmp_path / name).write_bytes(b"x")

    deleted = prune_miner_submission_files(tmp_path, current_block=40, cycle_length=20, max_age_cycles=1.5)

    assert sorted(deleted) == ["hotkey_alpha_block_10.pt"]
    assert (tmp_path / "hotkey_beta_block_20.pt").exists()
    assert (tmp_path / "hotkey_gamma_block_30.pt").exists()
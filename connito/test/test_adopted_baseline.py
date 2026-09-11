"""The validator's persisted model state is a pointer to a shard, not a checkpoint.

Everything here runs on a tiny model and safetensors files: the frozen
backbone is `lin.weight`'s absence from every shard, the moving part is the
value each shard writes into it.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from connito.shared.helper import expert_group_shard_name
from connito.validator import adopted_baseline as adopted
from connito.validator.run import _adopt_baseline, _apply_boot_overlay

GID = 4
SHARD = expert_group_shard_name(GID)


class _Tiny(nn.Module):
    def __init__(self, val: float = 0.1) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(val)


def _cfg(tmp_path: Path, **ckpt) -> SimpleNamespace:
    root = tmp_path / "ckpt"
    for d in ("exp_g4", "validator_checkpoint", "miner_submission"):
        (root / d).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        ckpt=SimpleNamespace(**{
            "checkpoint_path": root / "exp_g4",
            "validator_checkpoint_path": root / "validator_checkpoint",
            "miner_submission_path": root / "miner_submission",
            "resume_from_ckpt": True, "use_pretrained_only": False, **ckpt,
        }),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=GID)),
    )


def _shard(path: Path, val: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lin.weight": torch.full((2, 4), val)}, str(path))
    return path


def _pointer(cfg, val: float, *, global_ver: int, round_id: int = 9000) -> adopted.AdoptedBaseline:
    ab = adopted.AdoptedBaseline(
        path=_shard(adopted.baseline_dir(cfg) / f"round_{round_id}.safetensors", val),
        round_id=round_id, global_ver=global_ver, model_hash="cd" * 32, revision="abc123",
    )
    adopted.persist(cfg, ab)
    return ab


def _weight(model) -> torch.Tensor:
    return model.lin.weight.detach()


# --- the pointer itself --------------------------------------------------------

def test_pointer_round_trips(tmp_path):
    cfg = _cfg(tmp_path)
    ab = _pointer(cfg, 0.5, global_ver=9384)

    assert adopted.load(cfg) == ab
    assert not list(cfg.ckpt.checkpoint_path.glob("*.tmp")), "write must be atomic"


def test_an_unreadable_pointer_reads_as_none(tmp_path):
    cfg = _cfg(tmp_path)
    adopted.pointer_path(cfg).write_text("{not json")
    assert adopted.load(cfg) is None


def test_the_pointer_ranks_like_a_checkpoint(tmp_path):
    """Boot compares it against downloaded checkpoints by `global_ver`."""
    ab = adopted.AdoptedBaseline(path=Path("/x"), round_id=1, global_ver=7, model_hash="ab" * 32)
    ck = ab.as_checkpoint()
    assert (ck.global_ver, ck.model_hash, ck.path) == (7, "ab" * 32, Path("/x"))


# --- boot ---------------------------------------------------------------------

def test_boot_overlays_the_pointer(tmp_path):
    cfg = _cfg(tmp_path)
    own = _pointer(cfg, 0.7, global_ver=9384)
    model = _Tiny(0.1)

    source, chosen = _apply_boot_overlay(cfg, model, own)

    assert source == "pointer"
    assert chosen.global_ver == 9384
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.7))


def test_boot_migrates_a_legacy_globalver_once(tmp_path):
    """An upgraded validator must not reset to pretrained, and the legacy dir
    must survive for a rollback to the previous image."""
    cfg = _cfg(tmp_path)
    legacy = _shard(cfg.ckpt.checkpoint_path / "globalver_9384" / SHARD, 0.6)
    assert adopted.load(cfg) is None

    own = adopted.migrate_from_legacy_globalver(cfg)

    assert own is not None and own.global_ver == 9384
    assert own.path == adopted.baseline_dir(cfg) / "round_9384.safetensors"
    assert os.stat(own.path).st_ino == os.stat(legacy).st_ino, "a hardlink, not a copy"
    assert legacy.exists(), "legacy dir is left for rollback"
    assert adopted.load(cfg) == own, "pointer written, so the next boot skips migration"

    model = _Tiny(0.1)
    source, _ = _apply_boot_overlay(cfg, model, own, migrated=True)
    assert source == "legacy_globalver"
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.6))


def test_boot_prefers_the_higher_global_ver(tmp_path):
    """Same precedence as before between own state and the downloaded fleet cache."""
    cfg = _cfg(tmp_path)
    _shard(cfg.ckpt.validator_checkpoint_path / "uid_1_hotkey_hk_globalver_9500" / SHARD, 0.9)

    own = _pointer(cfg, 0.5, global_ver=9384)
    model = _Tiny(0.1)
    source, chosen = _apply_boot_overlay(cfg, model, own)
    assert source == "downloaded"
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.9))
    # The pointer follows: the next freeze must pin what the model holds, not
    # the older own shard — otherwise the first round is scored against a
    # base the miners never trained from.
    after = adopted.load(cfg)
    assert after.global_ver == 9500 and chosen.global_ver == 9500
    assert after.path == adopted.baseline_dir(cfg) / "round_9500.safetensors"
    assert after.path.is_file() and after.model_hash

    own = _pointer(cfg, 0.5, global_ver=9600)
    model = _Tiny(0.1)
    source, _ = _apply_boot_overlay(cfg, model, own)
    assert source == "pointer"
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.5))


def test_boot_with_nothing_is_pretrained(tmp_path):
    cfg = _cfg(tmp_path)
    model = _Tiny(0.1)

    source, chosen = _apply_boot_overlay(cfg, model, None)

    assert (source, chosen) == ("pretrained", None)
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.1))


def test_a_missing_pointer_target_falls_back_and_does_not_crash(tmp_path):
    cfg = _cfg(tmp_path)
    own = _pointer(cfg, 0.5, global_ver=9600)
    own.path.unlink()
    _shard(cfg.ckpt.validator_checkpoint_path / "uid_1_hotkey_hk_globalver_9500" / SHARD, 0.9)
    model = _Tiny(0.1)

    source, _ = _apply_boot_overlay(cfg, model, own)

    assert source == "downloaded"
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.9))


def test_use_pretrained_only_skips_the_overlay(tmp_path):
    cfg = _cfg(tmp_path, use_pretrained_only=True)
    own = _pointer(cfg, 0.7, global_ver=9384)
    model = _Tiny(0.1)

    assert _apply_boot_overlay(cfg, model, own)[0] == "pretrained"
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.1))


# --- merge --------------------------------------------------------------------

def test_merge_writes_the_pointer_not_a_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    shard = _shard(adopted.baseline_dir(cfg) / "round_9001.safetensors", 0.8)
    model = _Tiny(0.1)
    ref = {"path": str(shard), "round_id": 9001, "uid": 7, "model_hash": "ef" * 32, "revision": "r1"}

    assert _adopt_baseline(cfg, model, ref, global_ver=9384) == shard

    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.8))
    ab = adopted.load(cfg)
    assert (ab.path, ab.round_id, ab.global_ver, ab.model_hash, ab.revision) == (
        shard, 9001, 9384, "ef" * 32, "r1",
    )
    assert not list(cfg.ckpt.checkpoint_path.glob("globalver_*"))


def test_merge_hashes_the_file_when_the_upload_has_not_landed(tmp_path):
    """`model_hash` arrives with the publish thread's second write; the switch
    can beat it."""
    from connito.shared.helper import get_model_hash, load_state_dict_from_path

    cfg = _cfg(tmp_path)
    shard = _shard(adopted.baseline_dir(cfg) / "round_9001.safetensors", 0.8)

    _adopt_baseline(cfg, _Tiny(0.1), {"path": str(shard), "round_id": 9001}, global_ver=9384)

    expected = get_model_hash(load_state_dict_from_path(str(shard)), hex=True)
    assert adopted.load(cfg).model_hash == expected


def test_a_failed_adopt_leaves_the_previous_pointer(tmp_path):
    cfg = _cfg(tmp_path)
    before = _pointer(cfg, 0.5, global_ver=9384)
    foreign = adopted.baseline_dir(cfg) / "round_9001.safetensors"
    save_file({"nothing.here": torch.zeros(1)}, str(foreign))
    model = _Tiny(0.1)

    assert _adopt_baseline(cfg, model, {"path": str(foreign), "round_id": 9001}, global_ver=9500) is None

    assert adopted.load(cfg) == before
    torch.testing.assert_close(_weight(model), torch.full((2, 4), 0.1))

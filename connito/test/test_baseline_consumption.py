"""The chain-side half of the baseline handover.

`publish_round_baseline` (PR #230) uploads the round's best submission; this
covers what has to be true for anyone to *use* it. Three failure modes here are
silent — the validator logs nothing unusual and miners simply never converge:

- asking HF for a filename nobody writes,
- letting each miner pick a different validator's baseline,
- advertising one validator's revision beside another model's hash.

Run with `python -m pytest connito/test/test_baseline_consumption.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from connito.shared.checkpoints import ChainCheckpoint, ChainCheckpoints
from connito.shared.helper import (
    expert_group_shard_name,
    get_model_hash,
    load_state_dict_from_path,
)
from connito.shared.model import _build_download_targets
from connito.validator import distribute

GROUP_ID = 4


def _ckpt(uid: int, model_hash: str, stake: float, global_ver: int = 8338208) -> ChainCheckpoint:
    return ChainCheckpoint(
        signed_model_hash="ab" * 32, model_hash=model_hash, global_ver=global_ver,
        expert_group=GROUP_ID, uid=uid, ip="0.0.0.0", port=8000,
        hotkey=f"5Test{uid:03d}", stake=stake,
    )


# --- the download name -------------------------------------------------------

def test_download_asks_for_the_name_the_writer_produces():
    """The outage this fixes: the writer moved to `.safetensors` while the
    download path kept asking for `.pt`, and `download_checkpoint_from_hf` has
    no fallback — so every fetch raised HFFileMissingError. Both sides now
    derive the name from one helper; assert they agree rather than that either
    equals a literal, so they cannot drift apart again."""
    (_, downloaded), = _build_download_targets([GROUP_ID])
    assert downloaded == expert_group_shard_name(GROUP_ID)
    assert downloaded.endswith(".safetensors")


def test_shared_sentinel_is_still_a_no_op():
    assert _build_download_targets(["shared"]) == []


# --- stake decides which baseline everyone uses ------------------------------

def test_miners_follow_the_highest_stake_validator():
    """Each validator publishes its own pick, so their hashes differ. Without
    stake weighting a miner takes whichever entry came first and the subnet
    forks into incompatible models."""
    cks = ChainCheckpoints(checkpoints=[
        _ckpt(1, "aa" * 32, stake=31.4),
        _ckpt(2, "bb" * 32, stake=58734.8),
    ])
    kept = cks.filter_checkpoints(for_role="miner")
    assert [c.uid for c in kept.checkpoints] == [2]


def test_validators_and_miners_agree_on_the_winner():
    """Both roles must resolve to the same baseline, or validators evaluate
    against a model the miners never trained on."""
    pair = [_ckpt(1, "aa" * 32, stake=31.4), _ckpt(2, "bb" * 32, stake=58734.8)]
    as_miner = ChainCheckpoints(checkpoints=list(pair)).filter_checkpoints(for_role="miner")
    as_validator = ChainCheckpoints(checkpoints=list(pair)).filter_checkpoints(for_role="validator")
    assert [c.uid for c in as_miner.checkpoints] == [c.uid for c in as_validator.checkpoints]


def test_miner_version_gate_skip_survives_stake_selection():
    """Guard against over-deleting: routing miners through the stake block must
    not drag them back through the version-range gate, which drops a commit
    landing 1-2 blocks outside the window."""
    inside, above = 8338208, 8338209
    cks = ChainCheckpoints(checkpoints=[
        _ckpt(1, "aa" * 32, stake=1.0, global_ver=inside),
        _ckpt(2, "aa" * 32, stake=1.0, global_ver=above),
    ])
    kept = cks.filter_checkpoints(
        for_role="miner", min_allowed_version=inside - 100, max_allowed_version=inside,
    )
    assert sorted(c.uid for c in kept.checkpoints) == [1, 2]


# --- revision and hash must travel together ----------------------------------

WINNER_HASH = "cd" * 32


def _round_with_winner(uid: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        round_id=9000, val_losses={1: 3.9, uid: 3.1},
        uid_to_hotkey={1: "hkA", uid: "hkB"},
        uid_to_chain_checkpoint={uid: _ckpt(uid, WINNER_HASH, stake=1.0)},
        submission_block_range=(500, 600),
        prior_avg_scores={1: 0.0, uid: 1.5},
    )


def _config(submission_dir) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=SimpleNamespace(miner_submission_path=submission_dir),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=GROUP_ID)),
    )


@pytest.fixture
def stub_upload(monkeypatch):
    monkeypatch.setattr(distribute, "resolve_hf_repo_ids", lambda cfg: ("owner/up", "owner/up"))
    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess",
                        lambda **kw: "abc123def456")


def test_publish_reports_revision_and_hash_together(tmp_path, stub_upload):
    """Miners verify downloaded bytes against `model_hash`. Advertising the
    baseline's revision beside the merged model's hash makes every miner reject
    the fetch, so the commit must take both from the same place."""
    sub = tmp_path / "miner_submission"
    sub.mkdir()
    (sub / "uid_1_hotkey_hkB_block_550.safetensors").write_bytes(b"shard")

    out: dict = {}
    distribute.publish_round_baseline(round_obj=_round_with_winner(), config=_config(sub), out=out)

    assert out["revision"] == "abc123def456"
    assert out["model_hash"] == WINNER_HASH
    assert out["uid"] == 2


def test_failed_publish_leaves_nothing_to_advertise(tmp_path, monkeypatch):
    """An empty holder is the signal to commit no HF coordinates. If a failure
    left a stale or partial entry the validator would advertise a revision that
    does not exist."""
    sub = tmp_path / "miner_submission"
    sub.mkdir()
    (sub / "uid_1_hotkey_hkB_block_550.safetensors").write_bytes(b"shard")

    def _boom(**_):
        raise RuntimeError("HF unreachable")

    monkeypatch.setattr(distribute, "resolve_hf_repo_ids", lambda cfg: ("owner/up", "owner/up"))
    monkeypatch.setattr(distribute, "upload_checkpoint_to_hf_subprocess", _boom)

    out: dict = {}
    distribute.publish_round_baseline(round_obj=_round_with_winner(), config=_config(sub), out=out)
    assert out == {}


def test_winner_without_a_chain_commit_is_advertised_with_a_recomputed_hash(tmp_path, stub_upload):
    """A miner can be scored this round and still carry no valid chain commit —
    observed live, with every uid flagged `invalid chain checkpoints` at freeze.
    Relying on the miner's committed hash stranded the baseline permanently: we
    paid for the upload and then advertised nothing.

    We republish the bytes unchanged, so hashing what we uploaded is not just a
    fallback, it is the more authoritative answer.
    """
    sub = tmp_path / "miner_submission"
    sub.mkdir()
    shard = sub / "uid_1_hotkey_hkB_block_550.safetensors"
    save_file({"w": torch.zeros(4)}, str(shard))

    round_obj = _round_with_winner()
    round_obj.uid_to_chain_checkpoint = {}

    out: dict = {}
    distribute.publish_round_baseline(round_obj=round_obj, config=_config(sub), out=out)

    assert out["revision"] == "abc123def456"
    # Must match what a verifier computes, or every miner rejects the download.
    assert out["model_hash"] == get_model_hash(load_state_dict_from_path(shard), hex=True)


def test_publish_records_the_path_even_when_the_hash_is_unusable(tmp_path, stub_upload,
                                                                 monkeypatch):
    """`path` drives this validator's own model forward in the Merge window, so
    it must not be coupled to advertisability. Losing the hash costs the
    advertisement; it must not also freeze the local model."""
    sub = tmp_path / "miner_submission"
    sub.mkdir()
    (sub / "uid_1_hotkey_hkB_block_550.safetensors").write_bytes(b"not-a-safetensors")

    round_obj = _round_with_winner()
    round_obj.uid_to_chain_checkpoint = {}   # forces the recompute, which will raise

    out: dict = {}
    distribute.publish_round_baseline(round_obj=round_obj, config=_config(sub), out=out)

    assert out["uid"] == 2
    assert out["path"].endswith("uid_1_hotkey_hkB_block_550.safetensors")
    assert "model_hash" not in out


# --- a restart must not throw the merge away ---------------------------------

@pytest.mark.parametrize("load_global,expect_consulted", [(True, True), (False, False)])
def test_boot_consults_on_disk_state_only_when_asked(monkeypatch, tmp_path, load_global, expect_consulted):
    """`load_global_checkpoint=False` made the validator boot from pretrained
    weights and silently discard the merge. The comment justifying it claimed
    `reload_model_inplace` would recover the state, but that call is gated on
    `not _participated_in_merge`, which never holds after a restart. Measured
    6/6 against production restarts: the model only advanced when no kill
    intervened. Pin both directions so the flag cannot quietly flip back."""
    from connito.shared import model as shared_model

    consulted: list[bool] = []

    class _Model:
        def to(self, **kw):
            return self

        def gradient_checkpointing_enable(self):
            pass

    monkeypatch.setattr(shared_model, "get_base_model", lambda *a, **k: _Model())
    monkeypatch.setattr(shared_model, "select_best_checkpoint",
                        lambda **kw: consulted.append(True))

    cfg = SimpleNamespace(
        ckpt=SimpleNamespace(
            resume_from_ckpt=True, use_pretrained_only=False,
            validator_checkpoint_path=tmp_path, checkpoint_path=tmp_path,
        ),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=GROUP_ID), helper_group_id=2),
        model=SimpleNamespace(device="cpu", precision="fp16-mixed"),
    )
    shared_model.get_model_from_checkpoint(
        rank=0, config=cfg, expert_manager=SimpleNamespace(), partial=True,
        load_global_checkpoint=load_global,
    )
    assert bool(consulted) is expect_consulted

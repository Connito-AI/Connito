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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from connito.shared.checkpoints import (
    ChainCheckpoint,
    ChainCheckpoints,
    prune_miner_submission_files,
)
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
    """`for_role` names *whose* commits these are, not who is reading. A miner
    fetching the official model reads validator commits, so this is the path
    that must resolve to the highest-stake validator — each validator now
    publishes its own pick, and without stake weighting miners scatter across
    incompatible models."""
    cks = ChainCheckpoints(checkpoints=[
        _ckpt(1, "aa" * 32, stake=31.4),
        _ckpt(2, "bb" * 32, stake=58734.8),
    ])
    kept = cks.filter_checkpoints(for_role="validator")
    assert [c.uid for c in kept.checkpoints] == [2]


def test_miner_commits_are_never_collapsed_to_one_hash():
    """The regression that cost a day on the test validator.

    Reading *miner* commits must skip the stake-weighted majority vote
    entirely. Every miner trains its own model, so distinct hashes are the
    normal case, not a disagreement to resolve — running the vote here keeps
    only whichever arbitrary group shares the top-staked hash and silently
    deletes everyone else.

    Measured with the vote wrongly enabled: `competing_hashes=244`,
    `majority_stake=178.07`, and the validator scored 1 miner per round
    instead of ~90. Nothing errored; the scores simply stopped appearing.
    """
    cks = ChainCheckpoints(checkpoints=[
        _ckpt(uid, f"{uid:02x}" * 32, stake=float(uid) * 10.0)
        for uid in range(1, 6)
    ])
    kept = cks.filter_checkpoints(for_role="miner")
    assert sorted(c.uid for c in kept.checkpoints) == [1, 2, 3, 4, 5]


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
    """The winner comes from `val_losses` (miners we evaluated) while the hash
    comes from `uid_to_chain_checkpoint` (miners with a chain checkpoint at
    freeze). Nothing reconciles those two sets, and a restart pulls them apart:
    the round resumes from its journal with miners already scored, while the
    checkpoint map is rebuilt from current chain state.

    Observed live — uid 238 won its round, published, and advertised nothing.

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
    # The retained link, not the submission-dir name — that one is deleted by
    # the end-of-cycle prune before Merge reads it.
    assert out["path"].endswith("baseline/round_9000.safetensors")
    assert "model_hash" not in out


# --- the baseline file must outlive the cycle that produced it ---------------

def test_baseline_survives_the_end_of_cycle_prune(tmp_path, stub_upload):
    """The crash this prevents, in the order production runs it.

    `publish_round_baseline` records the winner at MinerCommit1; the
    end-of-cycle prune empties the submission dir seconds later; Merge loads
    the recorded path four phases after that. Observed live on 2026-08-28:
    round 8939822 published at 01:30:48, the prune deleted the winner's shard
    at 01:31:37, and Merge died with FileNotFoundError at 01:49:50.

    Recording a path into a directory that gets emptied is the bug, so this
    asserts the original name really is gone — otherwise the test would pass
    for the wrong reason.
    """
    sub = tmp_path / "miner_submission"
    sub.mkdir()
    shard = sub / "uid_1_hotkey_hkB_block_550.safetensors"
    save_file({"w": torch.zeros(4)}, str(shard))

    out: dict = {}
    distribute.publish_round_baseline(round_obj=_round_with_winner(), config=_config(sub), out=out)

    prune_miner_submission_files(sub, current_block=600, cycle_length=448, max_age_cycles=0)
    assert not shard.exists()

    assert set(load_state_dict_from_path(Path(out["path"]))) == {"w"}


def test_only_the_newest_baseline_is_pinned(tmp_path, stub_upload):
    """Each retained baseline holds a ~3 GB shard that the prune would
    otherwise free, so publishing must drop the previous round's link or disk
    grows by one shard every round."""
    sub = tmp_path / "miner_submission"
    sub.mkdir()
    for block, rid in ((550, 9000), (560, 9001)):
        shard = sub / f"uid_1_hotkey_hkB_block_{block}.safetensors"
        save_file({"w": torch.zeros(4)}, str(shard))
        round_obj = _round_with_winner()
        round_obj.round_id = rid
        distribute.publish_round_baseline(round_obj=round_obj, config=_config(sub), out={})

    retained = sorted(p.name for p in (sub.parent / "baseline").iterdir())
    assert retained == ["round_9001.safetensors"]


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

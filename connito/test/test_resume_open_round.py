"""Resuming a round whose eval window was still open at restart.

Before this path existed, a restart mid-round finalized the journal and every
still-pending miner scored 0. The pieces under test are the two halves of the
fix: `Round.freeze` persisting the base parameters it snapshots, and
`run.resume_open_round` rebuilding the round from that file plus the journal.

The base is the load-bearing part. A resumed miner must be scored against
the same base as one scored before the restart, because
`delta = max(0, baseline - val_loss)` is only comparable within a single base.
Every round's base is the pretrained shard the boot wrote, so a journal that
names anything else must refuse the resume rather than degrade it.

Fixtures follow `test_round_freeze_groups.py`: a tiny `nn.Module`, stubbed
chain reads, no network. Run with
`python -m pytest connito/test/test_resume_open_round.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from connito.validator import round_journal as rj
from connito.validator.round import Round, RoundRef

ROUND_ID = 800
CYCLE_LENGTH = 100


def _shard(root, group_id: int = 1) -> Path:
    """Where `setup_training` writes the pretrained shard for a group."""
    return Path(root) / "pretrained" / f"model_expgroup_{group_id}.safetensors"


class _ModelWithBuffer(nn.Module):
    """One parameter and one buffer.

    The buffer stands in for `FP8Linear`'s quantized weights, which are the
    reason only the `named_parameters()` subset is persisted: buffers are
    rebuilt identically at every boot and carry nothing process-specific.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2, bias=False)
        self.register_buffer("scale", torch.ones(2))
        with torch.no_grad():
            self.lin.weight.fill_(0.1)


def _config(*, flag: bool = False, my_hotkey: str = "vme") -> SimpleNamespace:
    return SimpleNamespace(
        chain=SimpleNamespace(hotkey_ss58=my_hotkey, netuid=7, network="mock"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=1)),
        evaluation=SimpleNamespace(
            enable_round_group_construction=flag,
            cohort_window_cycles=8,
            weight_group_1_size=3,
            weight_group_1_share=0.98,
            weight_group_2_size=5,
            weight_group_2_share=0.02,
            validation_group_a_size=3,
            validation_group_ab_total=13,
            validation_group_c_size=17,
            group_a_min_consensus=1,
            group_a_min_weight_per_validator=0.03,
            cohort_state_filename="cohort_state.json",
        ),
    )


def _metagraph(hotkeys: list[str]) -> SimpleNamespace:
    n = len(hotkeys)
    return SimpleNamespace(
        hotkeys=list(hotkeys),
        incentive=torch.zeros(n),
        weights=torch.zeros((n, n), dtype=torch.float32),
        S=torch.ones(n),
    )


def _freeze(*, checkpoint_path, base_shard=None, miners=("m0", "m1", "m2"), **kw):
    base_shard = base_shard or _shard(checkpoint_path or ".")
    hotkeys = ["vme", *miners]
    metagraph = _metagraph(hotkeys)
    assignment_result = SimpleNamespace(
        assignment={"vme": list(miners)},
        miners_with_checkpoint=list(miners),
        chain_checkpoints_by_hotkey={
            hk: SimpleNamespace(hf_repo_id="repo", hf_revision=hk) for hk in miners
        },
    )
    subtensor = SimpleNamespace(
        block=ROUND_ID, metagraph=lambda netuid=None: metagraph, network="mock"
    )
    with patch("connito.shared.chain.get_chain_commits", return_value=[]), \
         patch("connito.shared.cycle.get_combined_validator_seed", return_value="seed"), \
         patch("connito.shared.cycle.get_validator_seed_from_commit", return_value={}), \
         patch(
             "connito.shared.cycle.get_validator_miner_assignment",
             return_value=assignment_result,
         ):
        return Round.freeze(
            config=_config(),
            subtensor=subtensor,
            metagraph=metagraph,
            base_shard=base_shard,
            round_id=ROUND_ID,
            cycle_index=8,
            cycle_length=CYCLE_LENGTH,
            checkpoint_path=checkpoint_path,
            **kw,
        )


# ---------------------------------------------------------------------------
# Piece 1: freeze records the base shard
# ---------------------------------------------------------------------------


def test_freeze_records_the_base_shard_in_the_journal(tmp_path):
    """A path, not a parameter dump: the backbone is frozen, so the shard
    plus a pretrained backbone reproduces the base exactly."""
    shard = tmp_path / "baseline" / "round_99.safetensors"
    _freeze(checkpoint_path=tmp_path, base_shard=shard)

    journal = rj.load(rj.journal_path_for(tmp_path, ROUND_ID))
    assert journal.base_shard == str(shard)
    # And nothing multi-GB landed beside it.
    assert list(rj.journal_dir(tmp_path).glob("*.pt")) == []


def test_freeze_without_checkpoint_path_writes_nothing(tmp_path):
    """Legacy rounds (no journaling) must not start writing journals."""
    _freeze(checkpoint_path=None)
    assert not rj.journal_path_for(tmp_path, ROUND_ID).exists()


def test_freeze_records_seed_in_journal(tmp_path):
    """`seed` is what lets resume prove the eval batches match."""
    _freeze(checkpoint_path=tmp_path)
    journal = rj.load(rj.journal_path_for(tmp_path, ROUND_ID))
    assert journal is not None
    assert journal.seed == "seed"


# ---------------------------------------------------------------------------
# Piece 2: advance_cohort
# ---------------------------------------------------------------------------


def test_advance_cohort_false_reuses_state(tmp_path):
    """`maybe_advance_cohort` always re-elects, so a resume must skip it.

    Re-running it would build a different A/B/C roster than the round
    actually ran with, since the on-disk state is already this round's.
    """
    from connito.validator.cohort_state import CohortState

    # Groups hold uids, not hotkeys. uid 1..3 are m0..m2 in `_metagraph`.
    state = CohortState(
        cohort_epoch=3,
        expert_group="1",
        weight_group_1=(1,),
        weight_group_2=(2,),
        validation_group_a=(1,),
        validation_group_b=(2,),
        validation_group_c=(3,),
    )
    cfg = _config(flag=True)
    hotkeys = ["vme", "m0", "m1", "m2"]
    metagraph = _metagraph(hotkeys)
    assignment_result = SimpleNamespace(
        assignment={"vme": ["m0", "m1", "m2"]},
        miners_with_checkpoint=["m0", "m1", "m2"],
        chain_checkpoints_by_hotkey={
            hk: SimpleNamespace(hf_repo_id="repo", hf_revision=hk)
            for hk in ("m0", "m1", "m2")
        },
    )
    subtensor = SimpleNamespace(
        block=ROUND_ID, metagraph=lambda netuid=None: metagraph, network="mock"
    )

    with patch("connito.shared.chain.get_chain_commits", return_value=[]), \
         patch("connito.shared.cycle.get_combined_validator_seed", return_value="seed"), \
         patch("connito.shared.cycle.get_validator_seed_from_commit", return_value={}), \
         patch(
             "connito.shared.cycle.get_validator_miner_assignment",
             return_value=assignment_result,
         ), \
         patch(
             "connito.validator.round_groups.maybe_advance_cohort"
         ) as advance:
        resumed = Round.freeze(
            config=cfg,
            subtensor=subtensor,
            metagraph=metagraph,
            base_shard=_shard("."),
            round_id=ROUND_ID,
            cycle_index=8,
            cycle_length=CYCLE_LENGTH,
            cohort_state=state,
            checkpoint_path=None,
            advance_cohort=False,
        )

    advance.assert_not_called()
    assert resumed.cohort_state is state
    assert resumed.validation_group_a == (1,)
    assert resumed.weight_group_1 == (1,)


# ---------------------------------------------------------------------------
# Piece 3: the resume itself
# ---------------------------------------------------------------------------


def _resume(tmp_path, *, phase_name="Train", blocks_remaining=200, model=None, base_shard=None):
    """Drive `run.resume_open_round` against stubbed phase + chain reads."""
    from connito.validator import run as run_mod

    hotkeys = ["vme", "m0", "m1", "m2"]
    metagraph = _metagraph(hotkeys)
    assignment_result = SimpleNamespace(
        assignment={"vme": ["m0", "m1", "m2"]},
        miners_with_checkpoint=["m0", "m1", "m2"],
        chain_checkpoints_by_hotkey={
            hk: SimpleNamespace(hf_repo_id="repo", hf_revision=hk)
            for hk in ("m0", "m1", "m2")
        },
    )
    phase = SimpleNamespace(
        phase_name=phase_name,
        blocks_remaining_in_phase=blocks_remaining,
        cycle_length=CYCLE_LENGTH,
    )
    aggregator = SimpleNamespace(
        last_evaluated_per_uid=lambda: {},
        uid_score_pairs=lambda how=None: {},
    )
    round_ref = RoundRef()
    import threading

    eval_window = threading.Event()
    dl_closed = threading.Event()
    dl_closed.set()

    cfg = _config()
    cfg.ckpt = SimpleNamespace(checkpoint_path=str(tmp_path))

    with patch.object(run_mod, "get_phase_from_api", return_value=phase), \
         patch.object(
             run_mod,
             "get_blocks_from_previous_phase_from_api",
             # Shape matches the live API: a [start, end] pair per phase.
             return_value={"Submission": [ROUND_ID, ROUND_ID + 50]},
         ), \
         patch("connito.shared.chain.get_chain_commits", return_value=[]), \
         patch("connito.shared.cycle.get_combined_validator_seed", return_value="seed"), \
         patch("connito.shared.cycle.get_validator_seed_from_commit", return_value={}), \
         patch(
             "connito.shared.cycle.get_validator_miner_assignment",
             return_value=assignment_result,
         ):
        rid = run_mod.resume_open_round(
            config=cfg,
            subtensor=SimpleNamespace(
                block=ROUND_ID, metagraph=lambda netuid=None: metagraph, network="mock"
            ),
            lite_subtensor=SimpleNamespace(
                metagraph=lambda netuid=None, lite=None: metagraph
            ),
            global_model=model if model is not None else _ModelWithBuffer(),
            base_shard=base_shard or _shard(tmp_path),
            score_aggregator=aggregator,
            score_path=tmp_path / "score.json",
            round_ref=round_ref,
            eval_worker=None,
            eval_window_active=eval_window,
            download_window_closed=dl_closed,
        )
    return rid, round_ref, eval_window, dl_closed


def _seed_journal(tmp_path, **overrides):
    """Write a partially-progressed journal, as a mid-round restart would leave.

    Records this boot's pretrained shard by default so each refusal test
    below exercises its own reason rather than the base one.
    """
    overrides.setdefault("base_shard", str(_shard(tmp_path)))
    payload = dict(
        round_id=ROUND_ID,
        uid_to_hotkey={1: "m0", 2: "m1", 3: "m2"},
        scores={1: 2.25},
        scored_uids=(1,),
        failed_uids=(),
        validation_failed_uids=(2,),
        freeze_zero_uids=(3,),
        freeze_zero_hotkeys={3: "m2"},
        uid_to_val_loss={1: 1.5},
        roster_size=3,
        lifecycle_step=3,
        seed="seed",
        finalized=False,
    )
    payload.update(overrides)
    rj.write_atomic(rj.journal_path_for(tmp_path, ROUND_ID), rj.RoundJournal(**payload))


def test_resume_restores_verdicts_and_base(tmp_path):
    """The happy path: work already done is preserved, base is the boot's shard."""
    shard = _shard(tmp_path)
    _freeze(checkpoint_path=tmp_path, base_shard=shard)
    _seed_journal(tmp_path, base_shard=str(shard))

    # A different model object at resume, as after any restart.
    other = _ModelWithBuffer()
    with torch.no_grad():
        other.lin.weight.fill_(0.9)

    rid, round_ref, eval_window, dl_closed = _resume(tmp_path, model=other)

    assert rid == ROUND_ID
    resumed = round_ref.current
    assert resumed is not None
    # Verdicts carried over rather than redone.
    assert resumed.scored_uids == {1}
    assert resumed.scores == {1: 2.25}
    assert resumed.val_losses == {1: 1.5}
    assert resumed.validation_failed_uids == {2}
    assert resumed.freeze_zero_uids == {3}
    assert resumed.base_shard == shard
    # Workers are handed the round.
    assert eval_window.is_set()
    assert not dl_closed.is_set()


def test_resume_refused_when_the_journal_names_another_base(tmp_path):
    """Frozen under another task, or by a build that adopted round baselines:
    either way the rest of the roster would be scored against a different base
    than the miners already in `scores`."""
    other = _shard(tmp_path, group_id=2)
    _freeze(checkpoint_path=tmp_path, base_shard=other)
    _seed_journal(tmp_path, base_shard=str(other))

    rid, round_ref, eval_window, _ = _resume(tmp_path)

    assert rid is None
    assert round_ref.current is None
    assert not eval_window.is_set()


def test_resume_refused_on_a_journal_without_a_base(tmp_path):
    """Written before `base_shard` existed. Guessing the base would score the
    rest of the roster against a different one than the miners already in
    `scores`."""
    _freeze(checkpoint_path=tmp_path)
    _seed_journal(tmp_path, base_shard="")

    rid, round_ref, eval_window, _ = _resume(tmp_path)

    assert rid is None
    assert round_ref.current is None
    assert not eval_window.is_set()


def test_resume_refused_on_seed_mismatch(tmp_path):
    """A different seed means different eval batches within one round."""
    _freeze(checkpoint_path=tmp_path)
    _seed_journal(tmp_path, seed="a-different-seed")

    rid, round_ref, _, _ = _resume(tmp_path)

    assert rid is None
    assert round_ref.current is None


def test_resume_refused_when_finalized(tmp_path):
    _freeze(checkpoint_path=tmp_path)
    _seed_journal(tmp_path, finalized=True)
    assert _resume(tmp_path)[0] is None


def test_resume_refused_in_commit_phase(tmp_path):
    """MinerCommit1/2 would block startup inside `Round.freeze`."""
    _freeze(checkpoint_path=tmp_path)
    _seed_journal(tmp_path)
    assert _resume(tmp_path, phase_name="MinerCommit1")[0] is None


def test_resume_refused_when_nothing_left(tmp_path):
    """Every roster slot already has a verdict — nothing to resume."""
    _freeze(checkpoint_path=tmp_path)
    _seed_journal(tmp_path, scored_uids=(1, 2, 3), roster_size=3)
    assert _resume(tmp_path)[0] is None


def test_resume_marks_hotkey_drift_as_failed(tmp_path):
    """A re-registered uid must not inherit the previous miner's verdict.

    `add_score` resets a uid's whole history on hotkey mismatch, so silent
    mis-attribution is not a local error.
    """
    _freeze(checkpoint_path=tmp_path)
    # uid 1 was "m0" when the round froze; the journal claims someone else.
    _seed_journal(tmp_path, uid_to_hotkey={1: "someone-else", 2: "m1", 3: "m2"})

    rid, round_ref, _, _ = _resume(tmp_path)

    assert rid == ROUND_ID
    assert 1 in round_ref.current.failed_uids


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_prune_removes_the_journal(tmp_path):
    _freeze(checkpoint_path=tmp_path)
    assert rj.journal_path_for(tmp_path, ROUND_ID).exists()

    rj.prune_before_round(tmp_path, ROUND_ID + 1)

    assert not rj.journal_path_for(tmp_path, ROUND_ID).exists()


def test_journal_seed_is_backward_compatible():
    """A v3 journal written before `seed` existed must still load."""
    import json

    raw = json.dumps({
        "round_id": ROUND_ID,
        "roster_size": 3,
        "schema_version": 3,
        "finalized": False,
    })
    journal = rj.RoundJournal.from_json(raw)
    assert journal.seed == ""
    assert journal.round_id == ROUND_ID

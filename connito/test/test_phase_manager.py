"""Tests for `PhaseManager`, the block-height phase clock.

`PhaseManager` moved from `connito/sn_owner/cycle.py` to
`connito/shared/cycle.py` because the validator (`connito/validator/run.py`)
and the miner (`connito/miner/train.py`) both construct one — it is client
code, not owner-only code, and `sn_owner/` is a local development harness
rather than a deployed service.

It had no coverage before the move. These tests pin the arithmetic that the
whole cycle depends on: phase order, the exact block a phase starts and ends
on, and the wrap into the next cycle. Everything here is pure integer maths
over `config.cycle.*_period`, so the config and subtensor are stubs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from connito.shared.cycle import PhaseManager, PhaseNames

# Production defaults from `CycleCfg`; they sum to a 500-block cycle.
PERIODS = dict(
    distribute_period=20,
    train_period=300,
    commit_period=10,
    submission_period=80,
    validate_period=10,
    merge_period=50,
)
CYCLE_LENGTH = 500

# Half-open [start, end) offsets of each phase within one cycle.
PHASE_SPANS = [
    (PhaseNames.distribute, 0, 20),
    (PhaseNames.train, 20, 320),
    (PhaseNames.miner_commit_1, 320, 330),
    (PhaseNames.miner_commit_2, 330, 340),
    (PhaseNames.submission, 340, 420),
    (PhaseNames.validate, 420, 430),
    (PhaseNames.merge, 430, 480),
    (PhaseNames.validator_commit_1, 480, 490),
    (PhaseNames.validator_commit_2, 490, 500),
]


def make_manager(block: int = 0) -> PhaseManager:
    config = SimpleNamespace(cycle=SimpleNamespace(**PERIODS))
    return PhaseManager(config, SimpleNamespace(block=block))


def test_cycle_length_and_phase_order():
    pm = make_manager()
    assert pm.cycle_length == CYCLE_LENGTH
    assert [p["name"] for p in pm.phases] == [name for name, _, _ in PHASE_SPANS]


@pytest.mark.parametrize("name,start,end", PHASE_SPANS)
def test_get_phase_covers_each_phase_span(name, start, end):
    """First and last block of every phase resolve to that phase."""
    pm = make_manager()
    for block in (start, end - 1):
        resp = pm.get_phase(block)
        assert resp.phase_name == name, f"block {block}"
        assert resp.phase_start_block == start
        assert resp.phase_end_block == end - 1
        assert resp.cycle_index == 0


def test_get_phase_wraps_into_next_cycle():
    """Cycle N's blocks are offset by N * cycle_length, and cycle_index tracks it."""
    pm = make_manager()
    resp = pm.get_phase(CYCLE_LENGTH)
    assert resp.phase_name == PhaseNames.distribute
    assert resp.cycle_index == 1
    assert resp.cycle_block_index == 0
    assert resp.phase_start_block == CYCLE_LENGTH
    assert resp.phase_end_block == CYCLE_LENGTH + 19

    # 320 blocks into cycle 1 is MinerCommit1.
    resp = pm.get_phase(CYCLE_LENGTH + 320)
    assert resp.phase_name == PhaseNames.miner_commit_1
    assert resp.cycle_index == 1
    assert resp.phase_start_block == CYCLE_LENGTH + 320


def test_get_phase_reads_chain_block_when_none_passed():
    pm = make_manager(block=CYCLE_LENGTH + 340)
    resp = pm.get_phase()
    assert resp.block == CYCLE_LENGTH + 340
    assert resp.phase_name == PhaseNames.submission


def test_get_phase_rejects_negative_block():
    with pytest.raises(RuntimeError):
        make_manager().get_phase(-1)


def test_blocks_until_next_phase_wraps_past_phases():
    """A phase already past in this cycle resolves to its next-cycle occurrence."""
    pm = make_manager(block=100)  # mid-Train
    ranges = pm.blocks_until_next_phase()

    # Distribute [0,20) is behind us, so it wraps to the next cycle.
    start, end, until = ranges[PhaseNames.distribute]
    assert (start, end, until) == (CYCLE_LENGTH, CYCLE_LENGTH + 19, 400)

    # Submission is still ahead inside this cycle.
    start, _, until = ranges[PhaseNames.submission]
    assert (start, until) == (340, 240)


def test_previous_phase_block_ranges_is_one_cycle_behind():
    pm = make_manager(block=100)
    nxt = pm.blocks_until_next_phase()
    prev = pm.previous_phase_block_ranges()
    for name, (start, end, _) in nxt.items():
        assert prev[name] == (start - CYCLE_LENGTH, end - CYCLE_LENGTH)

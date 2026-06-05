"""Regression tests for `wait_till` resilience to a transient cycle-phase API outage.

Background: when the cycle-phase API (cycle-api.connito.ai) was briefly
unreachable — e.g. a DNS "Temporary failure in name resolution" blip —
`should_act` returns `(ready=False, blocks_till=poll_fallback_block,
phase_response=None)`. With a negative `block_offset`, `blocks_remaining`
(= blocks_till + block_offset) goes <= 0, so the wait loop used to `break` and
declare the phase "reached" while `phase_response` was still None, then
dereference `phase_response.phase_name` and raise

    AttributeError: 'NoneType' object has no attribute 'phase_name'

which propagated out of `run()` as "Quit training" and took the whole validator
down on a transient network blip. `wait_till` must instead keep polling until
the API recovers and only return a real `PhaseResponse`.
"""

from unittest import mock

import connito.shared.cycle as cycle
from connito.shared.cycle import PhaseNames, PhaseResponse


def _valid_phase(phase_name: str) -> PhaseResponse:
    return PhaseResponse(
        block=1000,
        cycle_length=100,
        cycle_index=10,
        cycle_block_index=5,
        phase_name=phase_name,
        phase_index=2,
        phase_start_block=995,
        phase_end_block=1010,
        blocks_into_phase=5,
        blocks_remaining_in_phase=5,
    )


def test_wait_till_survives_transient_phase_api_outage(monkeypatch):
    """A None phase_response (API down) must not crash; wait_till keeps polling
    and returns the real response once the API recovers."""
    phase = PhaseNames.miner_commit_1
    valid = _valid_phase(phase)

    # First two polls: cycle-api unreachable (None response, the exact shape
    # should_act returns when get_phase_from_api fails). Third poll: recovered.
    side_effect = [
        (False, 3, None),
        (False, 3, None),
        (True, 0, valid),
    ]
    monkeypatch.setattr(cycle, "_TEST_MODE", False)
    monkeypatch.setattr(cycle, "should_act", mock.Mock(side_effect=side_effect))
    monkeypatch.setattr(cycle.time, "sleep", lambda *_a, **_k: None)

    # block_offset=-15 reproduces the production call site (MinerCommit1, -15).
    result = cycle.wait_till(
        config=mock.Mock(), phase_name=phase, block_offset=-15
    )

    assert result is valid
    assert cycle.should_act.call_count == 3


def test_wait_till_does_not_break_with_none_then_returns_real_response(monkeypatch):
    """Even with a single transient failure immediately before recovery, the
    returned response is the real one (never None)."""
    phase = PhaseNames.miner_commit_1
    valid = _valid_phase(phase)

    monkeypatch.setattr(cycle, "_TEST_MODE", False)
    monkeypatch.setattr(
        cycle,
        "should_act",
        mock.Mock(side_effect=[(False, 3, None), (True, 0, valid)]),
    )
    monkeypatch.setattr(cycle.time, "sleep", lambda *_a, **_k: None)

    result = cycle.wait_till(
        config=mock.Mock(), phase_name=phase, block_offset=-15
    )

    assert result is not None
    assert result.phase_name == phase

"""The outer optimizer's momentum must not sit on the card between rounds.

DiLoCo-style SGD with momentum allocates a `momentum_buffer` the size of the
parameter set on its first `step()` and keeps it for the process lifetime. On
the partial global model that is a second ~9.5 GiB, and it was harmless right
up until `evaluation.full_topology_eval` asked for a 29.3 GiB frozen base in
the same window:

    round start, pre-merge   allocated =  9,513.3 MB
    steady, post-merge       allocated = 19,027.8 MB
    next round's base move   19.0 + 29.3 = 48.3 GiB  ->  OOM on a 44.39 GiB card

Which is exactly what happened: a CUDA OOM inside `prepare_for_round`'s
`model.to(device)` on the *second* round of every process, first round fitting
because momentum did not exist yet. Validators crash-looped every ~2 cycles.

The fix relocates the buffer, so the two things worth pinning are that it
actually moves and that moving it changes nothing about the optimisation.
The second matters more: a fix that quietly reset momentum would look correct
here and silently alter every validator's merge trajectory.

Run with `python -m pytest connito/test/test_outer_optimizer_offload.py`.
"""

from __future__ import annotations

import torch

from connito.validator.run import _move_outer_optimizer_state

MOMENTUM = 0.9
LR = 0.7


def _sgd(params) -> torch.optim.SGD:
    return torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, nesterov=True)


def _param(value: float = 1.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((4, 3), value))


def _step(optimizer, params, grad_value: float) -> None:
    for p in params:
        p.grad = torch.full_like(p, grad_value)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_no_buffers_before_the_first_step():
    """Why the first round of every process fitted and the second did not."""
    params = [_param()]
    optimizer = _sgd(params)
    assert _move_outer_optimizer_state(optimizer, "cpu") == 0.0
    assert optimizer.state == {} or all(
        "momentum_buffer" not in s for s in optimizer.state.values()
    )


def test_buffers_move_and_report_their_size():
    params = [_param()]
    optimizer = _sgd(params)
    _step(optimizer, params, 0.5)

    buffers = [s["momentum_buffer"] for s in optimizer.state.values()]
    assert buffers and all(b.device.type == "cpu" for b in buffers)

    expected_gb = sum(b.numel() * b.element_size() for b in buffers) / 1024 ** 3
    # Already on CPU in this environment, so the move is a no-op and reports 0.
    assert _move_outer_optimizer_state(optimizer, "cpu") == 0.0
    assert expected_gb > 0


def test_relocation_is_idempotent():
    params = [_param()]
    optimizer = _sgd(params)
    _step(optimizer, params, 0.5)
    before = [s["momentum_buffer"].clone() for s in optimizer.state.values()]

    for _ in range(3):
        _move_outer_optimizer_state(optimizer, "cpu")

    after = [s["momentum_buffer"] for s in optimizer.state.values()]
    for want, got in zip(before, after):
        assert torch.equal(want, got)


def test_offload_does_not_change_the_momentum_trajectory():
    """The claim the whole fix rests on: relocation, not reset.

    A buggy offload that dropped or zeroed the buffer would pass every
    device-placement assertion above and silently change how every validator
    merges. So run two optimizers over identical gradients — one relocated
    between every step, one left alone — and require bit-identical weights.
    """
    torch.manual_seed(0)
    plain_params = [_param(), _param(2.0)]
    moved_params = [_param(), _param(2.0)]
    plain, moved = _sgd(plain_params), _sgd(moved_params)

    for grad_value in (0.5, -0.25, 1.5, 0.75, -2.0):
        _step(plain, plain_params, grad_value)
        _step(moved, moved_params, grad_value)
        # The round boundary: park, then bring it back for the next step.
        _move_outer_optimizer_state(moved, "cpu")
        _move_outer_optimizer_state(moved, "cpu")

    for want, got in zip(plain_params, moved_params):
        assert torch.equal(want.data, got.data)

    for want_state, got_state in zip(plain.state.values(), moved.state.values()):
        assert torch.equal(want_state["momentum_buffer"], got_state["momentum_buffer"])


def test_every_parameter_group_is_covered():
    """`optimizer.state` is keyed per-parameter; a partial sweep would leave
    part of the ~9.5 GiB resident and the OOM would come back at a slightly
    later round rather than being fixed."""
    params = [_param(), _param(3.0), _param(-1.0)]
    optimizer = _sgd(params)
    _step(optimizer, params, 0.5)

    buffers = [s["momentum_buffer"] for s in optimizer.state.values()]
    assert len(buffers) == len(params)
    assert all(b.device.type == "cpu" for b in buffers)

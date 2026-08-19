"""Gradient buffers are allocated only for the group that will use one.

`build_grad_buff_from_model` built a buffer for *every* expert group, and
`validator.run` deleted all but its own on the next line. Each buffer is a real
`torch.zeros` in host RAM, so a validator allocated and binned a full second
copy on every start:

  - partial topology (what the fleet runs): 7.25 GB discarded
  - full topology: 27.4 GB discarded, on a host where that mattered

Observed on the tester, both buffers reporting the same size because on the full
model every group also claims every expert (a separate bug — the stacked-mode
bucketing matches by layer, not by expert; not fixed here):

    Built expert group grad buffer - 4   27456 MB   total_numel=14394851328
    Built expert group grad buffer - 2   27456 MB   total_numel=14394851328
    Disabling averager for non-active expert group  excluded_group_id=2

Run with `python -m pytest connito/test/test_grad_buffer_group_scope.py`.
"""

from __future__ import annotations

import torch
from torch import nn

from connito.validator.inter_validator_connection import build_grad_buff_from_model

# group_id -> layer_id -> [(local_slot, global_expert_id), ...]
#
# Both layers appear under both groups. An unassigned layer would leave its
# expert tensors owned by nobody, which sends them to the shared buffer and
# makes `test_shared_ownership_still_spans_every_group` fail for a reason that
# has nothing to do with what it is testing.
ASSIGNMENT = {
    4: {0: [(0, 0), (1, 1)], 1: [(0, 0), (1, 1)]},
    2: {0: [(0, 2), (1, 3)], 1: [(0, 2), (1, 3)]},
}


class _StackedModel(nn.Module):
    """Stacked expert storage: one tensor per layer holding every expert.

    The naming matters — `get_layer_expert_id` must find a layer and *no*
    expert index, or the function takes its per-expert branch instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.mlp = nn.Module()
            layer.mlp.experts = nn.Module()
            layer.mlp.experts.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 4))
            layer.mlp.experts.down_proj = nn.Parameter(torch.zeros(4, 4, 8))
        self.model.embed_tokens = nn.Embedding(4, 4)


def _buffer_elements(meta: dict) -> int:
    return int(meta["buff"].numel())


def test_every_group_is_built_when_no_scope_is_given():
    """The default has to stay as it was — other callers pass no scope."""
    metas = build_grad_buff_from_model(
        model=_StackedModel(), expert_group_assignment=ASSIGNMENT,
    )
    assert set(metas) == {4, 2}


def test_only_the_requested_group_is_allocated():
    metas = build_grad_buff_from_model(
        model=_StackedModel(), expert_group_assignment=ASSIGNMENT, group_ids=[4],
    )
    assert set(metas) == {4}


def test_the_skipped_group_costs_no_memory():
    """The regression guard.

    Asserting on the returned keys alone would have passed against the old code
    too — it built both and the caller deleted one. What changed is that the
    allocation never happens, so this compares total buffer elements.
    """
    model = _StackedModel()
    all_groups = build_grad_buff_from_model(
        model=model, expert_group_assignment=ASSIGNMENT,
    )
    one_group = build_grad_buff_from_model(
        model=model, expert_group_assignment=ASSIGNMENT, group_ids=[4],
    )

    total_all = sum(_buffer_elements(m) for m in all_groups.values())
    total_one = sum(_buffer_elements(m) for m in one_group.values())
    assert total_one < total_all
    assert total_one == _buffer_elements(all_groups[4])


def test_shared_ownership_still_spans_every_group():
    """Scoping the *allocation* must not narrow the *ownership* picture.

    `include_shared` collects tensors owned by no group. If the scope filtered
    the assignment itself, group 2's experts would look unowned and land in the
    shared buffer — averaged with every peer instead of within their group.
    Only the allocation loop is scoped; the name bucketing still runs over all
    groups, which is why this holds.
    """
    scoped = build_grad_buff_from_model(
        model=_StackedModel(),
        expert_group_assignment=ASSIGNMENT,
        include_shared=True,
        group_ids=[4],
    )
    shared_names = scoped["shared"]["param_names"]
    assert not any("experts" in name for name in shared_names)
    assert any("embed_tokens" in name for name in shared_names)


def test_an_unknown_group_id_allocates_nothing():
    metas = build_grad_buff_from_model(
        model=_StackedModel(), expert_group_assignment=ASSIGNMENT, group_ids=[99],
    )
    assert metas == {}

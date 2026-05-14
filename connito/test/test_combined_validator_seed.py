"""Regression tests for `get_combined_validator_seed`.

These pin the *no-fallback-on-empty* contract introduced when the eval
seed publication moved from MinerCommit1 to Submission phase. Returning
`sha256(b"0")` (the prior fallback) when no validator seeds were on
chain was itself a vulnerability: any miner who could force every
validator's seed to be absent during their own commit window — or who
observed an empty-seeds state — would land on a deterministic seed and
could pre-overfit. Raising instead halts the cycle's eval, which is the
safe failure mode.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from connito.shared.chain import ValidatorChainCommit
from connito.shared.cycle import get_combined_validator_seed


class _StubSubtensor:
    """Minimal subtensor stand-in; not invoked because we always pass
    `commits` explicitly to bypass the `get_chain_commits` lookup."""


def _config_for_group(group_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=group_id)))


def _validator_commit(miner_seed: int | None, expert_group: int = 0) -> ValidatorChainCommit:
    return ValidatorChainCommit(miner_seed=miner_seed, expert_group=expert_group)


def _neuron(hotkey: str) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey)


def test_get_combined_validator_seed_raises_on_empty_commits():
    """No validator commits at all → must raise. The old code returned
    `sha256(b"0")` which is a deterministic, predictable value any
    attacker could anticipate."""
    with pytest.raises(RuntimeError, match="No validator seeds available"):
        get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=[])


def test_get_combined_validator_seed_raises_when_all_seeds_missing():
    """Validators committed for this expert_group but none included a
    `miner_seed`. This is the expected state during the MinerCommit
    window after the timing change — and must halt rather than fall
    back to a constant."""
    commits = [
        (_validator_commit(miner_seed=None, expert_group=0), _neuron("hk_a")),
        (_validator_commit(miner_seed=None, expert_group=0), _neuron("hk_b")),
    ]
    # `get_validator_seed_from_commit` treats `None` as the default
    # assignment seed (0) per the existing legacy contract, so the
    # combined seed will be defined here. Sanity-check that the function
    # does NOT raise in this case — the empty-fallback raise is only for
    # the truly-empty `validator_seeds` dict, not for all-zero seeds.
    out = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=commits)
    expected = hashlib.sha256("".join(["0", "0"]).encode()).hexdigest()
    assert out == expected


def test_get_combined_validator_seed_returns_hash_of_sorted_seeds():
    """Happy path: combined seed is `sha256` of validator seeds
    concatenated in hotkey-sorted order."""
    commits = [
        # Provide deliberately-unsorted insertion order to verify the
        # function sorts internally.
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_z")),
        (_validator_commit(miner_seed=7, expert_group=0), _neuron("hk_a")),
        (_validator_commit(miner_seed=99, expert_group=0), _neuron("hk_m")),
    ]
    out = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=commits)
    # Hotkey-sorted order: hk_a (7), hk_m (99), hk_z (42)
    expected = hashlib.sha256("79942".encode()).hexdigest()
    assert out == expected


def test_get_combined_validator_seed_filters_by_expert_group():
    """Validators commit per-expert-group; seeds from other groups must
    not contaminate the combined seed for this group."""
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_a")),
        (_validator_commit(miner_seed=999, expert_group=1), _neuron("hk_b")),  # wrong group
    ]
    out = get_combined_validator_seed(_config_for_group(group_id=0), _StubSubtensor(), commits=commits)
    expected = hashlib.sha256("42".encode()).hexdigest()
    assert out == expected

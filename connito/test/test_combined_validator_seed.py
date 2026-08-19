"""Regression tests for `get_combined_validator_seed`.

After the block-hash-only cutover, the combined validator seed is
`sha256(block_hash).hexdigest()`, where `block_hash` is the hash of
the LAST block of the most recent completed MinerCommit2 phase.

Validator-committed `miner_seed` values no longer contribute. The
field is still present on `ValidatorChainCommit` for
`get_validator_miner_assignment` (which uses it for deterministic
validator → miner assignment, a separate concern from the
collusion-resistant eval seed), but it is not mixed into the eval
seed any longer because miners can read it on chain during their
commit window — making any seed mix that includes it predictable to a
colluding miner.

If `_get_minercommit2_block_hash` returns None (phase API or chain
RPC failure), `get_combined_validator_seed` MUST raise rather than
falling back to `sha256("")`. The fallback was the known deficiency
in the previous mixed-seed PR — a publicly-known constant that let
miners win the round trivially during transient outages.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from connito.shared.chain import ValidatorChainCommit
from connito.shared.cycle import PhaseNames, get_combined_validator_seed


# A stable, made-up block hash for tests. Real block hashes are
# 0x-prefixed 64-hex strings; matching that format ensures we exercise
# the same hashing code path.
TEST_BLOCK_HASH = "0xabcdef0123456789" + ("0" * 48)
TEST_PHASE_END_BLOCK = 8_184_162


class _StubSubtensor:
    """Minimal subtensor stand-in. The block-hash helper is patched
    per-test; `get_block_hash` is here only because the real subtensor
    exposes one and a few non-seed paths inside cycle.py touch it."""

    def get_block_hash(self, block: int) -> str:
        return TEST_BLOCK_HASH


def _config_for_group(group_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=group_id)))


def _validator_commit(miner_seed: int | None, expert_group: int = 0) -> ValidatorChainCommit:
    """Construct a chain commit. The `miner_seed` value is no longer
    mixed into the combined seed, but the field remains on the chain
    commit type — tests still need to instantiate it for the few
    sibling paths (assignment, telemetry) that read the field."""
    return ValidatorChainCommit(miner_seed=miner_seed, expert_group=expert_group)


def _neuron(hotkey: str) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey)


# Patches the block-hash helper so tests don't actually hit the phase
# API or a live subtensor. Returning None simulates RPC failure; a
# string simulates a successful chain query.
def _patch_block_hash(value: str | None):
    return patch(
        "connito.shared.cycle._get_minercommit2_block_hash",
        return_value=value,
    )


def test_combined_seed_is_sha256_of_block_hash():
    """Happy path: combined seed is sha256(block_hash). Nothing else
    contributes. Even when `commits` is passed and contains
    validator-committed `miner_seed` values, those values must NOT
    affect the result — that's the whole point of the cutover."""
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_z")),
        (_validator_commit(miner_seed=7, expert_group=0), _neuron("hk_a")),
        (_validator_commit(miner_seed=99, expert_group=0), _neuron("hk_m")),
    ]
    with _patch_block_hash(TEST_BLOCK_HASH):
        out = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    expected = hashlib.sha256(TEST_BLOCK_HASH.encode()).hexdigest()
    assert out == expected


def test_combined_seed_ignores_validator_committed_miner_seed():
    """Same block hash, different per-validator `miner_seed`
    commitments → IDENTICAL combined seed. Confirms the legacy
    validator_seeds component has truly been removed from the seed
    derivation (a half-finished refactor that still concatenated an
    empty placeholder would break this)."""
    block_hash = TEST_BLOCK_HASH
    commits_a = [(_validator_commit(miner_seed=1, expert_group=0), _neuron("hk_a"))]
    commits_b = [(_validator_commit(miner_seed=999_999, expert_group=0), _neuron("hk_a"))]
    commits_empty: list[tuple[ValidatorChainCommit, SimpleNamespace]] = []

    with _patch_block_hash(block_hash):
        a = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=commits_a)
        b = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=commits_b)
        empty = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=commits_empty)
        no_commits = get_combined_validator_seed(_config_for_group(), _StubSubtensor())

    assert a == b == empty == no_commits


def test_combined_seed_raises_when_block_hash_unavailable():
    """The previous behaviour fell back to `sha256("")` — a publicly
    known constant that let miners win the round trivially during
    transient phase-API / chain-RPC outages. The cutover replaces
    that fallback with a hard raise. The validator framework decides
    whether to retry or skip the round upstream."""
    with _patch_block_hash(None):
        with pytest.raises(RuntimeError, match="block hash unavailable"):
            get_combined_validator_seed(
                _config_for_group(), _StubSubtensor(), commits=[],
            )


def test_combined_seed_raises_when_block_hash_empty_string():
    """An empty-string block hash is treated the same as None — the
    `if not block_hash:` guard catches both, so a chain returning ""
    (defensive zero-value, unusual but possible) still raises rather
    than producing a guessable seed."""
    with _patch_block_hash(""):
        with pytest.raises(RuntimeError, match="block hash unavailable"):
            get_combined_validator_seed(
                _config_for_group(), _StubSubtensor(), commits=[],
            )


def test_combined_seed_changes_when_block_hash_changes():
    """Different block hashes → different combined seeds. This is the
    cycle-over-cycle rotation that makes the eval data slice fresh
    each round."""
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_a")),
    ]
    with _patch_block_hash(TEST_BLOCK_HASH):
        seed_a = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    other_block_hash = "0xdeadbeef" + ("0" * 56)
    with _patch_block_hash(other_block_hash):
        seed_b = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    assert seed_a != seed_b


def test_combined_seed_is_deterministic_for_same_block_hash():
    """Same input → same output, always. Required for every validator
    on the network to score the same rows for the same seed; without
    this, weight consensus breaks every round."""
    with _patch_block_hash(TEST_BLOCK_HASH):
        a = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=[])
        b = get_combined_validator_seed(_config_for_group(), _StubSubtensor(), commits=[])
    assert a == b
    # And cross-check: matches a hand-computed expected hash so a
    # future refactor that subtly changes the hashing construction
    # (e.g. switches separator, swaps inputs) fails loud.
    assert a == hashlib.sha256(TEST_BLOCK_HASH.encode()).hexdigest()

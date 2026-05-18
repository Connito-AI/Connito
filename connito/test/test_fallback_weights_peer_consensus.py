"""Tests for the peer-consensus shape of `_submit_fallback_weights`.

When the validator falls into the fallback path (no usable previous
weights of its own), the helper aggregates stake-weighted votes from
*differentiated* peers and emits even weight to the top-3. If every
other validator is itself in an even-weight fallback state, the
helper emits 100% weight to uid=0 rather than echoing the
uninformative signal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from connito.shared.chain import (
    SUBNET_OWNER_UID,
    SUBNET_OWNER_WEIGHT_SHARE,
    _FALLBACK_TOP_N_PEERS,
    _submit_fallback_weights,
    reserve_subnet_owner_share,
)


class _FakeMetagraph:
    def __init__(self, hotkeys: list[str]) -> None:
        self.hotkeys = list(hotkeys)


class _FakeNeuron:
    def __init__(
        self,
        stake: float,
        weights: list[tuple[int, float]],
    ) -> None:
        self.stake = stake
        self.weights = weights


class _FakeSubtensor:
    """Minimal stand-in matching the surface `_submit_fallback_weights`
    touches: `.metagraph`, `.neuron_for_uid`, `.set_weights`, and
    `.block` (read by `submit_weights` when logging a successful
    submission via the prev-weights reuse branch).
    """

    block: int = 1

    def __init__(
        self,
        metagraph: _FakeMetagraph,
        neurons: dict[int, _FakeNeuron],
    ) -> None:
        self._metagraph = metagraph
        self._neurons = neurons
        # Captured submission for assertions.
        self.set_weights_calls: list[dict] = []

    def metagraph(self, *, netuid: int):
        return self._metagraph

    def neuron_for_uid(self, *, uid: int, netuid: int):
        return self._neurons.get(uid)

    def set_weights(self, **kwargs):
        self.set_weights_calls.append(kwargs)
        return (True, "")


def _config():
    return SimpleNamespace(chain=SimpleNamespace(netuid=1))


def _wallet(hotkey: str):
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=hotkey))


@pytest.fixture
def patched_whitelist(monkeypatch):
    """Stub `get_validator_whitelist_from_api` so each test can pick its
    own validator set."""
    def _set(*hotkeys: str):
        def fake(_config):
            return list(hotkeys)
        monkeypatch.setattr(
            "connito.shared.cycle.get_validator_whitelist_from_api", fake,
        )
    return _set


def test_top_n_constant_matches_request():
    """The user-facing requirement was top-3."""
    assert _FALLBACK_TOP_N_PEERS == 3


def _assert_miners_plus_owner_share(call, expected_miner_uids):
    """The peer-consensus top-N path now reserves
    `SUBNET_OWNER_WEIGHT_SHARE` for `SUBNET_OWNER_UID` after the even
    miner split, so every successful peer submission carries
    [miner_uid_1, ..., miner_uid_N, owner_uid] with weights
    [share/N * (1 - owner_share)] * N + [owner_share].
    """
    expected_uids = [*expected_miner_uids, SUBNET_OWNER_UID]
    n = len(expected_miner_uids)
    per_miner = (1 - SUBNET_OWNER_WEIGHT_SHARE) / n
    expected_weights = [per_miner] * n + [SUBNET_OWNER_WEIGHT_SHARE]
    assert call["uids"] == expected_uids, (call["uids"], expected_uids)
    assert call["weights"] == pytest.approx(expected_weights)


def test_emits_top_3_when_differentiated_peer_present(patched_whitelist):
    # Validators: me (uid 0), peer1 (uid 1). Miners: uids 2..6.
    patched_whitelist("me_hk", "peer1_hk")
    metagraph = _FakeMetagraph(
        ["me_hk", "peer1_hk", "m_a", "m_b", "m_c", "m_d", "m_e"]
    )
    neurons = {
        # self: no prior weights
        0: _FakeNeuron(stake=0.0, weights=[]),
        # peer1: differentiated, top-3 should be uids 2, 3, 4
        1: _FakeNeuron(
            stake=100.0,
            weights=[(2, 0.5), (3, 0.3), (4, 0.1), (5, 0.07), (6, 0.03)],
        ),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    assert len(sub.set_weights_calls) == 1
    _assert_miners_plus_owner_share(sub.set_weights_calls[0], [2, 3, 4])


def test_even_peers_are_excluded_from_aggregation(patched_whitelist):
    """An even-weight peer's votes don't contribute to the top-3."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(
        ["me_hk", "peer1_hk", "peer2_hk", "m_a", "m_b", "m_c", "m_d"]
    )
    neurons = {
        0: _FakeNeuron(stake=0.0, weights=[]),
        # peer1: would dump 100 stake onto uids 3, 4, 5 evenly
        1: _FakeNeuron(
            stake=100.0,
            weights=[(3, 1 / 3), (4, 1 / 3), (5, 1 / 3)],
        ),
        # peer2: small stake, differentiated, picks uid 6
        2: _FakeNeuron(stake=10.0, weights=[(6, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    # Only peer2's vote counts → uid 6, plus the owner share.
    _assert_miners_plus_owner_share(sub.set_weights_calls[0], [6])


def test_all_peers_even_falls_back_to_uid_zero(patched_whitelist):
    """When every other validator is in even-weight fallback, the
    helper emits 100% weight on uid=0 rather than echoing."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(
        ["me_hk", "peer1_hk", "peer2_hk", "m_a", "m_b", "m_c"]
    )
    neurons = {
        0: _FakeNeuron(stake=0.0, weights=[]),
        # All peers in even-weight fallback shape.
        1: _FakeNeuron(stake=100.0, weights=[(3, 0.5), (4, 0.5)]),
        2: _FakeNeuron(
            stake=50.0,
            weights=[(3, 1 / 3), (4, 1 / 3), (5, 1 / 3)],
        ),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    call = sub.set_weights_calls[0]
    assert call["uids"] == [0]
    assert call["weights"] == [1.0]


def test_no_other_validators_falls_back_to_uid_zero(patched_whitelist):
    """Single-validator subnet (or whitelist of one) — no peers to
    consult — still emits uid=0 fallback rather than returning False."""
    patched_whitelist("me_hk")
    metagraph = _FakeMetagraph(["me_hk", "m_a"])
    neurons = {0: _FakeNeuron(stake=0.0, weights=[])}
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    assert sub.set_weights_calls[0]["uids"] == [0]


def test_single_uid_vote_counts_as_differentiated(patched_whitelist):
    """A peer voting all-on-one miner is making a concrete pick, not an
    even-weight fallback, and contributes to aggregation."""
    patched_whitelist("me_hk", "peer1_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "m_a"])
    neurons = {
        0: _FakeNeuron(stake=0.0, weights=[]),
        1: _FakeNeuron(stake=50.0, weights=[(2, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    _assert_miners_plus_owner_share(sub.set_weights_calls[0], [2])


def test_validator_uids_excluded_from_top_n(patched_whitelist):
    """A peer voting for another validator's UID (rare but possible)
    must not leak that validator into the miner-only top-N."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "peer2_hk", "m_a"])
    neurons = {
        0: _FakeNeuron(stake=0.0, weights=[]),
        # peer1 votes for peer2 (uid 2 — a validator) and miner uid 3
        1: _FakeNeuron(stake=100.0, weights=[(2, 0.5), (3, 0.5)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    # uid 2 is a validator and must be filtered; only uid 3 remains
    # (plus the subnet-owner share).
    _assert_miners_plus_owner_share(sub.set_weights_calls[0], [3])


def test_zero_stake_peers_skipped(patched_whitelist):
    """A peer with zero stake contributes nothing to the weighted sum
    (no behaviour change vs. before this PR, just confirming)."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(
        ["me_hk", "peer1_hk", "peer2_hk", "m_a", "m_b"]
    )
    neurons = {
        0: _FakeNeuron(stake=0.0, weights=[]),
        # zero stake — should be skipped despite differentiation
        1: _FakeNeuron(stake=0.0, weights=[(3, 0.7), (4, 0.3)]),
        # nonzero, differentiated
        2: _FakeNeuron(stake=50.0, weights=[(4, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    # Only peer2 counts → uid 4 (plus the subnet-owner share).
    _assert_miners_plus_owner_share(sub.set_weights_calls[0], [4])


def test_prev_weights_are_ignored(patched_whitelist):
    """Our own previous on-chain weights are intentionally not used as
    a fallback source — peer consensus is consulted unconditionally.
    Even when self has a differentiated history on miners 2 and 3, the
    submission reflects peer consensus (peer1 picks miner 2 only)."""
    patched_whitelist("me_hk", "peer1_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "m_a", "m_b"])
    neurons = {
        # Self has differentiated previous weights — these MUST be
        # ignored under the current contract.
        0: _FakeNeuron(stake=0.0, weights=[(2, 0.8), (3, 0.2)]),
        1: _FakeNeuron(stake=100.0, weights=[(2, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    ok = _submit_fallback_weights(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub,
        wait_for_inclusion=False, wait_for_finalization=False,
    )
    assert ok is True
    assert sub.set_weights_calls, "expected a set_weights call"
    # Peer consensus from peer1 → uid 2 only (plus owner share). uid 3
    # (only in our own prior weights, not in any peer's vote) must NOT
    # appear.
    _assert_miners_plus_owner_share(sub.set_weights_calls[0], [2])


# ---------------------------------------------------------------------
# reserve_subnet_owner_share: unit tests for the helper itself
# ---------------------------------------------------------------------


def test_reserve_owner_share_normal_case():
    out = reserve_subnet_owner_share({1: 0.6, 2: 0.4})
    assert out == pytest.approx({
        1: 0.6 * 0.95,
        2: 0.4 * 0.95,
        0: 0.05,
    })
    assert sum(out.values()) == pytest.approx(1.0)


def test_reserve_owner_share_empty_input():
    """Empty input → owner gets all of it."""
    assert reserve_subnet_owner_share({}) == {0: 1.0}


def test_reserve_owner_share_owner_already_present():
    """If uid=0 was already in the input, its scaled-down weight stacks
    on top of the reserved share."""
    out = reserve_subnet_owner_share({0: 0.2, 1: 0.8})
    assert out == pytest.approx({
        0: 0.2 * 0.95 + 0.05,
        1: 0.8 * 0.95,
    })
    assert sum(out.values()) == pytest.approx(1.0)


def test_reserve_owner_share_zero_share_is_passthrough():
    inp = {1: 0.5, 2: 0.5}
    out = reserve_subnet_owner_share(inp, share=0.0)
    assert out == inp
    assert 0 not in out


def test_reserve_owner_share_full_share_collapses_to_owner():
    out = reserve_subnet_owner_share({1: 0.5, 2: 0.5}, share=1.0)
    assert out == {0: 1.0}


def test_reserve_owner_share_custom_owner_uid():
    """The owner UID is configurable (defaults to subnet-owner uid=0)."""
    out = reserve_subnet_owner_share({1: 1.0}, owner_uid=42, share=0.1)
    assert out == pytest.approx({1: 0.9, 42: 0.1})


def test_reserve_owner_share_preserves_sum_with_unequal_weights():
    inp = {1: 0.7, 2: 0.2, 3: 0.1}
    out = reserve_subnet_owner_share(inp)
    assert sum(out.values()) == pytest.approx(1.0)
    # Ratios among miners preserved.
    assert out[1] / out[2] == pytest.approx(inp[1] / inp[2])
    assert out[2] / out[3] == pytest.approx(inp[2] / inp[3])

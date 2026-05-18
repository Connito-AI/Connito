"""Unit tests for `get_peer_consensus_top_uids`.

The helper reads other validators' on-chain weights and returns the
top-N stake-weighted miner UIDs. Tests rely on fake subtensor /
metagraph stubs rather than touching a real chain.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from connito.shared.chain import get_peer_consensus_top_uids


class _FakeMetagraph:
    def __init__(self, hotkeys: list[str]) -> None:
        self.hotkeys = list(hotkeys)


class _FakeNeuron:
    def __init__(self, stake: float, weights: list[tuple[int, float]]) -> None:
        self.stake = stake
        self.weights = weights


class _FakeSubtensor:
    def __init__(self, metagraph: _FakeMetagraph, neurons: dict[int, _FakeNeuron]) -> None:
        self._metagraph = metagraph
        self._neurons = neurons

    def metagraph(self, *, netuid: int):
        return self._metagraph

    def neuron_for_uid(self, *, uid: int, netuid: int):
        return self._neurons.get(uid)


def _config():
    return SimpleNamespace(chain=SimpleNamespace(netuid=1))


def _wallet(hotkey: str):
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=hotkey))


@pytest.fixture
def patched_whitelist(monkeypatch):
    """Stub `get_validator_whitelist_from_api` so we can pick which UIDs
    are treated as validators in each scenario."""
    def _set(*hotkeys: str):
        def fake(_config):
            return list(hotkeys)
        monkeypatch.setattr(
            "connito.shared.cycle.get_validator_whitelist_from_api", fake,
        )
    return _set


def test_returns_top_n_when_one_peer_is_differentiated(patched_whitelist):
    # Hotkeys: 0=me, 1=peer-validator, 10/11/12=miners
    patched_whitelist("me_hk", "peer1_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "m_a", "m_b", "m_c"])
    # peer1 weights miners 10,11,12 differently → differentiated.
    neurons = {
        1: _FakeNeuron(stake=100.0, weights=[(2, 0.5), (3, 0.3), (4, 0.2)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    assert out == [2, 3, 4]


def test_returns_none_when_all_peers_emit_even_weights(patched_whitelist):
    """If every peer has 2+ equal miner weights, signal is uninformative
    — return None so caller can escalate to a different fallback."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "peer2_hk", "m_a", "m_b", "m_c"])
    neurons = {
        1: _FakeNeuron(stake=100.0, weights=[(3, 0.5), (4, 0.5)]),  # even
        2: _FakeNeuron(stake=100.0, weights=[(3, 0.333), (4, 0.333), (5, 0.333)]),  # even
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    assert out is None


def test_even_peers_are_excluded_from_aggregation(patched_whitelist):
    """An even-weight peer must not pollute the aggregation — the
    returned ranking should reflect only the differentiated peer."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(
        ["me_hk", "peer1_hk", "peer2_hk", "m_a", "m_b", "m_c", "m_d"]
    )
    neurons = {
        # peer1 (even): would otherwise dump 100 stake onto uids 3,4,5
        1: _FakeNeuron(stake=100.0, weights=[(3, 0.333), (4, 0.333), (5, 0.333)]),
        # peer2 (differentiated): clear preference for uid=6
        2: _FakeNeuron(stake=10.0, weights=[(6, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    # Only peer2's votes count; uid 6 is the only miner with stake.
    assert out == [6]


def test_single_uid_vote_counts_as_differentiated(patched_whitelist):
    """A peer with weight on exactly one miner is a concrete pick, not
    an even-weight fallback — should contribute to aggregation."""
    patched_whitelist("me_hk", "peer1_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "m_a"])
    neurons = {
        1: _FakeNeuron(stake=50.0, weights=[(2, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    assert out == [2]


def test_validator_uids_excluded_from_result(patched_whitelist):
    """A peer voting for another validator (rare but possible) must not
    leak that validator into the miner-only result set."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "peer2_hk", "m_a"])
    neurons = {
        # peer1 votes for peer2 (uid=2, also a validator) and miner uid=3
        1: _FakeNeuron(stake=100.0, weights=[(2, 0.5), (3, 0.5)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    # uid 2 (peer2 validator) must be filtered out; uid 3 remains.
    assert out == [3]


def test_returns_none_when_no_other_validators(patched_whitelist):
    patched_whitelist("me_hk")
    metagraph = _FakeMetagraph(["me_hk", "m_a"])
    sub = _FakeSubtensor(metagraph, {})
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    assert out is None


def test_returns_none_when_self_not_in_metagraph(patched_whitelist):
    patched_whitelist("peer1_hk")
    metagraph = _FakeMetagraph(["peer1_hk", "m_a"])
    sub = _FakeSubtensor(metagraph, {})
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk_not_present"), subtensor=sub, top_n=3,
    )
    assert out is None


def test_zero_stake_peers_are_skipped(patched_whitelist):
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "peer2_hk", "m_a", "m_b"])
    neurons = {
        1: _FakeNeuron(stake=0.0, weights=[(3, 0.7), (4, 0.3)]),  # zero stake
        2: _FakeNeuron(stake=50.0, weights=[(3, 1.0)]),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    # peer1 contributed nothing; peer2 picked uid 3.
    assert out == [3]


def test_top_n_limits_output_length(patched_whitelist):
    patched_whitelist("me_hk", "peer1_hk")
    metagraph = _FakeMetagraph(
        ["me_hk", "peer1_hk", "m1", "m2", "m3", "m4", "m5"]
    )
    neurons = {
        1: _FakeNeuron(
            stake=100.0,
            weights=[(2, 0.5), (3, 0.3), (4, 0.1), (5, 0.07), (6, 0.03)],
        ),
    }
    sub = _FakeSubtensor(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    assert out == [2, 3, 4]


def test_per_peer_exception_does_not_abort_aggregation(patched_whitelist):
    """A failure reading one peer's neuron must not poison the result —
    other peers' contributions should still aggregate."""
    patched_whitelist("me_hk", "peer1_hk", "peer2_hk")
    metagraph = _FakeMetagraph(["me_hk", "peer1_hk", "peer2_hk", "m_a"])

    class _PartiallyBrokenSub(_FakeSubtensor):
        def neuron_for_uid(self, *, uid: int, netuid: int):
            if uid == 1:
                raise RuntimeError("transient RPC error")
            return super().neuron_for_uid(uid=uid, netuid=netuid)

    neurons = {2: _FakeNeuron(stake=50.0, weights=[(3, 1.0)])}
    sub = _PartiallyBrokenSub(metagraph, neurons)
    out = get_peer_consensus_top_uids(
        config=_config(), wallet=_wallet("me_hk"), subtensor=sub, top_n=3,
    )
    assert out == [3]

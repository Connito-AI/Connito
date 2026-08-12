"""Observer mode: a validator that shares a hotkey with a live one.

The contract is that `CONNITO_VALIDATOR_OBSERVER=1` closes every channel
through which a second process could act under the shared identity — chain
extrinsics here, the DHT in `run.py`. These tests pin the chain half plus the
telemetry label; the DHT skip is a branch in `run()` and is verified against a
running container instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from connito.validator.chain_submitter import (
    OBSERVER_ENV,
    ChainSubmitter,
    observer_mode_enabled,
)


@pytest.fixture
def observer_env(monkeypatch):
    monkeypatch.setenv(OBSERVER_ENV, "1")


@pytest.fixture
def submitter(observer_env):
    """A ChainSubmitter built in observer mode.

    `bittensor.AsyncSubtensor` and `AsyncRunner` are patched to blow up if
    touched: constructing either would mean the gate ran too late.
    """
    config = SimpleNamespace(
        chain=SimpleNamespace(lite_network="finney", network="finney"),
    )
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5Observer"

    with (
        patch("connito.validator.chain_submitter.bittensor.AsyncSubtensor") as subtensor,
        patch("connito.validator.chain_submitter.AsyncRunner") as runner,
    ):
        made = ChainSubmitter(config, wallet)
        assert not subtensor.called, "observer built an AsyncSubtensor"
        assert not runner.called, "observer started an AsyncRunner thread"
    return made


def test_observer_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv(OBSERVER_ENV, raising=False)
    assert observer_mode_enabled() is False


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "1 "])
def test_only_exactly_one_enables_observer_mode(monkeypatch, value):
    """Fail closed the other way round: a typo'd value must not silently
    disable submission on a validator that is supposed to be live."""
    monkeypatch.setenv(OBSERVER_ENV, value)
    assert observer_mode_enabled() is False


def test_commit_is_suppressed(submitter):
    with patch("connito.validator.chain_submitter.acommit_status") as commit:
        future = submitter.async_commit(SimpleNamespace())
    assert future.result() is False
    assert not commit.called


def test_weight_submission_is_suppressed(submitter):
    round_obj = SimpleNamespace(round_id=7, weights_submitted=False)
    with patch("connito.validator.chain_submitter.submit_weights_async") as submit:
        future = submitter.async_submit_weight(round_obj, {0: 1.0})
    assert future.result() is False
    assert not submit.called
    assert round_obj.weights_submitted is False


def test_fallback_weight_submission_is_suppressed(submitter):
    with patch("connito.validator.chain_submitter._asubmit_fallback_weights") as fallback:
        future = submitter.async_submit_fallback_weights()
    assert future.result() is False
    assert not fallback.called


def test_stop_is_safe_without_a_runner(submitter):
    """`run()` calls stop() in both its normal and its error path, and an
    observer never built a runner to stop."""
    submitter.stop()


def test_suppressed_futures_are_resolved(submitter):
    """The call sites never await these, so an unresolved Future would be
    collected with an exception attached and logged at GC. Resolved-on-return
    keeps the suppressed path silent."""
    for future in (
        submitter.async_commit(SimpleNamespace()),
        submitter.async_submit_weight(SimpleNamespace(round_id=1), {}),
        submitter.async_submit_fallback_weights(),
    ):
        assert future.done()
        assert future.exception() is None


def test_telemetry_distinguishes_observer_from_the_hotkey_it_shares():
    """hotkey and uid are identical between the two processes by design, so
    the label is the only thing telling the two scrapes apart."""
    from connito.shared.telemetry import CONNITO_VALIDATOR_INFO, set_validator_identity

    shared = dict(hotkey="5Shared", uid=0, version="v1", netuid=102)

    set_validator_identity(**shared, observer=True)
    assert CONNITO_VALIDATOR_INFO._value["observer"] == "1"

    set_validator_identity(**shared)
    assert CONNITO_VALIDATOR_INFO._value["observer"] == "0"


def test_live_validator_still_submits(monkeypatch):
    """The gate must not leak into a normal deployment."""
    monkeypatch.delenv(OBSERVER_ENV, raising=False)
    config = SimpleNamespace(
        chain=SimpleNamespace(lite_network="finney", network="finney"),
    )
    wallet = MagicMock()

    with (
        patch("connito.validator.chain_submitter.bittensor.AsyncSubtensor"),
        patch("connito.validator.chain_submitter.AsyncRunner") as runner,
    ):
        live = ChainSubmitter(config, wallet)
        assert live.observer is False
        live.async_commit(SimpleNamespace())

    assert runner.return_value.submit.called
    # The runner is a mock, so the coroutine it was handed is never awaited.
    runner.return_value.submit.call_args.args[0].close()

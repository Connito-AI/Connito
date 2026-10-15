"""A head read the endpoint cannot serve must not cost the validator its round or its exit code.

Cycle 17728: every `get_all_commitments(block=None)` page hit an archive
backend missing the head header, the error reached `run()`, and the process
exited 0 with the round lost. No network: the chain is a fake.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from connito.shared import chain as chain_mod
from connito.shared.telemetry import (
    VALIDATOR_ROUND_FREEZE_FAILURES_TOTAL,
    VALIDATOR_ROUND_FREEZE_RETRIES_TOTAL,
    VALIDATOR_ROUNDS_SKIPPED_TOTAL,
)
from connito.validator import round as round_mod

HEAD = 5000
ARCHIVE = "wss://archive.example:443"

try:
    from async_substrate_interface.errors import SubstrateRequestException as _SRE
except Exception:  # pragma: no cover
    _SRE = RuntimeError


def _unknown_block() -> Exception:
    return _SRE(
        "Cannot resolve a block range [0xab ... Some(0xcd)]. "
        "UnknownBlock: Header was not found in the database: 0xcd"
    )


class _FakeSubtensor:
    """Records the block every read asks for; fails the first `fail_reads` commitment reads."""

    def __init__(self, endpoint: str, *, shared: dict) -> None:
        self.chain_endpoint = endpoint
        self.shared = shared
        self.closed = False

    def get_current_block(self) -> int:
        return HEAD

    def get_all_commitments(self, *, netuid, block):
        self.shared["commit_blocks"].append(block)
        if self.shared["fail_reads"] > 0:
            self.shared["fail_reads"] -= 1
            raise _unknown_block()
        return {}

    def metagraph(self, *, netuid, block):
        self.shared["metagraph_blocks"].append(block)
        return SimpleNamespace(hotkeys=[], neurons=[])

    def close(self) -> None:
        self.closed = True


def _config(lite_network: str = "finney") -> SimpleNamespace:
    return SimpleNamespace(
        chain=SimpleNamespace(netuid=102, network="archive", lite_network=lite_network),
        cycle=SimpleNamespace(cycle_length=100),
    )


def _read(fail_reads: int, *, block=None, lite_network="finney"):
    """Run get_chain_commits against a fake chain; return what it touched."""
    shared = {"fail_reads": fail_reads, "commit_blocks": [], "metagraph_blocks": []}
    caller = _FakeSubtensor(ARCHIVE, shared=shared)
    created: list[_FakeSubtensor] = []

    def _new(network):
        sub = _FakeSubtensor(network, shared=shared)
        created.append(sub)
        return sub

    with patch.object(chain_mod.bittensor, "Subtensor", side_effect=_new), \
            patch.object(chain_mod.time, "sleep"), \
            patch("connito.shared.cycle.get_validator_whitelist_from_api", return_value=set()):
        result = chain_mod.get_chain_commits(_config(lite_network), caller, block=block)
    return SimpleNamespace(result=result, caller=caller, created=created, **shared)


def test_head_read_pins_one_block_for_commitments_and_metagraph():
    out = _read(0)
    assert out.commit_blocks == [HEAD]
    assert out.metagraph_blocks == [HEAD]
    assert out.created == []


def test_unknown_block_steps_back_and_reconnects_to_the_callers_endpoint():
    out = _read(2)
    assert out.commit_blocks == [HEAD, HEAD - 1, HEAD - 2]
    # The metagraph is read at the block whose commitments succeeded.
    assert out.metagraph_blocks == [HEAD - 2]
    assert [s.chain_endpoint for s in out.created] == [ARCHIVE, ARCHIVE]
    # Every subtensor this function made is closed; the caller's is not.
    assert all(s.closed for s in out.created)
    assert out.caller.closed is False


def test_step_back_stops_after_two_blocks_and_head_read_falls_back_to_lite():
    out = _read(4)
    assert out.commit_blocks == [HEAD, HEAD - 1, HEAD - 2, HEAD - 2, HEAD - 2]
    assert out.metagraph_blocks == [HEAD - 2]
    assert [s.chain_endpoint for s in out.created] == [ARCHIVE, ARCHIVE, "finney", "finney"]


def test_historical_read_never_uses_lite_or_steps_back():
    out = _read(4, block=4200)
    assert out.commit_blocks == [4200] * 5
    assert out.metagraph_blocks == [4200]
    assert {s.chain_endpoint for s in out.created} == {ARCHIVE}


def test_lite_fallback_skipped_when_lite_is_the_network():
    out = _read(4, lite_network="archive")
    assert {s.chain_endpoint for s in out.created} == {ARCHIVE}


def test_read_that_never_recovers_raises_after_retries():
    with pytest.raises(Exception, match="UnknownBlock"):
        _read(100)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _freeze(fail_times: int, *, deadline_s: float):
    """Drive freeze_before_deadline with a Round.freeze failing `fail_times` times."""
    clock = _Clock()
    calls = {"freeze": 0, "metagraph": 0}

    def fake_freeze(**kwargs):
        calls["freeze"] += 1
        if calls["freeze"] <= fail_times:
            raise _unknown_block()
        return SimpleNamespace(round_id=kwargs["round_id"], metagraph=kwargs["metagraph"])

    def fetch_metagraph():
        calls["metagraph"] += 1
        return f"metagraph-{calls['metagraph']}"

    with patch.object(round_mod.Round, "freeze", side_effect=fake_freeze), \
            patch.object(round_mod.time, "monotonic", clock.monotonic), \
            patch.object(round_mod.time, "sleep", clock.sleep):
        frozen = round_mod.freeze_before_deadline(
            fetch_metagraph, deadline=clock.now + deadline_s, round_id=17728,
        )
    return frozen, calls


def _counts():
    return tuple(c._value.get() for c in (
        VALIDATOR_ROUND_FREEZE_RETRIES_TOTAL,
        VALIDATOR_ROUND_FREEZE_FAILURES_TOTAL,
        VALIDATOR_ROUNDS_SKIPPED_TOTAL,
    ))


def test_freeze_retries_inside_submission_and_refetches_the_metagraph():
    before = _counts()
    frozen, calls = _freeze(3, deadline_s=600)
    assert frozen.new_round.round_id == 17728
    # The round was built from the metagraph fetched on the attempt that succeeded.
    assert frozen.metagraph == "metagraph-4" == frozen.new_round.metagraph
    assert calls == {"freeze": 4, "metagraph": 4}
    retries, failures, skipped = (a - b for a, b in zip(_counts(), before, strict=True))
    assert (retries, failures, skipped) == (3, 0, 0)


def test_freeze_that_never_recovers_skips_the_round_at_the_deadline():
    before = _counts()
    frozen, calls = _freeze(10_000, deadline_s=60)
    assert frozen is None
    # 60 s of 12 s backoffs: the deadline ends it, not an attempt cap.
    assert calls["freeze"] == 6
    retries, failures, skipped = (a - b for a, b in zip(_counts(), before, strict=True))
    assert (retries, failures, skipped) == (5, 1, 1)


def test_freeze_counters_never_raise_into_the_caller():
    broken = SimpleNamespace(inc=lambda: (_ for _ in ()).throw(RuntimeError("registry")))
    with patch.object(round_mod, "VALIDATOR_ROUND_FREEZE_RETRIES_TOTAL", broken), \
            patch.object(round_mod, "VALIDATOR_ROUND_FREEZE_FAILURES_TOTAL", broken), \
            patch.object(round_mod, "VALIDATOR_ROUNDS_SKIPPED_TOTAL", broken):
        frozen, _ = _freeze(10_000, deadline_s=30)
    assert frozen is None


def _run_tree() -> ast.FunctionDef:
    from connito.validator import run as run_mod

    return ast.parse(textwrap.dedent(inspect.getsource(run_mod.run))).body[0]


def test_skipped_freeze_returns_to_the_loop_top():
    skip = next(
        node for node in ast.walk(_run_tree())
        if isinstance(node, ast.If) and ast.unparse(node.test) == "frozen is None"
    )
    assert isinstance(skip.body[-1], ast.Continue)
    assert "round_ref.current = None" in ast.unparse(skip)


def test_unexpected_error_in_the_loop_exits_non_zero():
    handler = next(
        h for node in ast.walk(_run_tree()) if isinstance(node, ast.Try)
        for h in node.handlers if h.type is not None and ast.unparse(h.type) == "Exception"
        and any("Quit training" in ast.unparse(s) for s in h.body)
    )
    assert isinstance(handler.body[-1], ast.Raise) and handler.body[-1].exc is None

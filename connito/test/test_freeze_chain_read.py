"""A head read the endpoint cannot serve must not cost the validator its round or its exit code.

Cycle 17728: every `get_all_commitments(block=None)` page hit an archive
backend missing the head header, the error reached `run()`, and the process
exited 0 with the round lost. No network: the chain is a fake.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from connito.shared import chain as chain_mod
from connito.shared.cycle import PhaseNames
from connito.shared.telemetry import (
    VALIDATOR_ROUND_FREEZE_RETRIES_TOTAL,
    VALIDATOR_ROUNDS_SKIPPED_TOTAL,
)
from connito.validator import round as round_mod
from connito.validator import run as run_mod

HEAD = 5000
PIN = HEAD - 1  # one behind head, so a lagging backend still has the header
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

    def determine_block_hash(self, block):
        self.shared["hash_blocks"].append(block)
        if self.shared["null_hashes"] > 0:
            self.shared["null_hashes"] -= 1
            return None
        return f"0x{block:x}"

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


def _read(fail_reads: int, *, block=None, lite_network="finney", null_hashes=0, deadline=None):
    """Run get_chain_commits against a fake chain; return what it touched."""
    shared = {"fail_reads": fail_reads, "null_hashes": null_hashes,
              "commit_blocks": [], "metagraph_blocks": [], "hash_blocks": []}
    caller = _FakeSubtensor(ARCHIVE, shared=shared)
    created: list[_FakeSubtensor] = []

    def _new(network):
        sub = _FakeSubtensor(network, shared=shared)
        created.append(sub)
        return sub

    with patch.object(chain_mod.bittensor, "Subtensor", side_effect=_new), \
            patch.object(chain_mod.time, "sleep"), \
            patch("connito.shared.cycle.get_validator_whitelist_from_api", return_value=set()):
        result = chain_mod.get_chain_commits(_config(lite_network), caller, block=block, deadline=deadline)
    return SimpleNamespace(result=result, caller=caller, created=created, **shared)


def test_head_read_pins_one_block_for_commitments_and_metagraph():
    out = _read(0)
    assert out.commit_blocks == [PIN]
    assert out.metagraph_blocks == [PIN]
    assert out.created == []


def test_null_block_hash_is_not_read_as_head():
    # A backend without the block answers null, and bittensor would then query head.
    out = _read(0, null_hashes=1)
    assert out.hash_blocks == [PIN, PIN - 1]
    assert out.commit_blocks == [PIN - 1]
    assert out.metagraph_blocks == [PIN - 1]
    assert [s.chain_endpoint for s in out.created] == [ARCHIVE]


def test_unknown_block_steps_back_and_reconnects_to_the_callers_endpoint():
    out = _read(2)
    assert out.commit_blocks == [PIN, PIN - 1, PIN - 2]
    # The metagraph is read at the block whose commitments succeeded.
    assert out.metagraph_blocks == [PIN - 2]
    assert [s.chain_endpoint for s in out.created] == [ARCHIVE, ARCHIVE]
    # Every subtensor this function made is closed; the caller's is not.
    assert all(s.closed for s in out.created)
    assert out.caller.closed is False


def test_step_back_stops_after_two_blocks_and_head_read_falls_back_to_lite():
    out = _read(4)
    assert out.commit_blocks == [PIN, PIN - 1, PIN - 2, PIN - 2, PIN - 2]
    assert out.metagraph_blocks == [PIN - 2]
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


def test_passed_deadline_stops_retries():
    with pytest.raises(Exception, match="UnknownBlock"):
        _read(1, deadline=0.0)
    with patch.object(chain_mod.time, "monotonic", return_value=0.0):
        out = _read(1, deadline=1.0)
    assert out.commit_blocks == [PIN, PIN - 1]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _freeze(fail_times: int, *, deadline_s: float, freeze_takes_s: float = 0.0):
    """Drive freeze_before_deadline with a Round.freeze failing `fail_times` times."""
    clock = _Clock()
    calls = {"freeze": 0, "metagraph": 0, "deadlines": set()}

    def fake_freeze(**kwargs):
        calls["freeze"] += 1
        calls["deadlines"].add(kwargs["deadline"])
        clock.now += freeze_takes_s
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
    return tuple(c._value.get() for c in (VALIDATOR_ROUND_FREEZE_RETRIES_TOTAL, VALIDATOR_ROUNDS_SKIPPED_TOTAL))


def _delta(before):
    return tuple(a - b for a, b in zip(_counts(), before, strict=True))


def test_freeze_retries_inside_submission_and_refetches_the_metagraph():
    before = _counts()
    frozen, calls = _freeze(3, deadline_s=600)
    assert frozen.new_round.round_id == 17728
    # The round was built from the metagraph fetched on the attempt that succeeded.
    assert frozen.metagraph == "metagraph-4" == frozen.new_round.metagraph
    assert (calls["freeze"], calls["metagraph"]) == (4, 4)
    # Each attempt's chain reads stop retrying at the same deadline.
    assert calls["deadlines"] == {1600.0}
    assert _delta(before) == (3, 0)


def test_freeze_that_never_recovers_skips_the_round_at_the_deadline():
    before = _counts()
    frozen, calls = _freeze(10_000, deadline_s=60)
    assert frozen is None
    # 60 s of 12 s backoffs: the deadline ends it, not an attempt cap.
    assert calls["freeze"] == 6
    assert _delta(before) == (5, 1)


def test_freeze_finishing_after_the_deadline_skips_the_round():
    before = _counts()
    frozen, _ = _freeze(0, deadline_s=60, freeze_takes_s=61)
    assert frozen is None
    assert _delta(before) == (0, 1)


def test_freeze_counters_never_raise_into_the_caller():
    broken = SimpleNamespace(inc=lambda: (_ for _ in ()).throw(RuntimeError("registry")))
    with patch.object(round_mod, "VALIDATOR_ROUND_FREEZE_RETRIES_TOTAL", broken), \
            patch.object(round_mod, "VALIDATOR_ROUNDS_SKIPPED_TOTAL", broken):
        frozen, _ = _freeze(10_000, deadline_s=30)
    assert frozen is None


@pytest.fixture
def driver(tmp_path):
    """run() with every collaborator before and inside the loop stubbed; `wait_till` is scripted per test."""
    config = MagicMock()
    for name in ("base_checkpoint_path", "checkpoint_path", "miner_submission_path"):
        setattr(config.ckpt, name, str(tmp_path / name))
    config.log.base_metric_path = str(tmp_path / "metrics")
    config.run.record_cuda_mem_history = False
    config.evaluation.background_worker_enabled = False
    config.evaluation.enable_round_group_construction = False
    config.cycle.cycle_length = 100
    lite = MagicMock()
    lite.get_current_block.return_value = 0
    lite.metagraph.return_value.last_update.__getitem__.return_value.item.return_value = 0
    refs: list[round_mod.RoundRef] = []

    def _round_ref():
        refs.append(round_mod.RoundRef())
        return refs[-1]

    stubs = {name: MagicMock() for name in (
        "TelemetryManager", "resolve_telemetry_port", "validate_hf_distribution_config", "ChainSubmitter",
        "MetricLogger", "get_base_tokenizer", "setup_training", "MinerScoreAggregator", "SystemStatePoller",
        "PhaseManager", "get_blocks_from_previous_phase_from_api", "_sync_lite_metagraph", "set_validator_identity",
        "ValidatorChainCommit", "SignedModelHashChainCommit", "ModelCheckpoint", "_round_commit", "cleanup",
        "check_phase_expired", "prune_submissions_outside_window", "finalize_round_scores", "get_status",
        "wait_till", "freeze_before_deadline", "build_submission_uid_weights",
    )}
    stubs.update(
        setup_chain_worker=MagicMock(return_value=(MagicMock(), MagicMock(), lite)),
        resolve_score_path=MagicMock(return_value=tmp_path / "scores.json"),
        prune_miner_submission_files=MagicMock(return_value=[]),
        resolve_hf_repo_ids=MagicMock(return_value=(None, "repo")),
        observer_mode_enabled=MagicMock(return_value=True),
        _maybe_switch_task=MagicMock(return_value=False),
        RoundRef=_round_ref,
    )
    with patch.multiple(run_mod, **stubs), \
            patch("connito.validator.distribute.publish_round_baseline"), \
            patch("connito.validator.distribute.publish_round_podium"):
        yield SimpleNamespace(run=lambda: run_mod.run(0, 1, config), refs=refs, lite=lite, **stubs)


def _script(*, stop_at: tuple[int, str]):
    """wait_till for a run of cycles; raises KeyboardInterrupt at (cycle, phase).

    `state["passed"]` holds phases the chain is already past this cycle; a wait for
    one of them would sleep to the next cycle, so it is recorded as an overrun.
    """
    state = {"cycle": 0, "seen": [], "passed": set(), "overruns": []}

    def wait_till(config, phase, block_offset=0):
        if phase == PhaseNames.miner_commit_1 and block_offset == -5:
            state["cycle"] += 1
            state["passed"] = set()
        state["seen"].append((state["cycle"], phase))
        if phase in state["passed"]:
            state["overruns"].append((state["cycle"], phase))
        if (state["cycle"], phase) == stop_at:
            raise KeyboardInterrupt
        start = 1000 * state["cycle"]
        return SimpleNamespace(phase_start_block=start, phase_end_block=start + 10, cycle_length=100,
                               cycle_index=state["cycle"], blocks_remaining_in_phase=10)
    return wait_till, state


def _frozen(round_id: int, **fields) -> round_mod.FrozenRound:
    return round_mod.FrozenRound(MagicMock(round_id=round_id, **fields), MagicMock(hotkeys=[]))


def test_skip_that_overran_into_merge_goes_straight_to_validator_commit(driver):
    wait_till, state = _script(stop_at=(2, PhaseNames.validate))
    driver.wait_till.side_effect = wait_till
    cycle_two = _frozen(2000)

    def freeze(*args, **kwargs):
        if state["cycle"] == 1:
            state["passed"] = {PhaseNames.validate, PhaseNames.merge}
            return None
        return cycle_two

    driver.freeze_before_deadline.side_effect = freeze
    with pytest.raises(KeyboardInterrupt):
        driver.run()

    assert state["overruns"] == []
    cycle_one = [phase for cycle, phase in state["seen"] if cycle == 1]
    assert cycle_one[cycle_one.index(PhaseNames.submission) + 1:] == [
        PhaseNames.validator_commit_1, PhaseNames.validator_commit_2,
        PhaseNames.train, PhaseNames.miner_commit_1,
    ]
    driver.SignedModelHashChainCommit.assert_called_once()
    driver._maybe_switch_task.assert_called_once()
    assert driver.freeze_before_deadline.call_count == 2
    assert driver.refs[0].current is cycle_two.new_round


def test_skipped_round_is_not_refinalized_and_stale_weights_fall_back(driver):
    wait_till, state = _script(stop_at=(3, PhaseNames.validate))
    driver.wait_till.side_effect = wait_till
    round_a = _frozen(1000, weights_submitted=False)
    driver.freeze_before_deadline.side_effect = [round_a, None, _frozen(3000)]
    calls: list[tuple[str, int]] = []
    driver.finalize_round_scores.side_effect = lambda **kw: calls.append(("finalize", state["cycle"]))
    submitter = driver.ChainSubmitter.return_value
    submitter.async_submit_weight.side_effect = lambda *a: calls.append(("round_weights", state["cycle"]))
    submitter.async_submit_fallback_weights.side_effect = lambda: calls.append(("fallback", state["cycle"]))
    # last_update stays 0: weights are fresh until cycle 3, when weight_age 1000 > cycle_length 100.
    driver.lite.get_current_block.side_effect = lambda: 1000 if state["cycle"] >= 3 else 0
    with pytest.raises(KeyboardInterrupt):
        driver.run()

    assert driver.freeze_before_deadline.call_count == 3
    assert driver.finalize_round_scores.call_args.kwargs["round_obj"] is round_a.new_round
    assert submitter.async_submit_weight.call_args.args[0] is round_a.new_round
    # Round A is finalized and weighted once; after the skip, stale weights take the
    # fallback (equal split plus owner share), never a re-run of round A.
    assert calls == [("finalize", 2), ("round_weights", 2), ("fallback", 3)]


def test_unexpected_error_in_the_loop_exits_non_zero(driver):
    driver.wait_till.side_effect = RuntimeError("cycle-api down")
    with pytest.raises(RuntimeError, match="cycle-api down"):
        driver.run()
    driver.ChainSubmitter.return_value.stop.assert_called_once()

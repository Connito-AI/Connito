"""The round score, task gain minus the c4 and gate-usage penalties, and the c4 isolation gate beneath it.

CPU only, `evaluate_model` stubbed: `python -m pytest connito/test/test_replay_isolation_score.py`.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from prometheus_client import REGISTRY

from connito.shared.config import DatasetSourceCfg, EvalCfg
from connito.shared.evaluate import EvalDeadlineExceeded
from connito.validator import background_eval_worker as bew
from connito.validator import evaluator
from connito.validator.evaluator import MinerEvalJob, evaluate_one_miner_sync, usage_penalties
from connito.validator.round import select_baseline_uid

TASK, REPLAY = ["task batch"], ["replay batch"]
BASELINE, BASELINE_REPLAY, TOLERANCE, TARGET = 3.0, 2.0, 0.25, 0.25


@pytest.fixture
def statuses(monkeypatch) -> dict[int, str | None]:
    seen: dict[int, str | None] = {}
    monkeypatch.setattr(evaluator, "set_miner_eval_status", lambda uid, reason: seen.__setitem__(uid, reason))
    return seen


def _config(**evaluation) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=EvalCfg(**{"max_replay_loss_increase": TOLERANCE, "task_usage_target": TARGET, **evaluation}),
        dataloader=SimpleNamespace(world_size=1),
    )


def _score(monkeypatch, *, task_loss: float, replay_loss: float, replay_batches=REPLAY,
           baseline_replay: float | None = BASELINE_REPLAY, replay_nan_batches: int = 0,
           task_usage: float = TARGET, replay_usage: float = 0.0, base_raw: float = 0.0,
           **evaluation) -> tuple[MinerEvalJob, list]:
    """Score uid 7 against BASELINE; by default its gate usage costs nothing, so only the losses count."""
    calls: list = []

    def _evaluate(step, model, batches, *a, **kw):
        calls.append(batches)
        if batches is TASK:
            return {"val_loss": task_loss, "nan_batches": 0, "gate_usage": task_usage}
        return {"val_loss": replay_loss, "nan_batches": replay_nan_batches, "gate_usage": replay_usage}

    monkeypatch.setattr(evaluator, "load_model_from_path", lambda *a: MagicMock())
    monkeypatch.setattr(evaluator, "track_eval_latency", lambda: lambda fn: fn)  # the histogram is process-global
    monkeypatch.setattr(evaluator, "evaluate_model", _evaluate)
    job = evaluate_one_miner_sync(
        config=_config(**evaluation),
        model_path="m.safetensors", uid=7, hotkey="hotkey", base_model=MagicMock(), tokenizer=None,
        combined_seed="s", device=torch.device("cpu"), baseline_loss=BASELINE, step=1,
        cached_batches=TASK, replay_batches=replay_batches, baseline_replay_loss=baseline_replay,
        base_raw_score=base_raw,
    )
    return job, calls


def _replay_increase_gauge() -> float:
    return evaluator.VALIDATOR_MINER_REPLAY_LOSS_INCREASE.labels(miner_uid="7")._value.get()


# --- the score ------------------------------------------------------------------------------------


@pytest.mark.parametrize("change, score", [
    ({"task_loss": 2.5}, 0.5),  # task loss down: gain
    ({"task_loss": 3.25}, -0.25),  # task loss up: loss
    ({"replay_loss": BASELINE_REPLAY + 0.125}, 0.375),  # c4 loss up: penalty
    ({"replay_loss": BASELINE_REPLAY - 0.5}, 0.5),  # c4 loss down: no reward
    ({"task_usage": TARGET + 0.25}, 0.25),  # gate above the target on task tokens: penalty
    ({"task_usage": TARGET - 0.125}, 0.375),  # below it: the same penalty
    ({"replay_usage": 0.125}, 0.375),  # gate open on c4 tokens: penalty
])
def test_each_term_moves_the_score_with_its_sign(monkeypatch, statuses, change, score):
    job, _ = _score(monkeypatch, **{"task_loss": 2.5, "replay_loss": BASELINE_REPLAY, **change})

    assert job.score == pytest.approx(score)
    assert statuses[7] is None


def test_the_weights_and_target_are_locked_validator_fields():
    locked = EvalCfg().locked_defaults()
    assert {k: locked[k] for k in ("replay_loss_weight", "task_usage_weight", "task_usage_target",
                                   "replay_usage_weight")} == {
        "replay_loss_weight": 1.0, "task_usage_weight": 1.0, "task_usage_target": 0.2, "replay_usage_weight": 1.0,
    }


def test_the_weights_and_target_come_from_the_validator_config(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.125, task_usage=0.25,
                    replay_usage=0.0625, replay_loss_weight=2.0, task_usage_weight=3.0, task_usage_target=0.5,
                    replay_usage_weight=4.0)

    assert job.score == pytest.approx(0.5 - 2 * 0.125 - 3 * 0.25 - 4 * 0.0625)


BASE_USAGE = {"task": 0.05, "replay": 0.05}  # an untrained gate: about 0.05 on Qwen


def _base_raw(**evaluation) -> float:
    return -sum(usage_penalties(_config(**evaluation).evaluation, BASE_USAGE).values())


def test_the_bases_raw_score_is_its_gate_usage_terms():
    # |0.05 − 0.25| + 0.05, and with weights 2 and 3: 2·0.2 + 3·0.05.
    assert _base_raw() == pytest.approx(-0.25)
    assert _base_raw(task_usage_weight=2.0, replay_usage_weight=3.0) == pytest.approx(-0.55)


def test_a_miner_identical_to_the_base_scores_exactly_zero_like_the_base(monkeypatch, statuses):
    """The base's score is 0 by construction, and only a miner above it is published."""
    job, _ = _score(monkeypatch, task_loss=BASELINE, replay_loss=BASELINE_REPLAY, task_usage=BASE_USAGE["task"],
                    replay_usage=BASE_USAGE["replay"], base_raw=_base_raw())

    assert job.score == 0.0
    assert select_baseline_uid({7: job.score}, {}) is None


def test_the_score_is_the_raw_score_minus_the_bases(monkeypatch, statuses):
    log = MagicMock()
    monkeypatch.setattr(evaluator, "logger", log)
    # 0.1 of task gain with the base's usage: raw 0.1 − 0.25 is negative, the score 0.1 is positive.
    job, _ = _score(monkeypatch, task_loss=BASELINE - 0.1, replay_loss=BASELINE_REPLAY,
                    task_usage=BASE_USAGE["task"], replay_usage=BASE_USAGE["replay"], base_raw=_base_raw())

    (complete,) = (c.kwargs for c in log.info.call_args_list if c.args[0] == "evaluate_one_miner: complete")
    assert complete["raw"] == pytest.approx(-0.15)
    assert complete["base_raw"] == pytest.approx(-0.25)
    assert job.score == pytest.approx(0.1)
    assert select_baseline_uid({7: job.score}, {}) == 7  # published although its raw score is negative


def test_a_gate_failure_stays_minus_inf_whatever_the_bases_raw_score(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.5, base_raw=-10.0)

    assert job.score == -math.inf


def test_each_term_is_logged_and_published(monkeypatch, statuses):
    log = MagicMock()
    monkeypatch.setattr(evaluator, "logger", log)
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.125, task_usage=0.5,
                    replay_usage=0.125)

    terms = {"task_gain": 0.5, "replay_penalty": 0.125, "usage_penalty": 0.25, "replay_usage_penalty": 0.125}
    (complete,) = (c.kwargs for c in log.info.call_args_list if c.args[0] == "evaluate_one_miner: complete")
    assert {k: complete[k] for k in (*terms, "task_usage", "replay_usage", "score")} == {
        **terms, "task_usage": 0.5, "replay_usage": 0.125, "score": 0.0,
    }
    for term, value in terms.items():
        assert evaluator.VALIDATOR_MINER_SCORE_TERM.labels(miner_uid="7", term=term)._value.get() == value
    for split, value in {"task": 0.5, "replay": 0.125}.items():
        assert evaluator.VALIDATOR_MINER_GATE_USAGE.labels(miner_uid="7", split=split)._value.get() == value
    assert _replay_increase_gauge() == 0.125


def test_a_non_finite_gate_usage_scores_minus_inf(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY, task_usage=math.nan)

    assert job.score == -math.inf
    assert statuses[7] == "non_finite_loss"


# --- the c4 isolation gate ------------------------------------------------------------------------


def test_a_replay_loss_past_the_tolerance_is_never_ranked(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.5)

    assert (job.val_loss, job.score) == (math.inf, -math.inf)
    assert statuses[7] == "replay_loss_increase"
    assert _replay_increase_gauge() == 0.5


def test_a_nan_replay_increase_fails_the_gate(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=math.nan)

    assert job.score == -math.inf
    assert statuses[7] == "replay_loss_increase"


def test_a_replay_loss_exactly_at_the_tolerance_passes(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + TOLERANCE)

    assert job.val_loss == 2.5
    assert job.score == pytest.approx(0.5 - TOLERANCE)


def test_replay_batches_that_blow_up_fail_the_gate_even_when_the_rest_improve(monkeypatch, statuses):
    """Base replay batches 3.0 / 3.0 / 2.0; the miner's inf / inf / 2.6 averages 2.6 over the finite one."""
    job, _ = _score(monkeypatch, task_loss=2.0, replay_loss=2.6, baseline_replay=8.0 / 3.0, replay_nan_batches=2)

    assert job.score == -math.inf  # 1.0 if the dropped batches went unnoticed
    assert statuses[7] == "replay_loss_increase"


@pytest.mark.parametrize("replay_loss, nan_batches, reason", [
    (BASELINE_REPLAY + 0.5, 0, "increase"), (BASELINE_REPLAY, 1, "non_finite_batches"),
])
def test_the_gate_log_says_why_it_tripped(monkeypatch, statuses, replay_loss, nan_batches, reason):
    log = MagicMock()
    monkeypatch.setattr(evaluator, "logger", log)
    _score(monkeypatch, task_loss=2.5, replay_loss=replay_loss, replay_nan_batches=nan_batches)

    event = "evaluate_one_miner: c4 isolation gate failed; not ranked"
    assert [c.kwargs["reason"] for c in log.info.call_args_list if c.args[0] == event] == [reason]


def test_a_non_finite_replay_loss_is_never_ranked(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=math.inf)

    assert job.score == -math.inf


@pytest.mark.parametrize("replay_batches, baseline_replay", [(None, BASELINE_REPLAY), (REPLAY, None)])
def test_without_a_replay_split_or_baseline_the_gate_and_c4_terms_are_off(
    monkeypatch, statuses, replay_batches, baseline_replay,
):
    log = MagicMock()
    monkeypatch.setattr(evaluator, "logger", log)
    job, calls = _score(monkeypatch, task_loss=2.5, replay_loss=99.0, replay_usage=1.0,
                        replay_batches=replay_batches, baseline_replay=baseline_replay)

    assert calls == [TASK]
    assert (job.val_loss, job.score) == (2.5, 0.5)
    (complete,) = (c.kwargs for c in log.info.call_args_list if c.args[0] == "evaluate_one_miner: complete")
    assert not {"replay_penalty", "replay_usage_penalty", "replay_usage"} & complete.keys()


def test_a_miner_scored_without_a_c4_eval_loses_its_earlier_replay_series(monkeypatch, statuses):
    _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.125, replay_usage=0.125)
    _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY, replay_batches=None)

    def sample(name: str, **labels: str) -> float | None:
        return REGISTRY.get_sample_value(name, {"miner_uid": "7", **labels})

    assert sample("validator_miner_score_term", term="replay_penalty") is None
    assert sample("validator_miner_score_term", term="replay_usage_penalty") is None
    assert sample("validator_miner_gate_usage", split="replay") is None
    assert sample("validator_miner_score_term", term="task_gain") == 0.5
    assert sample("validator_miner_gate_usage", split="task") == TARGET


def test_the_replay_eval_honours_the_miners_deadline(monkeypatch, statuses):
    """A replay eval past the deadline unwinds like a task eval does, and the miner is not scored."""
    def _evaluate(step, model, batches, *a, deadline_monotonic=None, **kw):
        if batches is REPLAY and deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            raise EvalDeadlineExceeded("past the deadline")
        return {"val_loss": 2.5, "nan_batches": 0}

    monkeypatch.setattr(evaluator, "load_model_from_path", lambda *a: MagicMock())
    monkeypatch.setattr(evaluator, "track_eval_latency", lambda: lambda fn: fn)
    monkeypatch.setattr(evaluator, "evaluate_model", _evaluate)
    job = evaluate_one_miner_sync(
        config=_config(),
        model_path="m.safetensors", uid=7, hotkey="hotkey", base_model=MagicMock(), tokenizer=None,
        combined_seed="s", device=torch.device("cpu"), baseline_loss=BASELINE, step=1,
        deadline_monotonic=0.0, cached_batches=TASK, replay_batches=REPLAY, baseline_replay_loss=BASELINE_REPLAY,
    )

    assert job is None


def test_without_cached_batches_the_task_loader_leaves_out_replay(monkeypatch):
    seen: list = []
    monkeypatch.setattr(evaluator, "get_dataloader", lambda **kw: seen.append(kw["replay"]) or TASK)
    monkeypatch.setattr(evaluator, "load_model_from_path", lambda *a: MagicMock())
    monkeypatch.setattr(evaluator, "track_eval_latency", lambda: lambda fn: fn)
    monkeypatch.setattr(evaluator, "evaluate_model", lambda *a, **kw: {"val_loss": 2.5, "nan_batches": 0})
    job = evaluate_one_miner_sync(
        config=_config(),
        model_path="m.safetensors", uid=7, hotkey="hotkey", base_model=MagicMock(), tokenizer=None,
        combined_seed="s", device=torch.device("cpu"), baseline_loss=BASELINE, step=1,
    )

    assert seen == [False]
    assert job is not None


# --- the worker measures both baselines once per round ---------------------------------------------


def _row(length: int) -> dict[str, torch.Tensor]:
    mask = torch.tensor([[1] * length + [0] * (8 - length)])
    return {"input_ids": mask.clone(), "attention_mask": mask, "labels": mask.clone()}


@pytest.fixture
def worker_env(monkeypatch) -> SimpleNamespace:
    env = SimpleNamespace(loaders=[], evaluated=[], miner_kwargs=[], replay_rows=[_row(5), _row(3)],
                          task_baseline=BASELINE, replay_baseline=BASELINE_REPLAY, replay_nan_batches=0,
                          logger=MagicMock())

    def _get_dataloader(**kw):
        env.loaders.append(kw["replay"])
        return env.replay_rows if kw["replay"] else [TASK]

    def _evaluate(step, model, batches, *a, **kw):
        env.evaluated.append(batches)
        if batches == [TASK]:
            return {"val_loss": env.task_baseline, "nan_batches": 0, "gate_usage": 0.05}
        if isinstance(env.replay_baseline, Exception):
            raise env.replay_baseline
        return {"val_loss": env.replay_baseline, "nan_batches": env.replay_nan_batches, "gate_usage": 0.0625}

    def _evaluate_one_miner(**kw):
        env.miner_kwargs.append(kw)
        return MinerEvalJob(uid=kw["uid"], hotkey=kw["hotkey"], model_path="m", step=1, score=0.0, val_loss=1.0)

    monkeypatch.setattr("connito.shared.dataloader.get_dataloader", _get_dataloader)
    monkeypatch.setattr("connito.shared.evaluate.evaluate_model", _evaluate)
    monkeypatch.setattr(bew, "residual_shapes", lambda config: {})
    monkeypatch.setattr(bew, "load_state_dict_from_path", lambda path: {})
    monkeypatch.setattr(bew, "evaluate_one_miner_sync", _evaluate_one_miner)
    monkeypatch.setattr(bew, "logger", env.logger)
    monkeypatch.setattr(evaluator, "validate_miner_submission", lambda **kw: None)
    return env


def _worker(*, replay_source: bool = True) -> bew.BackgroundEvalWorker:
    sources = [DatasetSourceCfg(path="pmc"), DatasetSourceCfg(path="allenai/c4", replay=replay_source)]
    worker = bew.BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=EvalCfg(per_miner_eval_timeout_sec=5),
            dataloader=SimpleNamespace(world_size=1),
            task=SimpleNamespace(exp=SimpleNamespace(data=SimpleNamespace(dataset_sources=sources, sequence_length=8))),
        ),
        round_ref=MagicMock(), device=torch.device("cpu"), tokenizer=MagicMock(),
        merge_phase_active=threading.Event(), eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(),
    )
    worker.set_eval_base_model(torch.nn.Linear(2, 2))
    return worker


def _round(round_id: int = 100) -> MagicMock:
    round_obj = MagicMock(round_id=round_id, seed="s", base_shard="base.safetensors", baseline_fallback=False)
    round_obj.claim_for_eval.return_value = True
    round_obj.pop_downloaded.return_value = "m.safetensors"
    return round_obj


def _run_round(worker, round_obj, env, miners: int = 1) -> list[dict]:
    """Load the round and score `miners` miners; returns what each miner's eval was handed."""
    asyncio.run(worker._load_round_base(round_obj))
    start = len(env.miner_kwargs)
    for uid in range(1, miners + 1):
        asyncio.run(worker._evaluate_one(round_obj, uid=uid, hotkey=f"hotkey{uid}"))
    return env.miner_kwargs[start:]


def _warned(env, event: str) -> bool:
    return any(call.args and call.args[0] == event for call in env.logger.warning.call_args_list)


GATE_OFF = "bg-eval: task has no replay source; isolation gate off"


def test_both_baselines_are_measured_once_per_round_and_handed_to_every_miner(worker_env):
    first, second = _run_round(_worker(), _round(), worker_env, miners=2)

    assert worker_env.loaders == [False, True]
    assert len(worker_env.evaluated) == 2  # task then replay, never again for a miner
    assert first["replay_batches"] is second["replay_batches"] is worker_env.evaluated[1]
    for kw in (first, second):
        assert (kw["baseline_loss"], kw["baseline_replay_loss"]) == (BASELINE, BASELINE_REPLAY)


def _snapshot_log(env) -> dict:
    event = "bg-eval: round snapshot loaded"
    (loaded,) = (c.kwargs for c in env.logger.info.call_args_list if c.args[0] == event)
    return loaded


def test_the_bases_raw_score_is_measured_once_per_round_reported_and_handed_to_every_miner(worker_env):
    first, second = _run_round(_worker(), _round(), worker_env, miners=2)

    base_raw = -(abs(0.05 - 0.2) + 0.0625)  # the worker's base usage, and the default target and weights
    assert first["base_raw_score"] == second["base_raw_score"] == pytest.approx(base_raw)
    loaded = _snapshot_log(worker_env)
    assert (loaded["baseline_task_usage"], loaded["baseline_replay_usage"]) == (0.05, 0.0625)
    assert loaded["base_raw_score"] == pytest.approx(base_raw)
    assert REGISTRY.get_sample_value("validator_round_base_raw_score") == pytest.approx(base_raw)


def test_without_a_replay_source_the_bases_raw_score_has_no_c4_term(worker_env):
    (kw,) = _run_round(_worker(replay_source=False), _round(), worker_env)

    assert kw["base_raw_score"] == pytest.approx(-0.15)


def test_the_replay_split_is_batched_within_the_tasks_sequence_length(worker_env):
    (kw,) = _run_round(_worker(), _round(), worker_env)

    # Rows of 5 and 3 real tokens cannot share one 8-position pass.
    assert [int(b["attention_mask"].sum()) for b in kw["replay_batches"]] == [3, 5]


def test_a_task_without_a_replay_source_turns_the_gate_off(worker_env):
    round_obj = _round()
    (kw,) = _run_round(_worker(replay_source=False), round_obj, worker_env)

    assert round_obj.baseline_fallback is False  # the round still scores and publishes normally
    assert worker_env.loaders == [False]
    assert (kw["replay_batches"], kw["baseline_replay_loss"]) == (None, None)
    assert _warned(worker_env, GATE_OFF)


def test_a_round_never_inherits_the_previous_rounds_replay_baseline(worker_env):
    worker = _worker()
    _run_round(worker, _round(100), worker_env)
    worker.config.task.exp.data.dataset_sources[1].replay = False  # the next task has no replay source
    (kw,) = _run_round(worker, _round(101), worker_env)

    assert kw["baseline_replay_loss"] is None


def test_a_fallback_round_never_inherits_the_previous_rounds_base_raw_score(worker_env):
    worker = _worker()
    _run_round(worker, _round(100), worker_env)
    worker_env.replay_baseline = RuntimeError("oom")  # the next round falls back after measuring the task baseline
    (kw,) = _run_round(worker, _round(101), worker_env)

    assert kw["base_raw_score"] == 0.0


def _fallback_counter():
    from connito.shared.telemetry import ERRORS_TOTAL

    return ERRORS_TOTAL.labels(component="bg_eval", kind="baseline_fallback")


def test_a_round_on_the_fallback_baseline_is_flagged_and_counted(worker_env, monkeypatch):
    def _broken(path):
        raise OSError("disk gone")

    monkeypatch.setattr(bew, "load_state_dict_from_path", _broken)
    before = _fallback_counter()._value.get()
    round_obj = _round()
    (kw,) = _run_round(_worker(), round_obj, worker_env)

    assert round_obj.baseline_fallback is True
    assert (kw["baseline_loss"], kw["baseline_replay_loss"], kw["base_raw_score"]) == (100.0, None, 0.0)
    assert _fallback_counter()._value.get() == before + 1


@pytest.mark.parametrize("unusable", [
    {"task_baseline": math.inf},               # every score would be non-finite
    {"replay_baseline": RuntimeError("oom")},  # from here on, the c4 check cannot run this round
    {"replay_rows": []},
    {"replay_baseline": math.inf},
    {"replay_nan_batches": 1},
], ids=["task-baseline-inf", "replay-baseline-raises", "empty-c4-split", "replay-baseline-inf", "replay-nan-batches"])
def test_an_unusable_baseline_takes_the_fallback(worker_env, unusable):
    for name, value in unusable.items():
        setattr(worker_env, name, value)
    round_obj = _round()
    (kw,) = _run_round(_worker(), round_obj, worker_env)

    assert round_obj.baseline_fallback is True
    assert (kw["baseline_loss"], kw["base_raw_score"]) == (100.0, 0.0)  # even when the task baseline was measured

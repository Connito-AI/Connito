"""The c4 isolation gate: a miner whose replay loss rises past the locked tolerance scores 0.

CPU only, `evaluate_model` stubbed: `python -m pytest connito/test/test_replay_isolation_score.py`.
"""

from __future__ import annotations

import asyncio
import math
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from connito.shared.config import DatasetSourceCfg
from connito.validator import background_eval_worker as bew
from connito.validator import evaluator
from connito.validator.evaluator import MinerEvalJob, evaluate_one_miner_sync

TASK, REPLAY = ["task batch"], ["replay batch"]
BASELINE, BASELINE_REPLAY, TOLERANCE = 3.0, 2.0, 0.25


@pytest.fixture
def statuses(monkeypatch) -> dict[int, str | None]:
    seen: dict[int, str | None] = {}
    monkeypatch.setattr(evaluator, "set_miner_eval_status", lambda uid, reason: seen.__setitem__(uid, reason))
    return seen


def _score(monkeypatch, *, task_loss: float, replay_loss: float, replay_batches=REPLAY,
           baseline_replay: float | None = BASELINE_REPLAY, replay_nan_batches: int = 0) -> tuple[MinerEvalJob, list]:
    calls: list = []

    def _evaluate(step, model, batches, *a, **kw):
        calls.append(batches)
        if batches is TASK:
            return {"val_loss": task_loss, "nan_batches": 0}
        return {"val_loss": replay_loss, "nan_batches": replay_nan_batches}

    monkeypatch.setattr(evaluator, "load_model_from_path", lambda *a: MagicMock())
    monkeypatch.setattr(evaluator, "track_eval_latency", lambda: lambda fn: fn)  # the histogram is process-global
    monkeypatch.setattr(evaluator, "evaluate_model", _evaluate)
    job = evaluate_one_miner_sync(
        config=SimpleNamespace(evaluation=SimpleNamespace(max_replay_loss_increase=TOLERANCE)),
        model_path="m.safetensors", uid=7, hotkey="hotkey", base_model=MagicMock(), tokenizer=None,
        combined_seed="s", device=torch.device("cpu"), baseline_loss=BASELINE, step=1,
        cached_batches=TASK, replay_batches=replay_batches, baseline_replay_loss=baseline_replay,
    )
    return job, calls


def _replay_increase_gauge() -> float:
    return evaluator.VALIDATOR_MINER_REPLAY_LOSS_INCREASE.labels(miner_uid="7")._value.get()


def test_a_replay_loss_past_the_tolerance_scores_zero(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.5)

    assert (job.val_loss, job.score) == (math.inf, 0.0)
    assert statuses[7] == "replay_loss_increase"
    assert _replay_increase_gauge() == 0.5


def test_a_replay_loss_within_the_tolerance_keeps_the_task_score(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + 0.125)

    assert job.val_loss == 2.5
    assert job.score == pytest.approx(0.5 ** 1.2)
    assert statuses[7] is None


def test_a_replay_loss_exactly_at_the_tolerance_passes(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=BASELINE_REPLAY + TOLERANCE)

    assert job.val_loss == 2.5


def test_replay_batches_that_blow_up_fail_the_gate_even_when_the_rest_improve(monkeypatch, statuses):
    """Base replay batches 3.0 / 3.0 / 2.0; the miner's inf / inf / 2.6 averages 2.6 over the finite one."""
    job, _ = _score(monkeypatch, task_loss=2.0, replay_loss=2.6, baseline_replay=8.0 / 3.0, replay_nan_batches=2)

    assert job.score == 0.0  # 1.0 if the dropped batches went unnoticed
    assert statuses[7] == "replay_loss_increase"


def test_a_non_finite_replay_loss_scores_zero(monkeypatch, statuses):
    job, _ = _score(monkeypatch, task_loss=2.5, replay_loss=math.inf)

    assert job.score == 0.0


@pytest.mark.parametrize("replay_batches, baseline_replay", [(None, BASELINE_REPLAY), (REPLAY, None)])
def test_without_a_replay_split_or_baseline_the_gate_is_off(monkeypatch, statuses, replay_batches, baseline_replay):
    job, calls = _score(monkeypatch, task_loss=2.5, replay_loss=99.0,
                        replay_batches=replay_batches, baseline_replay=baseline_replay)

    assert calls == [TASK]
    assert job.val_loss == 2.5


# --- the worker measures both baselines once per round ---------------------------------------------


def _row(length: int) -> dict[str, torch.Tensor]:
    mask = torch.tensor([[1] * length + [0] * (8 - length)])
    return {"input_ids": mask.clone(), "attention_mask": mask, "labels": mask.clone()}


@pytest.fixture
def worker_env(monkeypatch) -> SimpleNamespace:
    env = SimpleNamespace(loaders=[], evaluated=[], replay_baseline=BASELINE_REPLAY, miner_kwargs=[])

    def _get_dataloader(**kw):
        env.loaders.append(kw["replay"])
        return [_row(5), _row(3)] if kw["replay"] else [TASK]

    def _evaluate(step, model, batches, *a, **kw):
        env.evaluated.append(batches)
        return {"val_loss": BASELINE if batches == [TASK] else env.replay_baseline}

    def _evaluate_one_miner(**kw):
        env.miner_kwargs.append(kw)
        return MinerEvalJob(uid=kw["uid"], hotkey=kw["hotkey"], model_path="m", step=1, score=0.0, val_loss=1.0)

    monkeypatch.setattr("connito.shared.dataloader.get_dataloader", _get_dataloader)
    monkeypatch.setattr("connito.shared.evaluate.evaluate_model", _evaluate)
    monkeypatch.setattr(bew, "residual_shapes", lambda config: {})
    monkeypatch.setattr(bew, "load_state_dict_from_path", lambda path: {})
    monkeypatch.setattr(bew, "evaluate_one_miner_sync", _evaluate_one_miner)
    monkeypatch.setattr(evaluator, "validate_miner_submission", lambda **kw: None)
    return env


def _worker(*, replay_source: bool = True) -> bew.BackgroundEvalWorker:
    sources = [DatasetSourceCfg(path="pmc"), DatasetSourceCfg(path="allenai/c4", replay=replay_source)]
    worker = bew.BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=5, top_k_miners_to_reward=3),
            dataloader=SimpleNamespace(world_size=1),
            task=SimpleNamespace(exp=SimpleNamespace(data=SimpleNamespace(dataset_sources=sources, sequence_length=8))),
        ),
        round_ref=MagicMock(), device=torch.device("cpu"), tokenizer=MagicMock(),
        merge_phase_active=threading.Event(), eval_window_active=threading.Event(),
        gpu_eval_lock=threading.Lock(),
    )
    worker.set_eval_base_model(torch.nn.Linear(2, 2))
    return worker


def _round() -> MagicMock:
    round_obj = MagicMock(round_id=100, seed="s", base_shard="base.safetensors")
    round_obj.claim_for_eval.return_value = True
    round_obj.pop_downloaded.return_value = "m.safetensors"
    return round_obj


def test_both_baselines_are_measured_once_per_round_and_handed_to_every_miner(worker_env):
    worker, round_obj = _worker(), _round()
    asyncio.run(worker._load_round_base(round_obj))
    for uid in (1, 2):
        asyncio.run(worker._evaluate_one(round_obj, uid=uid, hotkey=f"hotkey{uid}"))

    assert worker_env.loaders == [False, True]
    assert len(worker_env.evaluated) == 2  # task then replay, never again for a miner
    assert worker_env.evaluated[1] is worker._replay_batches
    assert (worker._loaded_baseline_loss, worker._loaded_baseline_replay_loss) == (BASELINE, BASELINE_REPLAY)
    for kw in worker_env.miner_kwargs:
        assert kw["replay_batches"] is worker._replay_batches
        assert kw["baseline_replay_loss"] == BASELINE_REPLAY


def test_a_task_without_a_replay_source_turns_the_gate_off(worker_env):
    worker = _worker(replay_source=False)
    asyncio.run(worker._load_round_base(_round()))

    assert worker_env.loaders == [False]
    assert (worker._replay_batches, worker._loaded_baseline_replay_loss) == (None, None)


def test_a_non_finite_replay_baseline_turns_the_gate_off(worker_env):
    worker_env.replay_baseline = math.inf
    worker = _worker()
    asyncio.run(worker._load_round_base(_round()))

    assert worker._loaded_baseline_loss == BASELINE
    assert worker._loaded_baseline_replay_loss is None


def test_a_round_on_the_fallback_baseline_is_flagged_and_counted(worker_env, monkeypatch):
    from connito.shared.telemetry import ERRORS_TOTAL

    def _broken(path):
        raise OSError("disk gone")

    monkeypatch.setattr(bew, "load_state_dict_from_path", _broken)
    counter = ERRORS_TOTAL.labels(component="bg_eval", kind="baseline_fallback")
    before = counter._value.get()
    worker, round_obj = _worker(), _round()
    asyncio.run(worker._load_round_base(round_obj))

    assert round_obj.baseline_fallback is True
    assert worker._loaded_baseline_loss == 100.0
    assert counter._value.get() == before + 1

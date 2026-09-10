"""`BackgroundEvalWorker` must follow the active task's expert group.

Captured at construction, so a task change would leave the worker validating
against the group it started on — score=0 for every miner in the round, not a
degraded score. Hence the one rule the setter enforces: not mid-window.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from connito.validator.background_eval_worker import BackgroundEvalWorker

# Two groups that name different experts, mirroring the shape of
# `expert_group_assignment`: {group_id: {layer_id: [(my_idx, org_idx)]}}.
GROUP_A = {0: {0: [(0, 0)]}}
GROUP_B = {4: {0: [(1, 1)]}}


def _worker(eval_window: threading.Event) -> BackgroundEvalWorker:
    return BackgroundEvalWorker(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(per_miner_eval_timeout_sec=1.0),
        ),
        round_ref=MagicMock(),
        device=torch.device("cpu"),
        tokenizer=MagicMock(),
        merge_phase_active=threading.Event(),
        eval_window_active=eval_window,
        gpu_eval_lock=threading.Lock(),
        expert_group_assignment=GROUP_A,
    )


def test_swap_takes_effect_when_the_window_is_closed():
    worker = _worker(threading.Event())
    worker.set_expert_group_assignment(GROUP_B)
    assert worker._expert_group_assignment == GROUP_B


def test_swap_is_refused_while_the_eval_window_is_open():
    window = threading.Event()
    window.set()
    worker = _worker(window)

    with pytest.raises(RuntimeError, match="eval window"):
        worker.set_expert_group_assignment(GROUP_B)


def test_a_refused_swap_leaves_the_previous_assignment_intact():
    """A setter that mutated before raising would pass the test above."""
    window = threading.Event()
    window.set()
    worker = _worker(window)

    with pytest.raises(RuntimeError):
        worker.set_expert_group_assignment(GROUP_B)

    assert worker._expert_group_assignment == GROUP_A


def test_validation_uses_the_swapped_assignment(monkeypatch, tmp_path):
    """The read site must follow the swap, not a construction-time copy.

    Drives the real `_evaluate_one` path far enough to reach
    `validate_miner_submission` and captures what it was handed.
    """
    worker = _worker(threading.Event())
    worker.set_expert_group_assignment(GROUP_B)
    monkeypatch.setattr(worker, "_prune_non_top", lambda round_obj: None)

    seen = {}

    def _capture(*, round_obj, uid, model_path, expert_group_assignment):
        seen["assignment"] = expert_group_assignment
        return "rejected"  # non-None short-circuits before the GPU eval

    monkeypatch.setattr(
        "connito.validator.evaluator.validate_miner_submission", _capture
    )

    round_obj = MagicMock()
    round_obj.round_id = 100
    round_obj.claim_for_eval.return_value = True
    round_obj.pop_downloaded.return_value = tmp_path / "model.safetensors"

    asyncio.run(worker._evaluate_one(round_obj, uid=1, hotkey="hk_abcdef"))

    assert seen["assignment"] == GROUP_B

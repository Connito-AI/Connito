"""The miner's I/O process reads the task from config per job, so a switch
that lands on config at Distribute is what the next download job fetches.

Run with `python -m pytest connito/test/test_miner_io_follows_switch.py`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import yaml

from connito.miner import model_io
from connito.shared.config import MinerConfig

SHIPPED = "exp_pmc_residual"
TARGET = "exp_switch_target"


def _task_dir(root: Path, name: str, group_id: int, k: int) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    body["residual"]["k"] = k
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))


@pytest.fixture
def config(tmp_path: Path) -> MinerConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4, k=1)
    _task_dir(tmp_path, TARGET, group_id=7, k=2)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    return MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)


def test_each_download_job_fetches_the_group_and_spec_config_has_now(config, monkeypatch) -> None:
    """One worker, two jobs, and a switch landing on config between them, as it does between two Distributes."""
    asked: list = []

    def _fetch(current_model_meta, config_, subtensor, wallet, *, expert_group_ids, expected):
        asked.append((expert_group_ids, expected))
        if len(asked) == 1:
            config.switch_active_task(TARGET)
        return None  # nothing newer on chain -> FileNotReadyError path, logged

    monkeypatch.setattr(model_io, "fetch_model_from_chain_validator", _fetch)
    # The spec as a function of the task's K, so the test needs no checkpoint.
    monkeypatch.setattr(model_io, "residual_shapes", lambda config_: {"k": config_.task.exp.residual.k})
    monkeypatch.setattr(model_io, "select_best_checkpoint", lambda **kw: None)
    monkeypatch.setattr(model_io, "check_phase_expired", lambda *a, **k: None)
    queue: Queue = Queue()
    for _ in range(2):
        queue.put(model_io.Job(job_type=model_io.JobType.DOWNLOAD, phase_response=None))
    queue.put(None)  # poison pill

    model_io.download_worker(
        config, wallet=None, download_queue=queue, current_model_meta=None,
        current_model_hash="x", shared_state=SimpleNamespace(lock=threading.Lock()), subtensor=object(),
    )

    assert asked == [([4], {"k": 1}), ([7], {"k": 2})]

"""The miner's I/O process reads the task from config per job, so a switch
that lands on config at Distribute is what the next download job fetches.

Run with `python -m pytest connito/test/test_miner_io_follows_switch.py`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import yaml

from connito.miner import model_io
from connito.shared.config import MinerConfig

SHIPPED = "exp_nemotron_c4"
TARGET = "exp_switch_target"
HELPER = "exp_helper"


def _task_dir(root: Path, name: str, group_id: int, org_base: int) -> None:
    body = yaml.safe_load(Path(f"expert_groups/{SHIPPED}/config.yaml").read_text())
    body["group_id"] = group_id
    d = root / "expert_groups" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(body))
    (d / "expert_assignment.json").write_text(json.dumps({"0": [[0, org_base], [1, org_base + 1]]}))


@pytest.fixture
def config(tmp_path: Path) -> MinerConfig:
    _task_dir(tmp_path, SHIPPED, group_id=4, org_base=10)
    _task_dir(tmp_path, TARGET, group_id=7, org_base=20)
    _task_dir(tmp_path, HELPER, group_id=2, org_base=30)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"root_path": str(tmp_path)},
        "chain": {"hotkey_ss58": "test-hk", "coldkey_ss58": "test-ck", "uid": 0},
    }))
    return MinerConfig.from_path(cfg_path, active_task=None, auto_update_config=True)


def _run_one_download_job(config, monkeypatch) -> dict:
    """Push one DOWNLOAD job through `download_worker` and return what it
    asked the chain fetch for."""
    asked: dict = {}

    def _fetch(current_model_meta, config, subtensor, wallet, *, expert_group_ids, expert_group_assignment):
        asked.update(expert_group_ids=expert_group_ids, expert_group_assignment=expert_group_assignment)
        return None  # nothing newer on chain -> FileNotReadyError path, logged

    monkeypatch.setattr(model_io, "fetch_model_from_chain_validator", _fetch)
    monkeypatch.setattr(model_io, "select_best_checkpoint", lambda **kw: None)
    monkeypatch.setattr(model_io, "check_phase_expired", lambda *a, **k: None)

    queue: Queue = Queue()
    queue.put(model_io.Job(job_type=model_io.JobType.DOWNLOAD, phase_response=None))
    queue.put(None)  # poison pill
    model_io.download_worker(
        config, wallet=None, download_queue=queue, current_model_meta=None,
        current_model_hash="x", shared_state=SimpleNamespace(lock=threading.Lock()), subtensor=object(),
    )
    return asked


def test_the_download_job_fetches_the_group_config_has_now(config, monkeypatch) -> None:
    """The group id and its expert table are both read per job, so a switch
    between two Distributes changes what the next job asks for."""
    before = _run_one_download_job(config, monkeypatch)
    config.switch_active_task(TARGET)
    after = _run_one_download_job(config, monkeypatch)

    assert before["expert_group_ids"] == [4] and 4 in before["expert_group_assignment"]
    assert after["expert_group_ids"] == [7] and 7 in after["expert_group_assignment"]
    assert 4 not in after["expert_group_assignment"]


def test_the_download_dir_is_the_tasks_own_and_moves_with_it(config) -> None:
    """Two tasks' baseline caches never share a directory, so the checkpoint
    picker cannot find the previous task's shards after a switch."""
    before = config.ckpt.validator_checkpoint_path
    config.switch_active_task(TARGET)
    after = config.ckpt.validator_checkpoint_path

    assert before.name == SHIPPED and after.name == TARGET
    assert before.parent == after.parent
    assert after.is_dir()

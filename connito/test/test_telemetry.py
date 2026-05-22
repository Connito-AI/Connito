import math
import time
from types import SimpleNamespace

import requests
import torch
import torch.nn as nn
from prometheus_client import REGISTRY

from connito.shared.evaluate import evaluate_model, get_source_attribution
from connito.shared.telemetry import (
    MINER_VAL_LOSS_BATCH_HISTOGRAM,
    MINER_VAL_LOSS_BY_SOURCE,
    MINER_VAL_SCORED_BATCHES_BY_SOURCE,
    SUBNET_CURRENT_BLOCK,
    TelemetryManager,
    VALIDATOR_AVG_STEP_STATUS,
    VALIDATOR_MINER_SCORE,
    count_rpc_errors,
    track_eval_latency,
)

def test_telemetry_flow():
    print("1. Starting Prometheus Server...")
    # Use a custom port to avoid any collision for testing
    mgr = TelemetryManager()
    mgr.start_server(port=8111)

    print("2. Populating Gauges and Counters...")
    SUBNET_CURRENT_BLOCK.set(12345)
    
    # Check labelled metrics
    VALIDATOR_MINER_SCORE.labels(miner_uid="12").set(0.85)
    VALIDATOR_AVG_STEP_STATUS.labels(status="success").inc()
    VALIDATOR_AVG_STEP_STATUS.labels(status="success").inc()

    print("3. Testing Decorators...")
    @track_eval_latency()
    @count_rpc_errors()
    def fake_evaluation(fail=False):
        time.sleep(0.1)  # Simulate workload
        if fail:
            raise ConnectionError("Timeout hit")
            
    # Success path
    fake_evaluation(fail=False)
    
    # Error path
    try:
        fake_evaluation(fail=True)
    except ConnectionError:
        print("   Caught expected error for metric counting")

    print("4. Asserting HTTP Export...")
    # Give server a brief moment to stabilize
    time.sleep(0.5)
    
    resp = requests.get("http://localhost:8111/metrics")
    assert resp.status_code == 200, "Prometheus HTTP server not responding"
    
    content = resp.text
    
    assert 'subnet_current_block 12345.0' in content, "Missing block gauge"
    assert 'validator_miner_score{miner_uid="12"} 0.85' in content, "Missing labeled miner score"
    assert 'validator_avg_step_status_total{status="success"} 2.0' in content, "Missing labeled averager count"
    assert 'validator_eval_latency_seconds_count 2.0' in content, "Missing latency histogram count"
    
    if 'chain_rpc_errors_total_total 1.0' not in content and 'chain_rpc_errors_total 1.0' not in content:
        print("Received HTTP content:\n", content)
        assert False, "Missing exception counter"
    
    print("✅ All phase 1 tests passed successfully!")

class _ConstantLossModel(nn.Module):
    """Returns a fixed loss for every forward call so we can sanity-check
    `evaluate_model`'s per-source accumulators without needing real GPU
    or HF resources.
    """

    def __init__(self, device, loss: float = 2.0, aux_loss: float = 0.0):
        super().__init__()
        self.device = device
        self._loss = loss
        self._aux_loss = aux_loss

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(
            loss=torch.tensor(self._loss, device=self.device),
            aux_loss=torch.tensor(self._aux_loss, device=self.device),
        )


class _PerSourceLossModel(nn.Module):
    """Returns a different fixed loss depending on which batch index is
    being evaluated; used to verify per-source val_loss split is correct.
    """

    def __init__(self, device, loss_by_batch: list[float], aux_loss: float = 0.0):
        super().__init__()
        self.device = device
        self._loss_by_batch = loss_by_batch
        self._aux_loss = aux_loss
        self._call_idx = 0

    def forward(self, input_ids, attention_mask=None):
        loss_val = self._loss_by_batch[self._call_idx]
        self._call_idx += 1
        return SimpleNamespace(
            loss=torch.tensor(loss_val, device=self.device),
            aux_loss=torch.tensor(self._aux_loss, device=self.device),
        )


def _scrape_registry() -> str:
    """Render the default Prometheus registry as text — same path the
    HTTP exporter uses, but inline so we don't need a port-bound server.
    """
    from prometheus_client.exposition import generate_latest

    return generate_latest(REGISTRY).decode("utf-8")


def test_miner_val_loss_by_source_emits():
    """When `evaluate_model` is given a per-batch source_attribution map,
    it must populate the per-source Prometheus metrics for every distinct
    source observed.
    """
    device = torch.device("cpu")
    # 2 c4 batches @ loss=2.0, 2 nemotron @ loss=4.0.
    loss_by_batch = [2.0, 4.0, 2.0, 4.0]
    model = _PerSourceLossModel(device=device, loss_by_batch=loss_by_batch)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(4)]
    source_attribution = {0: "c4", 1: "nemotron", 2: "c4", 3: "nemotron"}

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
        source_attribution=source_attribution,
        expert_group="grp7",
        rank=3,
    )

    # Aggregate path unchanged: mean(2, 4, 2, 4) = 3.0.
    assert metrics["val_loss"] == 3.0
    assert metrics["scored_batches"] == 4
    assert metrics["nan_batches"] == 0

    # Per-source breakdown lives under `by_source`.
    assert "by_source" in metrics
    by_source = metrics["by_source"]
    assert set(by_source.keys()) == {"c4", "nemotron"}
    assert by_source["c4"]["val_loss"] == 2.0
    assert by_source["c4"]["scored_batches"] == 2
    assert by_source["nemotron"]["val_loss"] == 4.0
    assert by_source["nemotron"]["scored_batches"] == 2

    # Prometheus scrape confirms both source labels were emitted with
    # the propagated expert_group / rank labels.
    text = _scrape_registry()
    assert (
        'miner_val_loss_by_source{expert_group="grp7",rank="3",source="c4"} 2.0'
        in text
    ), "Missing per-source c4 gauge"
    assert (
        'miner_val_loss_by_source{expert_group="grp7",rank="3",source="nemotron"} 4.0'
        in text
    ), "Missing per-source nemotron gauge"
    assert (
        'miner_val_scored_batches_by_source{expert_group="grp7",rank="3",source="c4"} 2.0'
        in text
    ), "Missing scored-batches gauge for c4"
    assert (
        'miner_val_scored_batches_by_source{expert_group="grp7",rank="3",source="nemotron"} 2.0'
        in text
    ), "Missing scored-batches gauge for nemotron"
    # Histogram observes per-batch losses, so 2 c4 batches → count=2,
    # 2 nemotron batches → count=2. Distinct from the gauge, which
    # carries the per-source mean. This is the actionable distribution
    # for spotting tail-loss spikes within a source.
    assert (
        'miner_val_loss_batch_count{expert_group="grp7",rank="3",source="c4"} 2.0'
        in text
    ), "Histogram should record one observation per scored c4 batch"
    assert (
        'miner_val_loss_batch_count{expert_group="grp7",rank="3",source="nemotron"} 2.0'
        in text
    ), "Histogram should record one observation per scored nemotron batch"


def test_miner_val_loss_backwards_compat_no_source_attribution():
    """Legacy callers (validator's `_evaluate_on_fresh_loader_sync`) pass
    no `source_attribution`. Behavior must be unchanged: same return
    schema, no `by_source` key, no per-source metrics emitted.
    """
    device = torch.device("cpu")
    model = _ConstantLossModel(device=device, loss=2.5)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(3)]

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
    )

    assert metrics["val_loss"] == 2.5
    assert metrics["scored_batches"] == 3
    assert metrics["nan_batches"] == 0
    assert "by_source" not in metrics, (
        "Backwards-compat path leaked per-source data into the return dict"
    )


def test_miner_val_loss_per_source_nan_handling():
    """NaN batches must not pad a source's divisor. For the
    `nan_source`, every batch NaNs → its val_loss returns +inf and
    `scored_batches` stays at 0. The `clean_source` is unaffected.
    """
    device = torch.device("cpu")
    # c4: 2 NaN batches; nemotron: 2 finite @ loss=3.0.
    loss_by_batch = [float("nan"), 3.0, float("nan"), 3.0]
    model = _PerSourceLossModel(device=device, loss_by_batch=loss_by_batch)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(4)]
    source_attribution = {0: "c4", 1: "nemotron", 2: "c4", 3: "nemotron"}

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
        source_attribution=source_attribution,
        expert_group="grpnan",
        rank=0,
    )

    by_source = metrics["by_source"]
    # c4: 2 nan batches, 0 scored → val_loss=+inf so caller's
    # max(0, baseline - val_loss) clamps to 0 instead of giving the
    # miner a free win.
    assert math.isinf(by_source["c4"]["val_loss"])
    assert by_source["c4"]["scored_batches"] == 0
    assert by_source["c4"]["nan_batches"] == 2
    # nemotron clean.
    assert by_source["nemotron"]["val_loss"] == 3.0
    assert by_source["nemotron"]["scored_batches"] == 2
    assert by_source["nemotron"]["nan_batches"] == 0

    # Aggregate: 2 scored finite batches @ loss=3.0 → 3.0; 2 NaNs not in
    # divisor (per the existing exploit fix).
    assert metrics["val_loss"] == 3.0
    assert metrics["scored_batches"] == 2
    assert metrics["nan_batches"] == 2


def test_miner_val_loss_per_source_all_nan_returns_inf_aggregate():
    """100% NaN across every source must still hit the existing
    aggregate `+inf` path (preserving the NaN-exploit fix) and report
    `+inf` per-source as well.
    """
    device = torch.device("cpu")
    loss_by_batch = [float("nan"), float("nan"), float("nan"), float("nan")]
    model = _PerSourceLossModel(device=device, loss_by_batch=loss_by_batch)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(4)]
    source_attribution = {0: "c4", 1: "nemotron", 2: "c4", 3: "nemotron"}

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
        source_attribution=source_attribution,
    )

    assert math.isinf(metrics["val_loss"])
    assert metrics["scored_batches"] == 0
    by_source = metrics["by_source"]
    assert math.isinf(by_source["c4"]["val_loss"])
    assert math.isinf(by_source["nemotron"]["val_loss"])
    assert by_source["c4"]["scored_batches"] == 0
    assert by_source["nemotron"]["scored_batches"] == 0


def test_miner_val_loss_unknown_source_label_for_missing_attribution():
    """If `source_attribution` is provided but a particular batch_idx is
    not in the map, that batch's source label falls back to "unknown" —
    explicit catch-all so missing entries don't silently disappear.
    """
    device = torch.device("cpu")
    model = _ConstantLossModel(device=device, loss=2.0)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(3)]
    # Only batch 0 attributed; 1 and 2 should fall under "unknown".
    source_attribution = {0: "c4"}

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
        source_attribution=source_attribution,
    )

    by_source = metrics["by_source"]
    assert set(by_source.keys()) == {"c4", "unknown"}
    assert by_source["c4"]["scored_batches"] == 1
    assert by_source["unknown"]["scored_batches"] == 2


def test_get_source_attribution_returns_empty_without_source_names():
    """The helper must return an empty mapping when no source_names are
    provided — that signals the caller to fall back to two-pass eval.
    """
    attribution = get_source_attribution(dataloader=None, max_batches=10)
    assert attribution == {}


def test_get_source_attribution_round_robin():
    """When source_names are provided, the helper assigns batch indices
    round-robin across the supplied source list.
    """
    attribution = get_source_attribution(
        dataloader=None,
        max_batches=6,
        source_names=["c4", "nemotron"],
    )
    assert attribution == {
        0: "c4",
        1: "nemotron",
        2: "c4",
        3: "nemotron",
        4: "c4",
        5: "nemotron",
    }


if __name__ == "__main__":
    test_telemetry_flow()
    test_miner_val_loss_by_source_emits()
    test_miner_val_loss_backwards_compat_no_source_attribution()
    test_miner_val_loss_per_source_nan_handling()
    test_miner_val_loss_per_source_all_nan_returns_inf_aggregate()
    test_miner_val_loss_unknown_source_label_for_missing_attribution()
    test_get_source_attribution_returns_empty_without_source_names()
    test_get_source_attribution_round_robin()

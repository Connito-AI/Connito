"""Tests for the MoE `router_aux_loss_coef` lowering (1.0 -> 0.01) and the
routing-collapse early-warning telemetry that ships alongside it.

Context
-------
The validator scores miners with::

    val_loss = (loss_sum - aux_loss_sum) / scored_batches

(see ``connito/shared/evaluate.py:146``). The aux_loss term is *subtracted*
from val_loss, so any gradient capacity the model spends on the routing
balance loss is pure waste from a scoring standpoint. Dropping
``moe.router_aux_loss_coef`` from 1.0 to 0.01 lets the LM loss dominate
training while still keeping enough routing-balance signal to avoid total
collapse.

Collapse risk is detected via two histograms registered in
``connito.shared.telemetry``:

* ``moe_routing_entropy_distribution`` — per-layer routing entropy
* ``moe_expert_load_distribution`` — per-expert token-share

If routing entropy drops below
``MoECfg.router_collapse_alert_entropy_threshold`` (0.5 nats), operators
should roll the coefficient back toward 1.0.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from connito.shared.config import MoECfg
from connito.shared.evaluate import evaluate_model
from connito.shared.telemetry import (
    MOE_EXPERT_LOAD_HIST,
    MOE_ROUTING_ENTROPY_HIST,
    attach_moe_routing_telemetry_hooks,
    emit_moe_routing_telemetry,
)


def _histogram_sample_count(labeled_histogram) -> float:
    """Return total ``count`` for a labelled prometheus_client Histogram.

    ``Histogram._buckets`` holds non-cumulative per-bucket counts in this
    prometheus_client version (5.x): each ``MutexValue`` carries only the
    samples whose value fell into that exact bucket. The public ``_count``
    attribute does not exist, so we sum the non-cumulative bucket gauges
    to recover the total sample count without going through ``collect()``
    (which is O(n_metrics) and re-creates a new sample list each call).
    """
    return float(sum(bucket.get() for bucket in labeled_histogram._buckets))


# ----------------------------------------------------------------------
# Config defaults
# ----------------------------------------------------------------------


def test_router_aux_loss_coef_default_is_lowered() -> None:
    """The aux_loss coefficient must default to 0.01.

    Validator scoring (``connito/shared/evaluate.py:146``) subtracts
    ``aux_loss_sum`` from ``loss_sum``, so any aux_loss the model produces
    has zero direct effect on the miner's val_loss. The coefficient still
    influences *training* — at 1.0, ~half the gradient magnitude is spent
    keeping the router balanced. Dropping to 0.01 lets the LM loss
    dominate the gradient signal while keeping a token amount of balance
    pressure on the router so it does not collapse onto one expert.
    """
    cfg = MoECfg()
    assert cfg.router_aux_loss_coef == 0.01, (
        "router_aux_loss_coef must default to 0.01; validator scoring "
        "subtracts aux_loss from val_loss so we want the LM loss to "
        "dominate gradient updates"
    )


def test_router_collapse_alert_entropy_threshold_default() -> None:
    """Operator-facing alert threshold for router collapse must exist."""
    cfg = MoECfg()
    assert hasattr(cfg, "router_collapse_alert_entropy_threshold")
    threshold = cfg.router_collapse_alert_entropy_threshold
    assert isinstance(threshold, float)
    # 0.0 < threshold < ln(num_experts) for any reasonable expert count.
    assert 0.0 < threshold < math.log(MoECfg().num_experts)


def test_moe_cfg_field_types() -> None:
    """Pydantic v2 coerces; ensure both fields stay floats."""
    cfg = MoECfg()
    assert isinstance(cfg.router_aux_loss_coef, float)
    assert isinstance(cfg.router_collapse_alert_entropy_threshold, float)


# ----------------------------------------------------------------------
# Aux loss arithmetic at evaluation
# ----------------------------------------------------------------------


class _FixedAuxLossModel(nn.Module):
    """Minimal stand-in for an MoE causal-LM that returns a fixed LM loss
    and a fixed aux_loss per forward call. Used to verify the validator
    scoring math: LM loss must dominate when the aux loss is small.
    """

    def __init__(self, device: torch.device, lm_loss: float, aux_loss: float) -> None:
        super().__init__()
        self.device = device
        self._lm_loss = lm_loss
        self._aux_loss = aux_loss

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return SimpleNamespace(
            loss=torch.tensor(self._lm_loss, device=self.device),
            aux_loss=torch.tensor(self._aux_loss, device=self.device),
        )


def test_lm_loss_dominates_when_aux_loss_is_small() -> None:
    """At coef=0.01 the aux_loss magnitude bundled into outputs.loss is
    tiny, so val_loss = loss - aux_loss is approximately the pure LM loss.

    This mirrors what we expect after the coefficient drop: the validator
    subtracts aux_loss back out, so the remainder must equal the LM loss
    up to floating-point tolerance.
    """
    device = torch.device("cpu")
    # Simulate "loss = lm_loss + 0.01 * router_aux" returned by the model.
    raw_lm_loss = 2.5
    raw_aux_loss = 1.5
    coef = MoECfg().router_aux_loss_coef
    combined_loss = raw_lm_loss + coef * raw_aux_loss
    scaled_aux_loss = coef * raw_aux_loss
    model = _FixedAuxLossModel(
        device=device,
        lm_loss=combined_loss,
        aux_loss=scaled_aux_loss,
    )
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(3)]
    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
    )
    # val_loss = (combined - scaled_aux) per batch averaged
    assert metrics["val_loss"] == pytest.approx(raw_lm_loss, abs=1e-6)
    # val_aux_loss is just the mean aux_loss the validator observed.
    assert metrics["val_aux_loss"] == pytest.approx(scaled_aux_loss, abs=1e-6)


def test_high_aux_loss_coef_inflates_combined_loss() -> None:
    """Sanity check on the other direction: at the OLD coef of 1.0, the
    aux_loss term swamps the combined loss the model returns. The
    validator's subtraction recovers the LM loss either way, but during
    *training* this is wasted gradient magnitude that we are now reclaiming.
    """
    device = torch.device("cpu")
    raw_lm_loss = 2.5
    raw_aux_loss = 1.5
    old_coef = 1.0
    combined_loss_old = raw_lm_loss + old_coef * raw_aux_loss
    scaled_aux_loss_old = old_coef * raw_aux_loss
    new_coef = MoECfg().router_aux_loss_coef
    combined_loss_new = raw_lm_loss + new_coef * raw_aux_loss
    # At coef=1.0 the aux term contributes 1.5 to the combined loss.
    # At coef=0.01 it contributes 0.015 — two orders of magnitude less.
    assert combined_loss_old > combined_loss_new
    assert (combined_loss_old - raw_lm_loss) / (combined_loss_new - raw_lm_loss) == pytest.approx(
        old_coef / new_coef, rel=1e-6,
    )
    # And confirm the validator subtraction zeroes the aux contribution either way.
    model_old = _FixedAuxLossModel(
        device=device, lm_loss=combined_loss_old, aux_loss=scaled_aux_loss_old,
    )
    metrics_old = evaluate_model(
        step=0, model=model_old, eval_dataloader=[{"input_ids": torch.tensor([[1]])}],
        device=device, max_eval_batches=None,
    )
    assert metrics_old["val_loss"] == pytest.approx(raw_lm_loss, abs=1e-6)


# ----------------------------------------------------------------------
# Routing telemetry
# ----------------------------------------------------------------------


def test_routing_telemetry_metric_names_registered() -> None:
    """The two new histograms must be importable and carry the expected
    Prometheus names so dashboards / alerts can query them.
    """
    # `_name` is the prometheus_client base name; the actual exposed
    # metric appends `_bucket`, `_count`, `_sum`.
    assert MOE_ROUTING_ENTROPY_HIST._name == "moe_routing_entropy_distribution"
    assert MOE_EXPERT_LOAD_HIST._name == "moe_expert_load_distribution"
    # Label names match the dashboard schema agreed with the operator team.
    assert MOE_ROUTING_ENTROPY_HIST._labelnames == ("expert_group", "rank", "layer")
    assert MOE_EXPERT_LOAD_HIST._labelnames == ("expert_group", "rank", "expert_id")


class _FakeMoEBlock(nn.Module):
    """Minimal stand-in for ``CustomDeepseekV2Moe`` exposing the same
    routing surface area: a ``num_experts`` attribute and a
    ``route_tokens_to_experts(router_logits)`` method that returns
    ``(topk_idx, topk_weight)``.

    The capture hook in ``connito.shared.telemetry`` keys off the
    ``route_tokens_to_experts`` method name; using that exact name keeps
    this fake aligned with the production MoE block without pulling in
    the heavy DeepSeek-V2-Lite dependency tree.
    """

    def __init__(self, num_experts: int, topk_idx: torch.Tensor) -> None:
        super().__init__()
        self.num_experts = num_experts
        self._topk_idx = topk_idx

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        weights = torch.ones_like(self._topk_idx, dtype=torch.float32) / self._topk_idx.shape[-1]
        return self._topk_idx, weights


class _FakeMoEModel(nn.Module):
    """Wraps a few MoE blocks under ``model.layers.<i>.mlp`` so the
    layer-index regex in ``_iter_moe_blocks`` finds them. We hold the
    blocks via a nested ``layers`` ModuleList to mirror real production
    naming exactly (`model.layers.<i>.mlp`).
    """

    def __init__(self, blocks: list[_FakeMoEBlock]) -> None:
        super().__init__()
        # Wrap each MoE block in a tiny layer module that exposes an
        # ``mlp`` attribute, matching ``model.layers.<i>.mlp`` from the
        # real DeepSeek-V2-Lite layout.
        layers: list[nn.Module] = []
        for block in blocks:
            layer = nn.Module()
            layer.mlp = block
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self) -> None:  # pragma: no cover — never invoked in tests
        raise NotImplementedError


def _force_routing_observation(block: _FakeMoEBlock) -> None:
    """Invoke ``route_tokens_to_experts`` once so the forward hook fires
    and the capture attribute is populated.

    We pass a router_logits placeholder of the right shape; the fake
    block ignores its content but the hook only depends on the *return*
    tuple.
    """
    dummy_logits = torch.zeros(block._topk_idx.shape[0], block.num_experts)
    block.route_tokens_to_experts(dummy_logits)


def test_attach_hooks_and_emit_collapsed_routing() -> None:
    """When every token is routed to expert 0, the per-layer entropy is
    exactly 0.0 and the load distribution puts all mass on expert 0.

    The hook + emit pipeline must surface that signal so a Grafana alert
    keyed on ``moe_routing_entropy_distribution`` can fire.
    """
    num_tokens = 64
    num_experts = 8
    # Single-expert routing (all tokens land on expert 0) — total collapse.
    topk_idx = torch.zeros((num_tokens, 2), dtype=torch.long)
    block = _FakeMoEBlock(num_experts=num_experts, topk_idx=topk_idx)
    model = _FakeMoEModel([block])

    # 1) Attach hooks via the public helper.
    handles = attach_moe_routing_telemetry_hooks(model)
    assert handles, "Expected at least one hook to attach to a fake MoE block"

    # 2) Re-running attach is a no-op (idempotent) — same handle count
    # cannot stack on the same block.
    handles_again = attach_moe_routing_telemetry_hooks(model)
    assert handles_again == [], "attach_moe_routing_telemetry_hooks must be idempotent"

    # 3) Drive the routing call so the hook captures topk_idx.
    _force_routing_observation(block)

    # 4) Snapshot histogram counts, emit, and verify both histograms moved.
    entropy_metric = MOE_ROUTING_ENTROPY_HIST.labels(
        expert_group="exp_test_collapse", rank="0", layer="0",
    )
    load_metric_e0 = MOE_EXPERT_LOAD_HIST.labels(
        expert_group="exp_test_collapse", rank="0", expert_id="0",
    )
    load_metric_e1 = MOE_EXPERT_LOAD_HIST.labels(
        expert_group="exp_test_collapse", rank="0", expert_id="1",
    )
    before_entropy_sum = entropy_metric._sum.get()
    before_entropy_count = _histogram_sample_count(entropy_metric)
    before_e0_sum = load_metric_e0._sum.get()
    before_e1_sum = load_metric_e1._sum.get()

    emit_moe_routing_telemetry(
        model,
        expert_group="exp_test_collapse",
        rank=0,
    )

    after_entropy_sum = entropy_metric._sum.get()
    after_entropy_count = _histogram_sample_count(entropy_metric)
    after_e0_sum = load_metric_e0._sum.get()
    after_e1_sum = load_metric_e1._sum.get()

    # Entropy of a collapsed router is 0, so the histogram sum is
    # unchanged in value but the count went up.
    assert after_entropy_count - before_entropy_count == pytest.approx(1.0)
    # All mass on expert 0 → load_e0 sample is 1.0, load_e1 sample is 0.0.
    assert after_e0_sum - before_e0_sum == pytest.approx(1.0)
    assert after_e1_sum - before_e1_sum == pytest.approx(0.0)
    # The entropy observation itself is 0, so the sum delta is exactly 0.
    assert after_entropy_sum - before_entropy_sum == pytest.approx(0.0)

    # 5) Below-threshold guard: a collapsed router (entropy 0) must trip
    # the operator's collapse alert threshold (default 0.5).
    assert 0.0 < MoECfg().router_collapse_alert_entropy_threshold


def test_emit_uniform_routing_high_entropy() -> None:
    """When tokens spread uniformly across all experts, entropy
    approaches ln(num_experts) and per-expert load is 1/num_experts.

    Sanity check that the maths in ``emit_moe_routing_telemetry``
    matches the expected information-theoretic baseline.
    """
    num_experts = 8
    # Build a topk_idx that touches every expert equally: 64 tokens × 2
    # selections = 128 slots, 16 each across 8 experts.
    slots = torch.arange(128, dtype=torch.long) % num_experts
    topk_idx = slots.reshape(64, 2)
    block = _FakeMoEBlock(num_experts=num_experts, topk_idx=topk_idx)
    model = _FakeMoEModel([block])
    attach_moe_routing_telemetry_hooks(model)
    _force_routing_observation(block)

    entropy_metric = MOE_ROUTING_ENTROPY_HIST.labels(
        expert_group="exp_test_uniform", rank="0", layer="0",
    )
    load_metric = MOE_EXPERT_LOAD_HIST.labels(
        expert_group="exp_test_uniform", rank="0", expert_id="3",
    )
    before_entropy_sum = entropy_metric._sum.get()
    before_entropy_count = _histogram_sample_count(entropy_metric)
    before_load_sum = load_metric._sum.get()

    emit_moe_routing_telemetry(model, expert_group="exp_test_uniform", rank=0)

    after_entropy_count = _histogram_sample_count(entropy_metric)
    delta_entropy = entropy_metric._sum.get() - before_entropy_sum
    delta_load = load_metric._sum.get() - before_load_sum

    assert after_entropy_count - before_entropy_count == pytest.approx(1.0)
    # Uniform over 8 experts → entropy = ln(8) ≈ 2.0794
    assert delta_entropy == pytest.approx(math.log(num_experts), rel=1e-6)
    # Per-expert load is exactly 1/8.
    assert delta_load == pytest.approx(1.0 / num_experts, rel=1e-6)
    # And the routing-entropy value comfortably exceeds the collapse
    # threshold, so the alert would stay green.
    assert delta_entropy > MoECfg().router_collapse_alert_entropy_threshold


def test_emit_without_hooks_is_noop() -> None:
    """If hooks were never attached (or the block was just constructed)
    the helper must not raise — it must silently no-op so telemetry can
    never break training.
    """
    block = _FakeMoEBlock(
        num_experts=4,
        topk_idx=torch.zeros((2, 2), dtype=torch.long),
    )
    model = _FakeMoEModel([block])
    # Deliberately *do not* attach hooks. No prior observation exists.
    emit_moe_routing_telemetry(model, expert_group="exp_noop", rank=0)
    # Reaching this line at all means the helper swallowed the missing
    # observation. No assertion needed beyond the lack of exception.


def test_emit_on_model_without_moe_blocks_is_noop() -> None:
    """A model with no ``route_tokens_to_experts`` method anywhere must
    not raise either. Defends against accidentally invoking the helper on
    a non-MoE evaluator model.
    """

    class DenseOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    emit_moe_routing_telemetry(DenseOnly(), expert_group="exp_dense", rank=0)

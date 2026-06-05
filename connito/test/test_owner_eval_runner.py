"""Runner tests: gauge emission + failure isolation (needs prometheus_client)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("torch")


def _cfg(metrics):
    return SimpleNamespace(eval_pipeline=SimpleNamespace(enabled_metrics=metrics))


def test_run_eval_suite_emits_gauges_and_isolates_failures(monkeypatch):
    from connito.owner_eval import runner
    from connito.owner_eval.registry import REGISTRY
    from connito.shared import telemetry

    saved = dict(REGISTRY)
    REGISTRY.clear()
    try:
        class Good:
            name = "good"
            def evaluate(self, model, tokenizer, device, config):
                return {"good_x": 1.0, "good_y": 2.0}

        class Bad:
            name = "bad"
            def evaluate(self, model, tokenizer, device, config):
                raise RuntimeError("boom")

        REGISTRY["good"] = Good
        REGISTRY["bad"] = Bad

        results = runner.run_eval_suite(
            model=object(), tokenizer=object(), device="cpu",
            config=_cfg(["good", "bad"]),
            model_revision="globalver_7", cycle_index=5,
        )

        # good metrics returned and published
        assert results == {"good_x": 1.0, "good_y": 2.0}
        assert telemetry.OWNER_EVAL_METRIC.labels(metric="good_x")._value.get() == 1.0
        assert telemetry.OWNER_EVAL_METRIC.labels(metric="good_y")._value.get() == 2.0
        # status gauges reflect success/failure independently
        assert telemetry.OWNER_EVAL_STATUS.labels(metric="good")._value.get() == 1.0
        assert telemetry.OWNER_EVAL_STATUS.labels(metric="bad")._value.get() == 0.0
        # run info timestamp was stamped
        assert telemetry.OWNER_EVAL_LAST_RUN_TS._value.get() > 0
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)

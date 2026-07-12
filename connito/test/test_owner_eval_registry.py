"""Registry + cycle-gate tests (light deps: structlog only)."""

from types import SimpleNamespace

import pytest

from connito.owner_eval import registry
from connito.owner_eval.registry import register, build_enabled, REGISTRY
from connito.owner_eval.run import should_run_cycle


@pytest.fixture
def clean_registry():
    saved = dict(REGISTRY)
    REGISTRY.clear()
    try:
        yield REGISTRY
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


def _cfg(metrics):
    return SimpleNamespace(eval_pipeline=SimpleNamespace(enabled_metrics=metrics))


def test_register_populates_registry(clean_registry):
    @register
    class Foo:
        name = "foo"
        def evaluate(self, *a):  # pragma: no cover - not called
            return {}

    assert clean_registry["foo"] is Foo


def test_register_requires_name(clean_registry):
    with pytest.raises(ValueError):
        @register
        class NoName:
            name = ""


def test_register_rejects_duplicate(clean_registry):
    @register
    class A:
        name = "dup"
    with pytest.raises(ValueError):
        @register
        class B:
            name = "dup"


def test_build_enabled_order_and_skip_unknown(clean_registry):
    @register
    class M1:
        name = "m1"
    @register
    class M2:
        name = "m2"

    built = build_enabled(_cfg(["m2", "unknown", "m1"]))
    # preserves config order, drops unknown
    assert [type(e).__name__ for e in built] == ["M2", "M1"]


def test_builtin_metrics_register_on_import():
    # Importing the metrics package must register the three built-ins.
    pytest.importorskip("torch")
    import connito.owner_eval.metrics  # noqa: F401
    for name in ("gsm8k_ppl", "gsm8k_task", "mmlu"):
        assert name in REGISTRY


@pytest.mark.parametrize("cycle,interval,last,expected", [
    (5, 5, -1, True),
    (5, 5, 5, False),     # already ran this cycle
    (4, 5, -1, False),
    (10, 5, 5, True),
    (0, 5, -1, True),     # cycle 0 is on the boundary
    (-1, 5, -1, False),   # no phase / unknown
])
def test_should_run_cycle(cycle, interval, last, expected):
    assert should_run_cycle(cycle, interval, last) is expected

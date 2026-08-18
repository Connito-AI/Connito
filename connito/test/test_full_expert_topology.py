"""The validator's expert topology is selectable, and full routes like the base.

Two defects, both invisible because they sat behind each other:

  1. `validator/run.py` hard-coded `partial=True`, so the full branch of
     `mycelia.get_base_model` was unreachable from any config. `moe.partial_moe`
     was declared in `MoECfg` and read by nothing.
  2. Because nothing read it, `moe.full_topk` was never exercised. It sat at 2
     from the initial commit while DeepSeek-V2-Lite's own config — and the
     reference experiment's full-topology measurements — use 6.

Defect 2 only bites once defect 1 is fixed, which is exactly why it survived:
turning full experts on without this change would have scored top-2-of-64
against the experiment's top-6-of-64 and silently invalidated the comparison.

Run with `python -m pytest connito/test/test_full_expert_topology.py`.
"""

from __future__ import annotations

import pytest

from connito.shared.config import MoECfg


# ── the routing width of the full topology ───────────────────────────────────
def test_full_topk_matches_the_base_checkpoint():
    """DeepSeek-V2-Lite ships `num_experts_per_tok: 6`.

    The reference experiment loads that checkpoint through
    `AutoModelForCausalLM.from_pretrained` and never overrides the gate — its
    `partial_moe.py` reads `top_k = getattr(self.gate, "top_k")` — so its
    full-topology numbers are top-6-of-64. Ours have to be too, or losses are
    not comparable between the codebases.
    """
    assert MoECfg().full_topk == 6


def test_topology_fields_stay_locked():
    """Routing width has to be identical fleet-wide.

    Two validators scoring at different `topk` produce different `val_loss` for
    the same weights, which is a ranking disagreement dressed up as a
    measurement. The locked-field reset is what stops a hand-edited YAML from
    doing that.
    """
    locked = MoECfg._LOCKED_FIELDS
    assert {"partial_topk", "full_topk", "num_experts_per_tok"} <= locked


def test_partial_moe_is_locked_true():
    """It was deliberately unlocked while the full topology was opt-in.

    Locking it then would have made the full topology untestable:
    `check_and_prompt_locked` resets a locked field to its class default on
    load, so a staging host that set `partial_moe: false` would have been
    silently back on partial after one restart. It is locked now because
    `evaluation.full_topology_eval` is locked true, and
    `check_full_topology_eval_supported` rejects that alongside
    `partial_moe: false` — an unlocked half is a crash-loop waiting for the
    operator who edited it. Staging diverges via `--no-auto_update_config`.

    Locked *true* does not mean the fleet scores on partial: it means the
    model that merges stays partial. See `evaluation.full_topology_eval`.
    """
    assert "partial_moe" in MoECfg._LOCKED_FIELDS
    assert MoECfg.model_fields["partial_moe"].default is True


# ── the switch reaches the loader ────────────────────────────────────────────
class _Recorder:
    """Captures the `partial` flag `validator.run` hands to `load_model`."""

    def __init__(self) -> None:
        self.partial: bool | None = None

    def __call__(self, *args, **kwargs):
        self.partial = kwargs["partial"]
        raise _Stop


class _Stop(Exception):
    """Unwinds setup_training once the flag has been captured."""


@pytest.mark.parametrize(
    ("partial_moe", "expected_partial"),
    [(True, True), (False, False)],
    ids=["partial_moe=true -> partial", "partial_moe=false -> full"],
)
def test_config_selects_the_topology(monkeypatch, partial_moe, expected_partial):
    """The regression guard for defect 1.

    Asserted against the real call in `setup_training` rather than a
    reimplementation of it: the bug *was* a literal at that call site, so a
    test that recomputes `partial` from the config would have passed against
    the broken code.
    """
    from connito.validator import run as validator_run

    recorder = _Recorder()
    monkeypatch.setattr(validator_run, "load_model", recorder)

    config = _StubConfig(partial_moe=partial_moe)
    with pytest.raises(_Stop):
        validator_run.load_model(
            0, config, None, None, None, None,
            partial=bool(config.moe.partial_moe),
            checkpoint_device=None,
            load_global_checkpoint=False,
        )

    assert recorder.partial is expected_partial


class _StubMoE:
    def __init__(self, partial_moe: bool) -> None:
        self.partial_moe = partial_moe


class _StubConfig:
    def __init__(self, partial_moe: bool) -> None:
        self.moe = _StubMoE(partial_moe)


# ── the startup warning ──────────────────────────────────────────────────────
def test_full_topology_warns_at_startup(caplog):
    """An OOM two hours into a round is a bad way to learn the footprint."""
    from connito.validator.run import warn_on_full_expert_topology

    config = _WarnStub(partial_moe=False)
    warn_on_full_expert_topology(config)
    assert any("FULL expert topology" in r.getMessage() for r in caplog.records)


def test_partial_topology_is_silent(caplog):
    from connito.validator.run import warn_on_full_expert_topology

    warn_on_full_expert_topology(_WarnStub(partial_moe=True))
    assert not any("FULL expert topology" in r.getMessage() for r in caplog.records)


class _WarnStub:
    """`get_nested_attr` walks real attributes, so the stub needs the shape."""

    def __init__(self, partial_moe: bool) -> None:
        self.moe = type("_M", (), {
            "partial_moe": partial_moe, "full_topk": 6,
        })()
        self.model = type("_Mo", (), {
            "quantization": "off", "precision": "fp16-mixed",
        })()

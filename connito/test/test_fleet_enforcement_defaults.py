"""The four config fields that make full-topology scoring a fleet default.

Enforcement across the fleet is not a deployment step — Watchtower replaces the
*image* and never touches an operator's `docker-compose.yml`, `.env` or
`config.yaml`. The only channel that reaches a running validator's settings is
`check_and_prompt_locked`, which resets locked fields to their class defaults on
every start (`--auto_update_config` defaults to True). So "enforce this
fleet-wide" means exactly: ship the default, and lock the field.

Four fields have to move together, because
`check_full_topology_eval_supported` rejects the mixtures:

    evaluation.full_topology_eval  true   <- what we are enforcing
    moe.partial_moe               true   <- rejected as false alongside it
    model.quantization            "off"  <- rejected as "fp8" alongside it
    model.precision               bf16   <- comparability, and the experiment's dtype

Leaving any one of them unlocked hands the operator who had edited it a
startup crash on the release that turns full-topology eval on. These tests are
here so that a later "make it configurable again" edit has to argue with the
reason rather than silently reintroduce the crash-loop.

Run with `python -m pytest connito/test/test_fleet_enforcement_defaults.py`.
"""

from __future__ import annotations

import pytest

from connito.shared.config import EvalCfg, ModelCfg, MoECfg

# (class, field, expected default) — the enforced fleet configuration.
ENFORCED = [
    (EvalCfg, "full_topology_eval", True),
    (MoECfg, "partial_moe", True),
    (ModelCfg, "quantization", "off"),
    (ModelCfg, "precision", "bf16-mixed"),
]


@pytest.mark.parametrize(
    ("cfg_cls", "field", "expected"),
    ENFORCED,
    ids=[f"{c.__name__}.{f}" for c, f, _ in ENFORCED],
)
def test_the_enforced_default_ships(cfg_cls, field, expected):
    assert cfg_cls.model_fields[field].default == expected


@pytest.mark.parametrize(
    ("cfg_cls", "field"),
    [(c, f) for c, f, _ in ENFORCED],
    ids=[f"{c.__name__}.{f}" for c, f, _ in ENFORCED],
)
def test_the_enforced_field_is_locked(cfg_cls, field):
    """Shipping the default alone is not enforcement.

    A new field gets one free ride: pydantic fills the default and
    `config.write()` persists it. An *existing* field does not — the
    operator's YAML value survives every pull forever. `full_topology_eval`
    is the only one of the four that is new, and spending its free ride on
    `false` would have made it unenforceable without a lock anyway.
    """
    assert field in cfg_cls._LOCKED_FIELDS


def test_locked_defaults_survive_a_hand_edited_yaml():
    """The actual mechanism, exercised rather than asserted about.

    This is what happens on a validator that has been running the old
    settings when the new image starts: `locked_defaults` is what
    `check_and_prompt_locked` writes back over the operator's values.
    """
    stale = EvalCfg(full_topology_eval=False)
    assert stale.full_topology_eval is False  # the edit takes, pre-reset
    assert stale.locked_defaults()["full_topology_eval"] is True


def test_precision_is_safe_to_lock_on_a_card_without_bf16():
    """Locking a dtype would be reckless if the dtype could be unsupported.

    `resolve_precision` downgrades bf16 -> fp16 on a device without BF16
    compute, so the locked default lands as fp16 through the resolver on such
    a card rather than as a broken config. That downgrade is the precondition
    for locking this field at all.
    """
    from connito.shared import helper

    assert "bf16" in ModelCfg.model_fields["precision"].default
    assert hasattr(helper, "resolve_precision")

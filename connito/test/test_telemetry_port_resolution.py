"""Tests for `resolve_telemetry_port`.

`CONNITO_TELEMETRY_PORT` shipped in the validator image for months but was
never read by any code — the exporter port was hardcoded to `8200 + rank`.
An operator whose host already had 8200 occupied would set the variable,
see no effect, and end up with a validator that ran fine on chain while the
exporter failed to bind ("Address already in use") and served nothing.

The variable is a **base** port, not an absolute one: the effective port is
`base + rank`, matching the semantics of the 8200 default it replaces, so a
multi-rank deployment on one host stays collision-free.
"""
from __future__ import annotations

from connito.validator.run import DEFAULT_TELEMETRY_BASE_PORT, resolve_telemetry_port


# ---------------------------------------------------------------------------
# Default / fallback behaviour
# ---------------------------------------------------------------------------

def test_default_when_env_unset():
    assert resolve_telemetry_port(0, env={}) == 8200


def test_default_applies_rank_offset():
    assert resolve_telemetry_port(1, env={}) == 8201
    assert resolve_telemetry_port(3, env={}) == 8203


def test_blank_and_whitespace_are_treated_as_unset():
    assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": ""}) == 8200
    assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": "   "}) == 8200


# ---------------------------------------------------------------------------
# Override behaviour — the case this exists for
# ---------------------------------------------------------------------------

def test_override_is_used_verbatim_at_rank_zero():
    # The validator always runs rank 0, so an operator setting 8201 gets 8201.
    assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": "8201"}) == 8201


def test_override_is_a_base_not_an_absolute_port():
    # Multi-rank stays collision-free: an absolute override would point every
    # rank at the same port and all but one would fail to bind.
    env = {"CONNITO_TELEMETRY_PORT": "9100"}
    assert resolve_telemetry_port(0, env=env) == 9100
    assert resolve_telemetry_port(1, env=env) == 9101
    assert resolve_telemetry_port(2, env=env) == 9102


def test_override_tolerates_surrounding_whitespace():
    assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": " 8201 "}) == 8201


# ---------------------------------------------------------------------------
# Invalid input must never take the validator down
# ---------------------------------------------------------------------------

def test_non_numeric_falls_back_to_default():
    assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": "not-a-port"}) == 8200


def test_out_of_range_falls_back_to_default():
    for bad in ("0", "-1", "65536", "999999"):
        assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": bad}) == 8200


def test_base_plus_rank_overflow_falls_back_to_default():
    # 65535 is a legal base but overflows once rank is added.
    assert resolve_telemetry_port(1, env={"CONNITO_TELEMETRY_PORT": "65535"}) == (
        DEFAULT_TELEMETRY_BASE_PORT + 1
    )


def test_float_string_falls_back_rather_than_truncating():
    # Silently truncating "8201.9" to 8201 would be a surprising success.
    assert resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": "8201.9"}) == 8200


def test_resolution_never_raises_on_hostile_input():
    for bad in ("", "  ", "abc", "8201;rm -rf /", "0x2009", "١٢٣"):
        port = resolve_telemetry_port(0, env={"CONNITO_TELEMETRY_PORT": bad})
        assert isinstance(port, int) and 1 <= port <= 65535

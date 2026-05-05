"""Contract tests for /v1/state.json v2.0 (slim API).

Locks in the v2 schema: dropped subnet/phase sections, slim round
section, leaderboard rows without chain-derivable metadata. Anything
that breaks one of these tests is a backwards-incompatible payload
change and warrants a `PAYLOAD_VERSION` major bump.
"""

from __future__ import annotations

from types import SimpleNamespace

from connito.validator import api


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

class _FakeRound:
    """Minimal Round stand-in. Only `snapshot()` is consumed by api.py."""

    def __init__(
        self,
        *,
        foreground=(10, 11),
        val_loss=None,
        baseline=1.0,
        uid_to_hotkey=None,
        round_id=7,
    ):
        self._snap = {
            "round_id": round_id,
            "baseline_loss": baseline,
            "foreground_uids": tuple(foreground),
            "uid_to_hotkey": uid_to_hotkey or {10: "hkA", 11: "hkB", 99: "hkZ"},
            "uid_to_chain_checkpoint": {},
            "val_loss_by_uid": val_loss if val_loss is not None else {10: 0.5, 11: 0.8},
        }

    def snapshot(self) -> dict:
        # Return a shallow copy so a caller mutating the result doesn't
        # taint subsequent reads in the same test.
        return dict(self._snap)


def _state(**overrides) -> SimpleNamespace:
    """Build a fake `app.state` namespace mirroring what `build_app` stashes."""
    base = SimpleNamespace(
        config=None,
        round_ref=SimpleNamespace(current=_FakeRound()),
        score_aggregator=SimpleNamespace(
            uid_score_pairs=lambda how: (
                {10: 2.25, 11: 1.5} if how == "latest" else {10: 1.8, 11: 1.2}
            )
        ),
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="vHK")),
        validator_uid=3,
        git_version="vTest",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ----------------------------------------------------------------------------
# Envelope / version
# ----------------------------------------------------------------------------

def test_payload_version_is_2_0():
    snap = api._build_state_snapshot(_state())
    assert snap["meta"]["payload_version"] == "2.0"


def test_envelope_has_only_four_top_level_sections():
    snap = api._build_state_snapshot(_state())
    assert set(snap.keys()) == {"meta", "validator", "round", "leaderboard"}


def test_envelope_does_not_leak_dropped_v1_sections():
    """`subnet`, `phase`, and any of their nested fields must not appear."""
    snap = api._build_state_snapshot(_state())
    assert "subnet" not in snap
    assert "phase" not in snap


def test_meta_section_carries_version_and_timestamp():
    snap = api._build_state_snapshot(_state())
    meta = snap["meta"]
    assert meta["payload_version"] == "2.0"
    assert meta["connito_version"] == "vTest"
    assert isinstance(meta["snapshot_ts"], float)
    assert meta["snapshot_ts"] > 0


def test_validator_section_exposes_hotkey_and_uid():
    snap = api._build_state_snapshot(_state())
    assert snap["validator"] == {"hotkey": "vHK", "uid": 3}


def test_validator_section_tolerates_missing_wallet():
    s = _state(wallet=None)
    snap = api._build_state_snapshot(s)
    assert snap["validator"] == {"hotkey": None, "uid": 3}


# ----------------------------------------------------------------------------
# Round section
# ----------------------------------------------------------------------------

def test_round_section_has_only_id_and_baseline_loss():
    """v2 drops `stats` and `baseline_loss_history` from the round section."""
    snap = api._build_state_snapshot(_state())
    assert set(snap["round"].keys()) == {"id", "baseline_loss"}


def test_round_section_returns_nulls_when_no_round_active():
    s = _state(round_ref=SimpleNamespace(current=None))
    snap = api._build_state_snapshot(s)
    assert snap["round"] == {"id": None, "baseline_loss": None}


# ----------------------------------------------------------------------------
# Leaderboard rows
# ----------------------------------------------------------------------------

_EXPECTED_ROW_KEYS = {"uid", "score", "delta_loss", "val_loss"}


def test_leaderboard_row_has_exactly_the_v2_keys():
    snap = api._build_state_snapshot(_state())
    for row in snap["leaderboard"]:
        assert set(row.keys()) == _EXPECTED_ROW_KEYS, row


def test_leaderboard_row_does_not_leak_chain_or_prometheus_fields():
    """Fields that are derivable from chain (hotkey, hf_repo_id,
    hf_revision, in_assignment) or already published to Prometheus
    (weight_submitted) must not appear on rows. Re-adding any of
    these is a contract regression."""
    snap = api._build_state_snapshot(_state())
    for row in snap["leaderboard"]:
        for dropped in (
            "hotkey",
            "hf_repo_id",
            "hf_revision",
            "in_assignment",
            "weight_submitted",
        ):
            assert dropped not in row, f"{dropped!r} leaked into row {row}"


def test_leaderboard_sort_order_score_desc_with_nulls_last():
    snap = api._build_state_snapshot(_state())
    uids = [r["uid"] for r in snap["leaderboard"]]
    # uid 10 score 2.25, uid 11 score 1.5, uid 99 unscored → null at end
    assert uids == [10, 11, 99]


def test_unscored_uid_renders_with_null_signals():
    snap = api._build_state_snapshot(_state())
    by_uid = {r["uid"]: r for r in snap["leaderboard"]}
    assert by_uid[99] == {
        "uid": 99,
        "score": None,
        "delta_loss": None,
        "val_loss": None,
    }


def test_delta_loss_is_max_zero_baseline_minus_val_loss():
    snap = api._build_state_snapshot(_state())
    by_uid = {r["uid"]: r for r in snap["leaderboard"]}
    # baseline 1.0, uid 10 val_loss 0.5 → delta 0.5
    assert by_uid[10]["delta_loss"] == 0.5
    # uid 11 val_loss 0.8 → delta 0.2 (allow float slack)
    assert abs(by_uid[11]["delta_loss"] - 0.2) < 1e-9


def test_delta_loss_clamps_at_zero_when_val_loss_exceeds_baseline():
    """A worse-than-baseline submission yields delta_loss == 0, never negative."""
    s = _state(round_ref=SimpleNamespace(current=_FakeRound(
        val_loss={10: 5.0},
        baseline=1.0,
    )))
    snap = api._build_state_snapshot(s)
    by_uid = {r["uid"]: r for r in snap["leaderboard"]}
    assert by_uid[10]["delta_loss"] == 0.0


def test_delta_loss_is_null_when_baseline_is_missing():
    s = _state(round_ref=SimpleNamespace(current=_FakeRound(baseline=None)))
    snap = api._build_state_snapshot(s)
    for row in snap["leaderboard"]:
        assert row["delta_loss"] is None


def test_score_aggregator_failure_does_not_break_endpoint():
    """If `uid_score_pairs` raises, leaderboard still renders with null scores."""
    def _raising(*_args, **_kwargs):
        raise RuntimeError("aggregator down")

    s = _state(score_aggregator=SimpleNamespace(uid_score_pairs=_raising))
    snap = api._build_state_snapshot(s)
    for row in snap["leaderboard"]:
        assert row["score"] is None


def test_no_round_returns_empty_leaderboard():
    s = _state(round_ref=SimpleNamespace(current=None))
    snap = api._build_state_snapshot(s)
    assert snap["leaderboard"] == []


# ----------------------------------------------------------------------------
# Cross-section invariants
# ----------------------------------------------------------------------------

def test_round_baseline_matches_leaderboard_delta_arithmetic():
    """`round.baseline_loss` is the value rows compute `delta_loss` against —
    aggregators rely on this consistency to render delta the same way the
    validator did."""
    snap = api._build_state_snapshot(_state())
    baseline = snap["round"]["baseline_loss"]
    by_uid = {r["uid"]: r for r in snap["leaderboard"]}
    row = by_uid[10]
    assert baseline is not None and row["val_loss"] is not None
    assert row["delta_loss"] == max(0.0, baseline - row["val_loss"])

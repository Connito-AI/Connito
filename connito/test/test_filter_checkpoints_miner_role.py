"""Pin the version-range gate skip for the miner role in `filter_checkpoints`.

The gate compares each chain commit's ``global_ver`` against
``[min_allowed_version, max_allowed_version]`` and silently drops any
miner whose chain commit happens to be 1-2 blocks off the window. We
observed this in SN102 staging:

    filter_checkpoints: excluded (version exceeds max allowed)
        ckpt_ver=8338209  max_allowed_version=8338208  uid=131

— a miner that committed one block after the validator's expected
window was excluded, even though its actual checkpoint was fresh.

The fix: skip the version-range gate when ``for_role == "miner"``.
Validators downstream do their own evaluation-driven gating, so any
junk submission gets caught there. Validators evaluating each other's
commits (``for_role="validator"``) still need the gate because the
selected majority hash propagates into model state — that path is
unchanged.
"""
from __future__ import annotations


from connito.shared.checkpoints import ChainCheckpoint, ChainCheckpoints


def _ckpt(uid: int, global_ver: int) -> ChainCheckpoint:
    return ChainCheckpoint(
        signed_model_hash="ab" * 32,
        model_hash="cd" * 32,
        global_ver=global_ver,
        expert_group=0,
        uid=uid,
        ip="0.0.0.0",
        port=8000,
        hotkey=f"5Test{uid:03d}",
        stake=1.0,
    )


# A pair: one inside the version window, one one block above (the
# real-world race we observed in the staging log).
INSIDE = 8338208
ABOVE_MAX = 8338209


def test_miner_role_skips_version_range_gate():
    """Both inside and above-max commits survive when role=miner."""
    cks = ChainCheckpoints(checkpoints=[_ckpt(1, INSIDE), _ckpt(2, ABOVE_MAX)])

    filtered = cks.filter_checkpoints(
        for_role="miner",
        min_allowed_version=INSIDE - 100,
        max_allowed_version=INSIDE,
    )

    kept_uids = sorted(c.uid for c in filtered.checkpoints)
    assert kept_uids == [1, 2], (
        f"miner role must keep both inside-window and above-window commits; got {kept_uids}"
    )


def test_validator_role_still_applies_version_range_gate():
    """Regression: validator role's selection logic still rejects above-max."""
    # Build commits that all share the same model_hash, since the validator
    # path runs the majority-hash voter after the version filter. Use stake=1.0
    # so the majority vote is deterministic.
    cks = ChainCheckpoints(checkpoints=[_ckpt(1, INSIDE), _ckpt(2, ABOVE_MAX)])

    filtered = cks.filter_checkpoints(
        for_role="validator",
        min_allowed_version=INSIDE - 100,
        max_allowed_version=INSIDE,
    )

    kept_uids = sorted(c.uid for c in filtered.checkpoints)
    assert 2 not in kept_uids, (
        f"validator role must still reject above-max commits; got {kept_uids}"
    )


def test_miner_role_still_drops_incomplete_commits():
    """The completeness gate (signed_model_hash / model_hash / hotkey / ...)
    must still run for miners — we only relaxed the version range, not the
    rest of the validation surface."""
    incomplete = ChainCheckpoint(
        # signed_model_hash and model_hash missing — should be dropped by
        # the completeness gate even when for_role="miner"
        global_ver=INSIDE,
        expert_group=0,
        uid=99,
        ip="0.0.0.0",
        port=8000,
        hotkey="5Test099",
        stake=1.0,
    )
    complete = _ckpt(1, INSIDE)
    cks = ChainCheckpoints(checkpoints=[incomplete, complete])

    filtered = cks.filter_checkpoints(
        for_role="miner",
        min_allowed_version=INSIDE - 100,
        max_allowed_version=INSIDE,
    )

    kept_uids = sorted(c.uid for c in filtered.checkpoints)
    assert kept_uids == [1], (
        f"incomplete commits must still be dropped for miner role; got {kept_uids}"
    )


def test_miner_role_with_no_version_window_is_unaffected():
    """Sanity: callers that don't supply a version range get the same
    behaviour as before (no filtering by version, all valid commits pass)."""
    cks = ChainCheckpoints(checkpoints=[_ckpt(1, INSIDE), _ckpt(2, ABOVE_MAX)])

    filtered = cks.filter_checkpoints(
        for_role="miner",
        min_allowed_version=None,
        max_allowed_version=None,
    )

    kept_uids = sorted(c.uid for c in filtered.checkpoints)
    assert kept_uids == [1, 2]

"""Unit tests for `connito.shared.eval_shard_pick`.

Verifies the consensus-load-bearing properties of the seeded
shard-pick eval path:

  D1. Determinism — same (repo_id, name, revision, int_seed) yields
      byte-identical (shard, offset). Required for every validator
      to score the same rows for the same seed.

  D2. Mod safety — in_shard_offset is always strictly less than the
      pick's `offset_bound` and `shard_rows`. Guards against landing
      past end-of-stream and tripping `interleave_datasets`'s default
      `first_exhausted` strategy, which would collapse the whole eval
      round.

  D3. Per-source independence — two configured sources at the same
      seed produce independent (shard, offset) draws. The hash uses
      `repo_id` as a domain separator so picks don't correlate.

  D4. Coverage — across many seeds, picks distribute uniformly over
      the shard space (no shard pinning, no bias toward shard 0).

  D5. Loud failure modes — unknown source raises KeyError; misconfigured
      policy raises ValueError at module load; small-shard parquet
      footer raises RuntimeError; no silent fallback that could break
      consensus quietly.

  D6. Sort stability — shard list returned by `_list_shards` is
      sorted deterministically regardless of the order in which HF
      returns `siblings`. Required for cross-validator agreement.

  D7. Production policies pass module-load validation — the
      `_KNOWN_SOURCES` registry as shipped imports without raising.

  D8. Revision pin override is threaded through to HfApi.

Tests are offline-mockable. `HfApi().dataset_info` is patched to
return synthetic siblings so they run without network.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from connito.shared import eval_shard_pick


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _sibling(rfilename: str) -> SimpleNamespace:
    """Mimic the shape of `huggingface_hub.dataset_info().siblings[i]`."""
    return SimpleNamespace(rfilename=rfilename)


def _c4_info(shard_count: int = 16, *, with_noise: bool = True) -> SimpleNamespace:
    """Fake HF dataset_info() result for allenai/c4 with `shard_count`
    en/train shards plus (optionally) noise files we expect to be
    filtered out (validation, noblocklist, multilingual variants,
    metadata).
    """
    siblings = [
        _sibling(f"en/c4-train.{i:05d}-of-{shard_count:05d}.json.gz")
        for i in range(shard_count)
    ]
    if with_noise:
        siblings.extend([
            _sibling(".gitattributes"),
            _sibling("README.md"),
            _sibling("en/c4-validation.00000-of-00008.json.gz"),
            _sibling("en.noblocklist/c4-train.00000-of-01024.json.gz"),
            _sibling("multilingual/c4-train.fr.00000-of-00256.json.gz"),
        ])
    # Intentionally shuffled — _list_shards must impose its own sort.
    import random
    random.Random(0).shuffle(siblings)
    return SimpleNamespace(siblings=siblings, sha="abc123")


def _nemo_info(shard_count: int = 4) -> SimpleNamespace:
    siblings = [
        _sibling(f"4plus/part_{i:06d}.parquet")
        for i in range(shard_count)
    ]
    siblings.extend([
        _sibling("README.md"),
        _sibling("4plus_MIND/part_000000.parquet"),  # different config — filtered out
    ])
    return SimpleNamespace(siblings=siblings, sha="def456")


def _clear_caches():
    """All `lru_cache`-decorated helpers in eval_shard_pick. Tests
    that patch `HfApi` must clear these or stale results leak."""
    eval_shard_pick._resolve_revision.cache_clear()
    eval_shard_pick._list_shards.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_caches():
    _clear_caches()
    yield
    _clear_caches()


def _patch_hf_info(repo_id_to_info: dict[str, SimpleNamespace]):
    """Patch HfApi().dataset_info to return our synthetic infos."""
    def _fake_dataset_info(self, repo_id, *, revision=None, **_):
        if repo_id not in repo_id_to_info:
            raise KeyError(repo_id)
        return repo_id_to_info[repo_id]
    return patch.object(
        eval_shard_pick.HfApi, "dataset_info", _fake_dataset_info,
    )


def _patch_parquet_footer(rows_per_shard: int):
    """Stand in for `_shard_rows_via_parquet_footer` so tests don't
    actually download parquet files."""
    return patch.object(
        eval_shard_pick,
        "_shard_rows_via_parquet_footer",
        return_value=rows_per_shard,
    )


# -----------------------------------------------------------------------
# D1 — Determinism
# -----------------------------------------------------------------------

def test_pick_is_deterministic_for_same_seed_constant_source():
    """Same (source, seed, revision) → same shard + same offset.
    Without this, two validators with the same combined_seed would
    score different rows and weight consensus breaks."""
    info = _c4_info(shard_count=32)
    with _patch_hf_info({"allenai/c4": info}):
        a = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0x1234abcd,
        )
        _clear_caches()
        b = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0x1234abcd,
        )
    assert a.shard_path == b.shard_path
    assert a.in_shard_offset == b.in_shard_offset
    assert a.offset_bound == b.offset_bound
    assert a.shard_rows == b.shard_rows


def test_pick_is_deterministic_for_same_seed_parquet_source():
    """Determinism on the parquet path (offset bound = actual − headroom)."""
    info = _nemo_info(shard_count=8)
    with _patch_hf_info({"nvidia/Nemotron-CC-Math-v1": info}), \
         _patch_parquet_footer(rows_per_shard=120_000):
        a = eval_shard_pick.pick_shard_for_source(
            repo_id="nvidia/Nemotron-CC-Math-v1", name="4plus", int_seed=42,
        )
        _clear_caches()
        b = eval_shard_pick.pick_shard_for_source(
            repo_id="nvidia/Nemotron-CC-Math-v1", name="4plus", int_seed=42,
        )
    assert (a.shard_path, a.in_shard_offset, a.offset_bound) == \
           (b.shard_path, b.in_shard_offset, b.offset_bound)


def test_different_seeds_produce_different_picks_on_average():
    """Two different seeds should not always collide. With 32 shards
    and the seed space we sample, the probability of all picks
    landing on a single shard is vanishing — collisions for ALL 64
    test seeds would be ~32**-63."""
    info = _c4_info(shard_count=32)
    seeds_to_picks = {}
    with _patch_hf_info({"allenai/c4": info}):
        for s in range(64):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            seeds_to_picks[s] = (pick.shard_path, pick.in_shard_offset)

    distinct_shards = {sh for sh, _ in seeds_to_picks.values()}
    assert len(distinct_shards) >= 8, (
        f"Only {len(distinct_shards)} distinct shards across 64 seeds — "
        f"shard pick may not be properly seeded"
    )
    distinct_offsets = {off for _, off in seeds_to_picks.values()}
    assert len(distinct_offsets) >= 8


# -----------------------------------------------------------------------
# D2 — Mod safety
# -----------------------------------------------------------------------

def test_in_shard_offset_under_bound_constant_source():
    """For constant-source picks (C4), offset < safe_floor always.
    Module-load validation guarantees safe_floor + headroom ≤
    every verified shard's row count, so the read window can't run
    short."""
    info = _c4_info(shard_count=32)
    policy = eval_shard_pick._policy_for("allenai/c4", "en")
    with _patch_hf_info({"allenai/c4": info}):
        for s in range(200):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            assert 0 <= pick.in_shard_offset < pick.offset_bound
            assert pick.offset_bound == policy.safe_floor_rows
            assert pick.shard_rows == pick.offset_bound  # constant path

            # And the load-bearing invariant: even at the maximum
            # possible offset, every verified shard would retain
            # at least min_headroom_rows.
            min_verified = min(policy.verified_shard_rows.values())
            assert pick.in_shard_offset + policy.min_headroom_rows <= min_verified


def test_in_shard_offset_under_bound_parquet_source():
    """For parquet-source picks (Nemotron), offset < actual_rows − headroom."""
    info = _nemo_info(shard_count=8)
    policy = eval_shard_pick._policy_for("nvidia/Nemotron-CC-Math-v1", "4plus")
    with _patch_hf_info({"nvidia/Nemotron-CC-Math-v1": info}), \
         _patch_parquet_footer(rows_per_shard=120_000):
        for s in range(100):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="nvidia/Nemotron-CC-Math-v1",
                name="4plus", int_seed=s,
            )
            assert 0 <= pick.in_shard_offset < pick.offset_bound
            assert pick.offset_bound == 120_000 - policy.min_headroom_rows
            assert pick.shard_rows == 120_000
            # Headroom invariant
            assert pick.in_shard_offset + policy.min_headroom_rows <= pick.shard_rows


def test_parquet_shard_smaller_than_headroom_raises():
    """A shard so small that no safe offset exists must raise loud
    rather than silently collapsing the round to zero batches."""
    info = _nemo_info(shard_count=4)
    policy = eval_shard_pick._policy_for("nvidia/Nemotron-CC-Math-v1", "4plus")
    too_small = policy.min_headroom_rows - 1
    with _patch_hf_info({"nvidia/Nemotron-CC-Math-v1": info}), \
         _patch_parquet_footer(rows_per_shard=too_small):
        with pytest.raises(RuntimeError, match="below the configured min_headroom_rows"):
            eval_shard_pick.pick_shard_for_source(
                repo_id="nvidia/Nemotron-CC-Math-v1",
                name="4plus", int_seed=0,
            )


# -----------------------------------------------------------------------
# D3 — Per-source independence
# -----------------------------------------------------------------------

def test_two_sources_at_same_seed_pick_independently():
    """A miner that predicts source A's pick must NOT learn anything
    about source B's pick. The hash uses `repo_id` as a domain
    separator, so identical seeds across sources are independent
    draws on different hash domains."""
    c4 = _c4_info(shard_count=16)
    nemo = _nemo_info(shard_count=8)
    with _patch_hf_info({
        "allenai/c4": c4,
        "nvidia/Nemotron-CC-Math-v1": nemo,
    }), _patch_parquet_footer(120_000):
        coincidences = 0
        for s in range(32):
            a = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            b = eval_shard_pick.pick_shard_for_source(
                repo_id="nvidia/Nemotron-CC-Math-v1", name="4plus", int_seed=s,
            )
            a_idx = int(a.shard_path.split(".")[1].split("-")[0])
            b_idx = int(b.shard_path.split("_")[-1].split(".")[0])
            if a_idx % 8 == b_idx:
                coincidences += 1
        assert coincidences <= 16, (
            f"{coincidences}/32 seed-coincidences — domain separation broken?"
        )


# -----------------------------------------------------------------------
# D4 — Coverage
# -----------------------------------------------------------------------

def test_pick_distribution_covers_shards_uniformly():
    """Across many seeds, every shard should get picked roughly
    1/N of the time. A bug that pinned picks to a single shard
    would show up as one shard taking >50% of picks.

    Tolerance: with 16 shards and 1600 seeds, expected count per
    shard ≈ 100; allow each shard to land in [40, 200] — about
    6 sigma either way."""
    info = _c4_info(shard_count=16)
    counts: Counter[str] = Counter()
    with _patch_hf_info({"allenai/c4": info}):
        for s in range(1600):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            counts[pick.shard_path] += 1
    assert len(counts) == 16
    for shard, count in counts.items():
        assert 40 <= count <= 200, (
            f"shard {shard} picked {count} times — distribution suspect"
        )


# -----------------------------------------------------------------------
# D5 — Loud failure modes
# -----------------------------------------------------------------------

def test_unknown_source_raises_keyerror():
    """Adding a new source MUST go through `_KNOWN_SOURCES`. A silent
    default could let a misconfigured operator break consensus
    quietly."""
    with pytest.raises(KeyError, match="No shard-pick policy"):
        eval_shard_pick.pick_shard_for_source(
            repo_id="not/a-real-dataset", name=None, int_seed=42,
        )


def test_empty_shard_list_raises():
    """A filter that matched zero files would silently degrade today's
    setup. Raise explicitly so a typo in `path_prefix`/`path_suffix`
    surfaces at the first call."""
    bad_info = SimpleNamespace(siblings=[_sibling("README.md")], sha="zzz")
    with _patch_hf_info({"allenai/c4": bad_info}):
        with pytest.raises(RuntimeError, match="No shards matched"):
            eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=42,
            )


def test_policy_validation_rejects_safe_floor_above_verified_minus_headroom():
    """`safe_floor + min_headroom > min(verified_shard_rows)` is the
    bug that quietly over-skips small shards. Validation MUST raise
    at module-load (here: at construction of a fresh policy and a
    direct _validate_policy call) so it can't ship to production."""
    bad_policy = eval_shard_pick._SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",
        row_count_source="constant",
        safe_floor_rows=400_000,  # > verified 356_317 − 10 000
        min_headroom_rows=10_000,
        verified_shard_rows={
            "en/c4-train.00000-of-01024.json.gz": 356_317,
        },
    )
    with pytest.raises(ValueError, match=r"safe_floor_rows.*min_headroom_rows"):
        eval_shard_pick._validate_policy(("allenai/c4", "en"), bad_policy)


def test_policy_validation_rejects_constant_without_verified():
    """Constant-source policies without any spot-check entry have no
    way to validate `safe_floor_rows`."""
    bad_policy = eval_shard_pick._SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",
        row_count_source="constant",
        safe_floor_rows=100_000,
        verified_shard_rows={},
    )
    with pytest.raises(ValueError, match="at least one verified"):
        eval_shard_pick._validate_policy(("any", "any"), bad_policy)


def test_policy_validation_rejects_constant_without_safe_floor():
    bad_policy = eval_shard_pick._SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",
        row_count_source="constant",
        safe_floor_rows=None,
        verified_shard_rows={"en/x.json.gz": 100_000},
    )
    with pytest.raises(ValueError, match="requires safe_floor_rows"):
        eval_shard_pick._validate_policy(("any", "any"), bad_policy)


def test_policy_validation_rejects_unknown_row_count_source():
    bad_policy = eval_shard_pick._SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",
        row_count_source="not_a_real_kind",
    )
    with pytest.raises(ValueError, match="unknown row_count_source"):
        eval_shard_pick._validate_policy(("any", "any"), bad_policy)


def test_policy_validation_rejects_nonpositive_headroom():
    bad_policy = eval_shard_pick._SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",
        row_count_source="constant",
        safe_floor_rows=100_000,
        min_headroom_rows=0,
        verified_shard_rows={"en/x.json.gz": 200_000},
    )
    with pytest.raises(ValueError, match="min_headroom_rows must be > 0"):
        eval_shard_pick._validate_policy(("any", "any"), bad_policy)


# -----------------------------------------------------------------------
# D6 — Sort stability
# -----------------------------------------------------------------------

def test_shard_list_sort_is_independent_of_hf_return_order():
    """HF's siblings list order is not guaranteed to be stable. The
    pick must be insensitive to the order siblings come back in."""
    files = [f"en/c4-train.{i:05d}-of-00016.json.gz" for i in range(16)]
    import random
    rng1 = random.Random(1)
    rng2 = random.Random(2)
    files_order_a = list(files); rng1.shuffle(files_order_a)
    files_order_b = list(files); rng2.shuffle(files_order_b)
    info_a = SimpleNamespace(siblings=[_sibling(f) for f in files_order_a], sha="s")
    info_b = SimpleNamespace(siblings=[_sibling(f) for f in files_order_b], sha="s")
    with _patch_hf_info({"allenai/c4": info_a}):
        a = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0xdeadbeef,
        )
    _clear_caches()
    with _patch_hf_info({"allenai/c4": info_b}):
        b = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0xdeadbeef,
        )
    assert a.shard_path == b.shard_path
    assert a.in_shard_offset == b.in_shard_offset


def test_filter_drops_validation_and_other_configs():
    """`_KNOWN_SOURCES["allenai/c4", "en"]` must NOT pick up
    `en/c4-validation.*`, `en.noblocklist/...`, or `multilingual/...`
    files. A silent over-match here would change the reachable pool."""
    info = _c4_info(shard_count=8, with_noise=True)
    with _patch_hf_info({"allenai/c4": info}):
        for s in range(50):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            assert pick.shard_path.startswith("en/c4-train.")
            assert pick.shard_path.endswith(".json.gz")
            assert "validation" not in pick.shard_path
            assert "noblocklist" not in pick.shard_path
            assert "multilingual" not in pick.shard_path


# -----------------------------------------------------------------------
# D7 — Production policies pass module-load validation
# -----------------------------------------------------------------------

def test_shipped_policies_pass_validation():
    """The `_KNOWN_SOURCES` dict as shipped must already pass the
    same validation that runs at module load. This catches a developer
    bumping `safe_floor_rows` without updating `verified_shard_rows`
    (or vice-versa) before commit."""
    for key, policy in eval_shard_pick._KNOWN_SOURCES.items():
        eval_shard_pick._validate_policy(key, policy)


# -----------------------------------------------------------------------
# D8 — Revision pin override
# -----------------------------------------------------------------------

def test_revision_pin_override_is_threaded_through():
    """`revision_override` must reach `_resolve_revision` (and from
    there `_list_shards`)."""
    info = _c4_info(shard_count=4)
    seen_revisions: list[str] = []

    def _capturing_dataset_info(self, repo_id, *, revision=None, **_):
        seen_revisions.append(revision)
        return info

    with patch.object(
        eval_shard_pick.HfApi, "dataset_info", _capturing_dataset_info,
    ):
        eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=1,
            revision_override="my-explicit-sha",
        )
    assert seen_revisions[0] == "my-explicit-sha"

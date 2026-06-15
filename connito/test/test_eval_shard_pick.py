"""Unit tests for `connito.shared.eval_shard_pick`.

Verifies the consensus-load-bearing properties of the seeded
shard-pick eval path:

  D1. Determinism — same (repo_id, name, revision, int_seed) yields
      byte-identical (shard, offset). Required for every validator
      to score the same rows for the same seed.

  D2. Mod safety — in_shard_offset is always strictly less than
      shard_rows. Guards against landing past end-of-stream and
      tripping `interleave_datasets`'s default `first_exhausted`
      strategy, which would collapse the whole eval round.

  D3. Per-source independence — two configured sources at the same
      seed produce independent (shard, offset) draws. The hash uses
      `repo_id` as a domain separator so picks don't correlate.

  D4. Coverage — across many seeds, picks distribute uniformly over
      the shard space (no shard pinning, no bias toward shard 0).

  D5. Loud failure modes — unknown source raises KeyError; missing
      row-count table entry raises KeyError. No silent fallback that
      could break consensus quietly.

  D6. Sort stability — shard list returned by `_list_shards` is
      sorted deterministically regardless of the order in which HF
      returns `siblings`. Required for cross-validator agreement.

  D7. Bundled-table integration — the checked-in
      `connito/shared/data/c4_en_shard_rows.json` is loadable through
      the importlib.resources path and contains every shard listed
      in `_KNOWN_SOURCES`'s policy for c4-en (or at least the four
      bootstrap entries; a precondition before flipping the gate is
      that this is fully populated).

Tests are offline-mockable. `HfApi().dataset_info` is patched to
return synthetic siblings so they run without network. The bundled
JSON table is read for real to verify resource-loading works end to
end.
"""
from __future__ import annotations

import json
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
    eval_shard_pick._row_count_table.cache_clear()


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


def _patch_row_table(table: dict[str, int]):
    """Override the bundled C4 row-count table for test isolation."""
    return patch.object(
        eval_shard_pick,
        "_load_row_count_table",
        return_value=table,
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

def test_pick_is_deterministic_for_same_seed():
    """Same (source, seed, revision) → same shard + same offset.
    Without this, two validators with the same combined_seed would
    score different rows and weight consensus breaks."""
    info = _c4_info(shard_count=32)
    table = {
        f"en/c4-train.{i:05d}-of-00032.json.gz": 356_317
        for i in range(32)
    }
    with _patch_hf_info({"allenai/c4": info}), _patch_row_table(table):
        a = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0x1234abcd,
        )
        _clear_caches()
        b = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0x1234abcd,
        )
    assert a.shard_path == b.shard_path
    assert a.in_shard_offset == b.in_shard_offset
    assert a.shard_rows == b.shard_rows


def test_different_seeds_produce_different_picks_on_average():
    """Two different seeds should not always collide. With 32 shards
    and the seed space we sample, the probability of all picks
    landing on a single shard is vanishing — collisions for ALL 64
    test seeds would be ~32**-63."""
    info = _c4_info(shard_count=32)
    table = {
        f"en/c4-train.{i:05d}-of-00032.json.gz": 356_317
        for i in range(32)
    }
    seeds_to_picks = {}
    with _patch_hf_info({"allenai/c4": info}), _patch_row_table(table):
        for s in range(64):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            seeds_to_picks[s] = (pick.shard_path, pick.in_shard_offset)

    # At least 8 distinct shards across 64 seeds. (Expected ~26 if
    # uniform; 8 is a generous floor that catches "stuck on one shard"
    # bugs without flaking on rare runs.)
    distinct_shards = {sh for sh, _ in seeds_to_picks.values()}
    assert len(distinct_shards) >= 8, (
        f"Only {len(distinct_shards)} distinct shards across 64 seeds — "
        f"shard pick may not be properly seeded"
    )

    # At least 8 distinct offsets too. (Modulo the shard rows ~356k,
    # collision-free expectation across 64 seeds is essentially
    # certain.)
    distinct_offsets = {off for _, off in seeds_to_picks.values()}
    assert len(distinct_offsets) >= 8


# -----------------------------------------------------------------------
# D2 — Mod safety
# -----------------------------------------------------------------------

def test_in_shard_offset_is_always_under_shard_rows_c4():
    """The whole point of `hash % shard_rows` (rather than
    `randrange(0, fixed_max)`) is that the bound is the chosen
    shard's actual row count — so offset < shard_rows ALWAYS."""
    info = _c4_info(shard_count=32)
    # Mix shard sizes so we'd catch a bug that used max(shard_sizes)
    # as the bound instead of the actual chosen shard's size.
    table = {}
    for i in range(32):
        shard = f"en/c4-train.{i:05d}-of-00032.json.gz"
        table[shard] = 350_000 if i % 2 == 0 else 360_000
    with _patch_hf_info({"allenai/c4": info}), _patch_row_table(table):
        for s in range(200):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            assert 0 <= pick.in_shard_offset < pick.shard_rows, (
                f"offset {pick.in_shard_offset} not in [0, {pick.shard_rows}) "
                f"for shard {pick.shard_path} seed {s}"
            )


def test_in_shard_offset_is_always_under_shard_rows_parquet():
    """Same property on the parquet path (footer-fetched rows)."""
    info = _nemo_info(shard_count=8)
    with _patch_hf_info({"nvidia/Nemotron-CC-Math-v1": info}), \
         _patch_parquet_footer(rows_per_shard=120_000):
        for s in range(100):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="nvidia/Nemotron-CC-Math-v1",
                name="4plus",
                int_seed=s,
            )
            assert 0 <= pick.in_shard_offset < pick.shard_rows


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
    c4_table = {
        f"en/c4-train.{i:05d}-of-00016.json.gz": 356_317
        for i in range(16)
    }
    with _patch_hf_info({
        "allenai/c4": c4,
        "nvidia/Nemotron-CC-Math-v1": nemo,
    }), _patch_row_table(c4_table), _patch_parquet_footer(120_000):
        # Run 32 seeds; count how often source-A's shard index equals
        # source-B's shard index. With shard counts (16, 8), the
        # probability that index_A == index_B (modulo the smaller
        # count) for a single seed is 1/8 = 12.5%. Over 32 seeds the
        # expected count is 4; observing > 16 (more than 4 sigma above
        # mean) would indicate the hash domain separation is broken.
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
    1/N of the time. A bug that pinned picks to a single shard (or
    a small subset) would show up as one shard taking >50% of picks.

    Tolerance: with 16 shards and 1600 seeds, expected count per
    shard ≈ 100; standard deviation ≈ sqrt(100 * 15/16) ≈ 9.7.
    Allow each shard to land in [40, 200] — about 6 sigma either
    way. Catches "always picks shard 0" bugs without flaking on
    legitimate RNG variance.
    """
    info = _c4_info(shard_count=16)
    table = {
        f"en/c4-train.{i:05d}-of-00016.json.gz": 356_317
        for i in range(16)
    }
    counts: Counter[str] = Counter()
    with _patch_hf_info({"allenai/c4": info}), _patch_row_table(table):
        for s in range(1600):
            pick = eval_shard_pick.pick_shard_for_source(
                repo_id="allenai/c4", name="en", int_seed=s,
            )
            counts[pick.shard_path] += 1

    assert len(counts) == 16, (
        f"Only {len(counts)} distinct shards picked across 1600 seeds; "
        f"expected all 16"
    )
    for shard, count in counts.items():
        assert 40 <= count <= 200, (
            f"shard {shard} picked {count} times — distribution suspect"
        )


# -----------------------------------------------------------------------
# D5 — Loud failure modes
# -----------------------------------------------------------------------

def test_unknown_source_raises_keyerror():
    """Adding a new source MUST go through `_KNOWN_SOURCES`. A silent
    default (e.g. "just glob the shards") could let a misconfigured
    operator break consensus quietly. Surface the misconfiguration at
    the first pick call instead."""
    with pytest.raises(KeyError, match="No shard-pick policy"):
        eval_shard_pick.pick_shard_for_source(
            repo_id="not/a-real-dataset", name=None, int_seed=42,
        )


def test_missing_row_count_table_entry_raises_keyerror():
    """When the table lookup misses, fail loud — landing on the
    wrong row count is the kind of silent bug that breaks consensus
    one round in twenty and is hell to debug."""
    info = _c4_info(shard_count=4)
    # Table missing entries for shards 2 and 3.
    partial_table = {
        "en/c4-train.00000-of-00004.json.gz": 356_317,
        "en/c4-train.00001-of-00004.json.gz": 356_317,
    }
    with _patch_hf_info({"allenai/c4": info}), _patch_row_table(partial_table):
        # Iterate seeds until we hit a seed that picks one of the
        # missing shards. With 4 shards we expect to hit it within
        # the first ~10 seeds with overwhelming probability.
        raised = False
        for s in range(40):
            try:
                eval_shard_pick.pick_shard_for_source(
                    repo_id="allenai/c4", name="en", int_seed=s,
                )
            except KeyError as e:
                assert "row-count table" in str(e)
                raised = True
                break
        assert raised, (
            "Expected KeyError for missing table entry across 40 seeds — "
            "either the partial-table fixture didn't cover any picked "
            "shard, or the loud-failure path is suppressed."
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


# -----------------------------------------------------------------------
# D6 — Sort stability
# -----------------------------------------------------------------------

def test_shard_list_sort_is_independent_of_hf_return_order():
    """HF's siblings list order is not guaranteed to be stable. The
    pick must be insensitive to the order siblings come back in.

    We build two infos with the SAME files in DIFFERENT orders,
    flush caches between them, and verify the pick is identical for
    a fixed seed."""
    files = [f"en/c4-train.{i:05d}-of-00016.json.gz" for i in range(16)]
    table = {f: 356_317 for f in files}

    import random
    rng1 = random.Random(1)
    rng2 = random.Random(2)
    files_order_a = list(files)
    rng1.shuffle(files_order_a)
    files_order_b = list(files)
    rng2.shuffle(files_order_b)

    info_a = SimpleNamespace(siblings=[_sibling(f) for f in files_order_a], sha="s")
    info_b = SimpleNamespace(siblings=[_sibling(f) for f in files_order_b], sha="s")

    with _patch_hf_info({"allenai/c4": info_a}), _patch_row_table(table):
        a = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0xdeadbeef,
        )
    _clear_caches()
    with _patch_hf_info({"allenai/c4": info_b}), _patch_row_table(table):
        b = eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=0xdeadbeef,
        )

    assert a.shard_path == b.shard_path
    assert a.in_shard_offset == b.in_shard_offset


def test_filter_drops_validation_and_other_configs():
    """`_KNOWN_SOURCES["allenai/c4", "en"]` must NOT pick up
    `en/c4-validation.*`, `en.noblocklist/...`, or `multilingual/...`
    files. A silent over-match here would change the reachable pool
    and could break consensus during a rollout that adds noise files."""
    info = _c4_info(shard_count=8, with_noise=True)
    table = {
        f"en/c4-train.{i:05d}-of-00008.json.gz": 356_317
        for i in range(8)
    }
    with _patch_hf_info({"allenai/c4": info}), _patch_row_table(table):
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
# D7 — Bundled-table integration
# -----------------------------------------------------------------------

def test_bundled_c4_row_count_table_loads():
    """The checked-in JSON resource is loadable via importlib.resources
    and parses as a dict[str, int]. This is the production loading
    path; if importlib.resources can't find the file (wrong
    `package_data`, missing `__init__.py`, etc.), the validator boots
    fine but blows up on the first eval round."""
    table = eval_shard_pick._load_row_count_table("data/c4_en_shard_rows.json")
    assert isinstance(table, dict)
    assert len(table) >= 1, "Bundled table is empty — bootstrap entries missing"
    for k, v in table.items():
        assert k.startswith("en/c4-train.")
        assert k.endswith(".json.gz")
        assert isinstance(v, int) and v > 0


def test_bundled_c4_table_contains_shard_zero_and_matches_empirical():
    """Spot-check: the bootstrap table must contain shard 0 with the
    empirically-verified row count. Hard-codes the verified value so
    a future regeneration that produces a different count for shard 0
    fails loud — that's almost certainly a bug in the regenerator
    rather than a real change in the upstream dataset."""
    table = eval_shard_pick._load_row_count_table("data/c4_en_shard_rows.json")
    assert table.get("en/c4-train.00000-of-01024.json.gz") == 356_317


def test_revision_pin_override_is_threaded_through():
    """`revision_override` must reach `_resolve_revision` (and from
    there `_list_shards`). Without this, the per-source pin from
    `DataCfg.eval_source_revision_pin` would silently default to
    the policy revision and operators would think they pinned when
    they hadn't."""
    info = _c4_info(shard_count=4)
    table = {
        f"en/c4-train.{i:05d}-of-00004.json.gz": 356_317
        for i in range(4)
    }
    seen_revisions: list[str] = []

    def _capturing_dataset_info(self, repo_id, *, revision=None, **_):
        seen_revisions.append(revision)
        return info

    with patch.object(
        eval_shard_pick.HfApi, "dataset_info", _capturing_dataset_info,
    ), _patch_row_table(table):
        eval_shard_pick.pick_shard_for_source(
            repo_id="allenai/c4", name="en", int_seed=1,
            revision_override="my-explicit-sha",
        )
    # The first call is `_resolve_revision("allenai/c4", "my-explicit-sha")`.
    assert seen_revisions[0] == "my-explicit-sha"

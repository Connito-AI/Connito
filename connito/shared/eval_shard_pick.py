"""Seeded shard-pick + mod-offset for eval-data sampling.

Replaces the older `.shuffle(seed) + .skip(small)` pattern, which could only
ever reach the head of each shard — leaving most of every source permanently
unreachable and small enough for a miner to memorize.

The pick scheme decomposes "skip N rows into the dataset" as:

    pick_shard(N // rows_per_shard) + skip(N % rows_per_shard)

- The shard pick is O(1): it selects which HF parquet/json.gz file to
  open via `data_files=[chosen]`. No bytes downloaded by the pick
  itself; the subsequent stream fetches only that one shard.
- The in-shard offset is bounded by ONE shard's row count (a few
  hundred thousand) instead of the dataset's (hundreds of millions).
  Worst-case decode-and-discard is seconds per source per round.

Across rounds with rotating seeds, every shard is eventually picked
and every row within it is eventually offset-to. Whole-dataset reach
with per-round cost bounded by one shard.

Sizing the in-shard offset bound — path B (safe floor) vs path A (per-shard table)
---------------------------------------------------------------------------------

`.skip(N)` past end-of-stream silently exhausts the iterator; the
default `interleave_datasets(stopping_strategy="first_exhausted")` then
collapses the whole eval round to zero (or short) batches. The offset
mod-bound must therefore be ≤ (actual shard rows) − (downstream
consumption + safety margin).

Two ways to learn "actual shard rows":

  * Path A — per-shard exact counts. Maximum reach (every row
    reachable). Cost: enumerate every shard once (hours for C4-en
    json.gz which has no footer; one HTTP-range read per parquet
    shard for Nemotron).

  * Path B — a per-source SAFE FLOOR constant. Lossy by
    construction: rows past the floor are unreachable. For
    publisher-balanced datasets (C4: shard sizes 356,317 or 356,318
    across all spot-checked samples) the loss is small enough that
    we pay it in exchange for zero per-shard accounting.

This module implements path B. The policy registry declares either:

  * `row_count_source="constant"` → use `safe_floor_rows` as the bound
    for every shard. Validated at module load against a small set of
    `verified_shard_rows` so a typo or stale config (safe_floor too
    high relative to actual shard sizes) raises BEFORE any round
    picks land short.

  * `row_count_source="parquet_footer"` → read `num_rows` from the
    parquet footer at pick time. Cheap (~32 KB HTTP-range fetch).
    Bound = `actual_rows − min_headroom_rows`.

Anti-memorization properties are unchanged from today:
    * Shard selection AND in-shard offset are both seed-derived, so a
      miner who can compute `combined_seed` can predict the exact rows.
    * The defense is therefore (i) `combined_seed` itself mixes the
      late-bound MinerCommit2 block hash (see
      `connito/shared/cycle.py:_get_minercommit2_block_hash`), and
      (ii) the per-round reachable pool (one shard × `safe_floor` rows
      ≈ ~340 k) is wide enough that overfitting within the remaining
      commit window is infeasible.

Consensus depends on every validator deriving an IDENTICAL shard list.
That requires:
    1. A pinned dataset revision (commit SHA, not `main`) per source.
       Without this, HF reordering / adding / replacing shards mid-
       rollout would cause two validators to pick different rows for
       the same seed and break weight consensus for that round.
    2. Deterministic file-list sort (plain lexicographic on the full
       `siblings.rfilename`).
    3. The `safe_floor_rows` / `min_headroom_rows` constants must
       match across validators — they live in code (this module), not
       config, so the rollout discipline is just "deploy the same
       commit." Operators MUST re-verify `verified_shard_rows` and
       bump `safe_floor_rows` together if the upstream dataset is
       ever re-uploaded with different shard sizes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from huggingface_hub import HfApi

from connito.shared.app_logging import structlog
from connito.shared.helper import h256_int


logger = structlog.get_logger(__name__)


# Per-source shard-list filter, revision pin, and offset-bound policy.
#
# `path_prefix`/`path_suffix` match against `siblings.rfilename`. Keep them
# narrow: over-matching (picking up `validation/` files, say) fails silently at
# config time and only shows up as a cross-validator consensus break.
#
# `revision` should pin a commit SHA. `"main"` moves under us and risks the
# same divergence; operators override via `eval_source_revision_pin`.
#
# `row_count_source` decides how the in-shard offset bound is resolved:
#   - "constant": use `safe_floor_rows`. Required for footerless formats
#     (json.gz); rows past the floor are unreachable by construction.
#   - "parquet_footer": read `num_rows` from the footer at pick time
#     (~32 KB range request); bound is `actual_rows - min_headroom_rows`.
#
# `min_headroom_rows` is what must remain readable after the skip lands, sized
# against the eval pipeline's per-round consumption with a wide safety margin.
# `safe_floor_rows` must stay below `min(verified_shard_rows) -
# min_headroom_rows`; both invariants are enforced at module load against
# `verified_shard_rows`, a spot-check sample of counted shards.
@dataclass(frozen=True)
class _SourceShardPolicy:
    path_prefix: str
    path_suffix: tuple[str, ...]
    revision: str
    row_count_source: str  # "constant" | "parquet_footer" | "verified_table"
    safe_floor_rows: int | None = None
    min_headroom_rows: int = 10_000
    verified_shard_rows: dict[str, int] = field(default_factory=dict)
    # Overrides `_SHARD_NAME_FILTER` for sources that don't use the
    # `train`/`part_` leaf convention. Listing-based policies only;
    # `verified_table` takes its shard list from the table.
    leaf_name_pattern: str | None = None
    # Load the shard through this generic builder against a resolved
    # `hf_hub_url`. Required for repos shipping a loading script, which
    # `load_dataset(repo_id, ...)` would execute — defeating both the shard
    # pick and the `trust_remote_code` opt-out.
    load_builder: str | None = None


# Known sources. Add new entries here, NOT via config — the consensus
# rules require every validator to use the same policy.
_KNOWN_SOURCES: dict[tuple[str, str | None], _SourceShardPolicy] = {
    ("allenai/c4", "en"): _SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",  # operator-overridable; see DataCfg
        row_count_source="constant",
        # C4 shards are publisher-balanced at ~356 k rows, so a floor of
        # 340 000 costs ~4.6 % coverage per shard and avoids enumerating all
        # 1 024 shards. Re-spot-check `verified_shard_rows` and adjust this
        # floor before pointing at a re-uploaded revision.
        safe_floor_rows=340_000,
        min_headroom_rows=10_000,
        verified_shard_rows={
            "en/c4-train.00000-of-01024.json.gz": 356_317,
            "en/c4-train.00001-of-01024.json.gz": 356_318,
            "en/c4-train.00500-of-01024.json.gz": 356_317,
            "en/c4-train.01023-of-01024.json.gz": 356_317,
        },
    ),
    ("nvidia/Nemotron-CC-Math-v1", "4plus"): _SourceShardPolicy(
        path_prefix="4plus/",
        path_suffix=(".parquet",),
        revision="main",
        row_count_source="parquet_footer",
        # safe_floor_rows omitted — parquet footer provides the exact
        # count per shard at pick time.
        min_headroom_rows=10_000,
    ),
    ("joelniklaus/Multi_Legal_Pile", "all_all"): _SourceShardPolicy(
        # Native data files, deliberately not the `all_all` builder script:
        # it needs `trust_remote_code` and streams external repos that can't
        # be pinned or row-counted. Eval is therefore a subset of the training
        # distribution, which is acceptable.
        path_prefix="data/",
        path_suffix=(".jsonl.xz",),
        # Pinned at registration time (2026-07-21). Bump together with a
        # re-count of the table if the dataset is ever re-uploaded.
        revision="911e1d214162fd11d2c78d3f1428cbfcbe07782c",
        row_count_source="verified_table",
        min_headroom_rows=10_000,
        # `.jsonl.xz` has no footer, so rows were counted offline at the
        # pinned revision and frozen here. The table doubles as the allowlist;
        # shards too small to leave a safe offset above the headroom floor are
        # deliberately absent.
        load_builder="json",
        verified_shard_rows={
            "data/bg/legislation/bulgaria_marcell.jsonl.xz": 29_549,
            "data/cs/caselaw/czechia_constitutional_court.jsonl.xz": 73_086,
            "data/cs/caselaw/czechia_supreme_administrative_court.jsonl.xz": 52_660,
            "data/cs/caselaw/czechia_supreme_court.jsonl.xz": 111_977,
            "data/da/legislation/denmark_ddsc.jsonl.xz": 64_043,
            "data/de/caselaw/germany_openlegaldata.jsonl.xz": 201_676,
            "data/de/caselaw/switzerland_entscheidsuche.jsonl.xz": 308_612,
            "data/de/legislation/germany_openlegaldata.jsonl.xz": 52_918,
            "data/de/legislation/switzerland_lexfind.jsonl.xz": 16_981,
            "data/en/legislation/uk_uk_lex.jsonl.xz": 36_499,
            "data/fr/caselaw/france_cass.jsonl.xz": 113_844,
            "data/fr/caselaw/luxembourg_judoc.jsonl.xz": 37_902,
            "data/fr/caselaw/switzerland_entscheidsuche.jsonl.xz": 237_734,
            "data/fr/legislation/belgium_ejustice.jsonl.xz": 10_613,
            "data/fr/legislation/switzerland_lexfind.jsonl.xz": 10_680,
            "data/hu/legislation/hungary_marcell.jsonl.xz": 26_821,
            "data/it/caselaw/switzerland_entscheidsuche.jsonl.xz": 69_653,
            "data/nl/legislation/belgium_ejustice.jsonl.xz": 10_556,
            "data/pl/legislation/poland_marcell.jsonl.xz": 27_485,
            "data/pt/caselaw/brazil_cjpg_0.jsonl.xz": 3_489_624,
            "data/pt/caselaw/brazil_cjpg_1.jsonl.xz": 3_213_178,
            "data/pt/caselaw/brazil_cjpg_2.jsonl.xz": 3_094_216,
            "data/pt/caselaw/brazil_cjpg_3.jsonl.xz": 3_019_375,
            "data/pt/caselaw/brazil_cjpg_4.jsonl.xz": 1_252_241,
            "data/pt/caselaw/brazil_creta.jsonl.xz": 3_128_292,
            "data/pt/caselaw/brazil_rulingbr.jsonl.xz": 10_623,
            "data/ro/legislation/romania_marcell.jsonl.xz": 163_264,
            "data/sk/legislation/slovakia_marcell.jsonl.xz": 13_055,
            "data/sl/legislation/slovenia_marcell.jsonl.xz": 24_445,
        },
    ),
}


_SHARD_NAME_FILTER = re.compile(r"(train|part_)", re.IGNORECASE)


def _validate_policy(key: tuple[str, str | None], policy: _SourceShardPolicy) -> None:
    """Enforce per-policy invariants at module load.

    Failure here is a deployment bug, not a runtime condition. Raise
    loud (and at import time) so the validator process refuses to
    boot rather than running a misconfigured eval that quietly draws
    biased samples or collapses rounds.
    """
    repo_id, name = key
    if policy.row_count_source not in {"constant", "parquet_footer", "verified_table"}:
        raise ValueError(
            f"Policy {repo_id}/{name}: unknown row_count_source "
            f"{policy.row_count_source!r}"
        )
    if policy.min_headroom_rows <= 0:
        raise ValueError(
            f"Policy {repo_id}/{name}: min_headroom_rows must be > 0"
        )
    if policy.row_count_source == "verified_table":
        # The table is the allowlist: every listed shard must leave a valid
        # offset after headroom. Too-small shards belong out of the table, not
        # zero-bounded at pick time.
        if not policy.verified_shard_rows:
            raise ValueError(
                f"Policy {repo_id}/{name}: row_count_source='verified_table' "
                f"requires a non-empty verified_shard_rows table (it doubles "
                f"as the shard allowlist)"
            )
        for shard_path, rows in policy.verified_shard_rows.items():
            if rows <= policy.min_headroom_rows:
                raise ValueError(
                    f"Policy {repo_id}/{name}: verified shard {shard_path!r} "
                    f"has {rows} rows ≤ min_headroom_rows "
                    f"({policy.min_headroom_rows}); no safe offset exists. "
                    f"Remove it from the table."
                )
            if not shard_path.startswith(policy.path_prefix) or not shard_path.endswith(
                policy.path_suffix
            ):
                raise ValueError(
                    f"Policy {repo_id}/{name}: table entry {shard_path!r} does "
                    f"not match path_prefix/path_suffix — typo in the table?"
                )
    if policy.row_count_source == "constant":
        if policy.safe_floor_rows is None or policy.safe_floor_rows <= 0:
            raise ValueError(
                f"Policy {repo_id}/{name}: row_count_source='constant' "
                f"requires safe_floor_rows > 0"
            )
        if not policy.verified_shard_rows:
            raise ValueError(
                f"Policy {repo_id}/{name}: row_count_source='constant' "
                f"requires at least one verified_shard_rows entry so the "
                f"safe_floor can be sanity-checked at module load"
            )
        threshold = policy.safe_floor_rows + policy.min_headroom_rows
        for shard_path, rows in policy.verified_shard_rows.items():
            if rows < threshold:
                raise ValueError(
                    f"Policy {repo_id}/{name}: verified shard {shard_path!r} "
                    f"has {rows} rows but safe_floor_rows "
                    f"({policy.safe_floor_rows}) + min_headroom_rows "
                    f"({policy.min_headroom_rows}) = {threshold} > {rows}. "
                    f"An offset draw could over-skip the shard and "
                    f"collapse the round. Lower safe_floor_rows."
                )
    elif policy.row_count_source == "parquet_footer":
        if policy.safe_floor_rows is not None:
            # Not an error, but flag which value wins.
            logger.warning(
                "Policy has both row_count_source='parquet_footer' and "
                "safe_floor_rows set; safe_floor_rows will be ignored",
                repo_id=repo_id, name=name,
            )


# Validate at import: a misconfigured policy reaching production silently is
# exactly the failure this module exists to prevent.
for _key, _policy in _KNOWN_SOURCES.items():
    _validate_policy(_key, _policy)


def _policy_for(path: str, name: str | None) -> _SourceShardPolicy:
    key = (path, name)
    if key not in _KNOWN_SOURCES:
        raise KeyError(
            f"No shard-pick policy registered for source ({path!r}, {name!r}). "
            f"Add an entry to `_KNOWN_SOURCES` in `eval_shard_pick.py` and "
            f"verify that data_files=[shard] yields the same rows as the "
            f"canonical load path before flipping the feature flag."
        )
    return _KNOWN_SOURCES[key]


@lru_cache(maxsize=8)
def _resolve_revision(repo_id: str, requested: str) -> str:
    """Resolve `requested` (which may be `main`) to a commit SHA so the
    pin used for shard listing matches the pin used for shard loading.

    Even when an operator config asks for `main`, we resolve it ONCE per
    validator-process startup and reuse that SHA for the rest of the
    process's life. That avoids the failure mode where two validators
    boot at different times and see different `main` heads.
    """
    api = HfApi()
    info = api.dataset_info(repo_id, revision=requested)
    sha = getattr(info, "sha", None)
    if not sha:
        # Older HF versions don't always expose `sha`; ship the requested
        # value rather than crash. Divergence would surface as mismatched
        # losses.
        logger.warning(
            "HF dataset_info did not expose `sha`; using requested revision verbatim",
            repo_id=repo_id, requested=requested,
        )
        return requested
    return sha


@lru_cache(maxsize=8)
def _list_shards(repo_id: str, name: str | None, revision: str) -> tuple[str, ...]:
    """Return the deterministically-sorted shard list for a source.

    Cached per (repo_id, name, revision). Sort is plain lex over the
    full `rfilename` string so any validator computing this against
    the same revision gets the same tuple.
    """
    policy = _policy_for(repo_id, name)
    if policy.row_count_source == "verified_table":
        # The frozen table is the allowlist. Listing from the HF API here
        # would reintroduce the consensus hazard it exists to remove.
        return tuple(sorted(policy.verified_shard_rows))
    info = HfApi().dataset_info(repo_id, revision=revision)
    name_filter = (
        re.compile(policy.leaf_name_pattern, re.IGNORECASE)
        if policy.leaf_name_pattern
        else _SHARD_NAME_FILTER
    )
    filtered = []
    for f in info.siblings:
        rf = f.rfilename
        if not rf.startswith(policy.path_prefix):
            continue
        if not rf.endswith(policy.path_suffix):
            continue
        # Guards against sidecar/metadata files that share the prefix and
        # suffix. Intentionally narrow, per source format.
        leaf = rf.split("/")[-1]
        if not name_filter.search(leaf):
            continue
        filtered.append(rf)
    if not filtered:
        raise RuntimeError(
            f"No shards matched policy for ({repo_id!r}, {name!r}, rev={revision!r}). "
            f"Check `_SourceShardPolicy.path_prefix`/`path_suffix` against the actual "
            f"dataset layout — a silent mis-match here will break consensus."
        )
    return tuple(sorted(filtered))


def _shard_rows_via_parquet_footer(repo_id: str, revision: str, shard_path: str) -> int:
    """Read num_rows from the parquet footer without downloading the full file."""
    # Lazy: eager pyarrow import slows test collection.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id, shard_path, repo_type="dataset", revision=revision,
        )
        return int(pq.read_metadata(local).num_rows)
    except Exception as e:
        # Re-raise with the (repo_id, shard, revision) context so the
        # validator log carries enough to diagnose without grepping.
        raise RuntimeError(
            f"Failed to read parquet footer for "
            f"{repo_id}@{revision}:{shard_path}: {type(e).__name__}: {e}"
        ) from e


def _resolve_offset_bound(
    *, repo_id: str, name: str | None, revision: str, shard_path: str,
    policy: _SourceShardPolicy,
) -> int:
    """Return the upper bound for the in-shard offset modulo.

    Always leaves `min_headroom_rows` rows in the source after the
    skip lands, so the downstream pipeline (filter + interleave +
    collator) can fill a round without exhausting the source.

    For `constant` policies the safe_floor is pre-validated at module
    load (verified samples ≥ safe_floor + min_headroom).

    For `parquet_footer` policies the bound is computed at pick time
    from the actual shard's row count. A shard small enough that
    `actual_rows ≤ min_headroom_rows` is a deployment error — there's
    no safe offset to pick — and raises rather than silently
    collapsing the round.
    """
    if policy.row_count_source == "constant":
        # Module-load validation guarantees safe_floor_rows is set.
        assert policy.safe_floor_rows is not None
        return policy.safe_floor_rows

    if policy.row_count_source == "verified_table":
        # Module-load validation guarantees the shard is in the table
        # with rows > min_headroom_rows (the table is the allowlist
        # `_list_shards` picks from).
        rows = policy.verified_shard_rows[shard_path]
        return rows - policy.min_headroom_rows

    if policy.row_count_source == "parquet_footer":
        actual_rows = _shard_rows_via_parquet_footer(repo_id, revision, shard_path)
        bound = actual_rows - policy.min_headroom_rows
        if bound <= 0:
            raise RuntimeError(
                f"Shard {shard_path!r} for {repo_id}/{name} has {actual_rows} "
                f"rows, which is below the configured min_headroom_rows "
                f"({policy.min_headroom_rows}). No safe offset exists; this "
                f"shard is too small for the eval pipeline's per-round "
                f"consumption budget. Either drop the source from the "
                f"shard-pick path or shrink min_headroom_rows."
            )
        return bound

    raise ValueError(f"Unknown row_count_source: {policy.row_count_source!r}")


@dataclass(frozen=True)
class ShardPick:
    """Result of one seed-driven pick for one source.

    `offset_bound` is the integer the hash was mod-ed by — i.e. the
    maximum allowed offset + 1. For `constant` policies it equals the
    policy's `safe_floor_rows`. For `parquet_footer` policies it
    equals `actual_shard_rows − min_headroom_rows`.

    `shard_rows` is the actual shard row count when known
    (parquet_footer source); for constant-source picks it is set to
    `offset_bound` because we deliberately don't enumerate row counts
    for that path. Code that wants "is offset < shard_rows" should
    check against `shard_rows`; code that wants "the chosen mod
    bound" should check against `offset_bound`.
    """
    repo_id: str
    name: str | None
    revision: str
    shard_path: str
    offset_bound: int
    shard_rows: int  # for the constant path this equals offset_bound (we don't know the true count)
    in_shard_offset: int
    # Propagated from the policy: when set, `load_streaming_shard` loads
    # via this generic builder against a resolved URL (script-bypass).
    load_builder: str | None = None


def pick_shard_for_source(
    *,
    repo_id: str,
    name: str | None,
    int_seed: int,
    revision_override: str | None = None,
) -> ShardPick:
    """Pick one shard and a uniform in-shard offset for one source.

    Deterministic from (repo_id, name, revision, int_seed). All
    validators on the same revision pin produce the same pick.

    `int_seed` is the same integer derived from `combined_seed` that
    feeds the existing `.shuffle(seed=int_seed, ...)` call — see
    `get_tokenised_dataset` for the construction:
        `int_seed = int(str(seed)[:8], 16)`.
    Reusing it keeps the seed wiring identical to today.
    """
    policy = _policy_for(repo_id, name)
    requested_revision = revision_override or policy.revision
    revision = _resolve_revision(repo_id, requested_revision)

    shards = _list_shards(repo_id, name, revision)
    if not shards:  # _list_shards already raises but be explicit
        raise RuntimeError(
            f"Empty shard list for ({repo_id!r}, {name!r}, rev={revision!r})"
        )

    # Two independent hash draws off the same seed. Different
    # domain-separator strings ensure shard choice and in-shard offset
    # don't correlate.
    shard_idx = h256_int("eval_shard_pick", repo_id, str(name), int_seed) % len(shards)
    chosen = shards[shard_idx]
    offset_bound = _resolve_offset_bound(
        repo_id=repo_id, name=name, revision=revision,
        shard_path=chosen, policy=policy,
    )
    # `% offset_bound` lets the mod be either the safe_floor or the
    # footer-derived actual_rows - headroom; either way, after `.skip`
    # the source retains at least `min_headroom_rows` for the
    # downstream pipeline.
    offset = (
        h256_int("eval_in_shard_offset", repo_id, str(name), int_seed) % offset_bound
    )

    # Constant policies don't know the real shard size, so report the bound as
    # `shard_rows` — callers asserting `offset < shard_rows` still see a
    # correct invariant.
    if policy.row_count_source in {"parquet_footer", "verified_table"}:
        actual_shard_rows = offset_bound + policy.min_headroom_rows
    else:
        actual_shard_rows = offset_bound

    return ShardPick(
        repo_id=repo_id,
        name=name,
        revision=revision,
        shard_path=chosen,
        offset_bound=offset_bound,
        shard_rows=actual_shard_rows,
        in_shard_offset=offset,
        load_builder=policy.load_builder,
    )


def load_streaming_shard(
    pick: ShardPick,
    *,
    split_name: str = "train",
    extra_load_kwargs: dict[str, Any] | None = None,
):
    """Open a streaming HF dataset reading ONLY the picked shard.

    `data_files=` bypasses the dataset's loading script (if any), so the
    returned schema is whatever the raw file format gives. The caller
    is expected to `.select_columns([text_column])` and `.map(...)` the
    result the same way today's `_load_streaming_split` does — so the
    downstream pipeline is unchanged.

    Equivalence with the canonical load path must be verified per
    source before flipping the feature flag for production.
    """
    # Lazy, so this module stays importable without the datasets stack.
    from datasets import load_dataset

    if pick.load_builder:
        # The repo ships a loading script that `load_dataset(repo_id, ...)`
        # would execute, streaming different files entirely. A generic builder
        # reads exactly the picked shard and needs no `trust_remote_code`.
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(
            pick.repo_id, pick.shard_path, repo_type="dataset", revision=pick.revision
        )
        load_kwargs = {"data_files": [url], "streaming": True}
        if extra_load_kwargs:
            load_kwargs.update(extra_load_kwargs)
        ds = load_dataset(pick.load_builder, **load_kwargs)
    else:
        load_kwargs = {
            "data_files": [pick.shard_path],
            "streaming": True,
            "revision": pick.revision,
        }
        if extra_load_kwargs:
            load_kwargs.update(extra_load_kwargs)
        ds = load_dataset(pick.repo_id, **load_kwargs)
    if split_name in ds:
        return ds[split_name]
    # `data_files=` with a single file lands the rows under "train" by
    # default. Fall back to whatever the only split is.
    only = next(iter(ds.keys()))
    if only != split_name:
        logger.debug(
            "Streaming shard had no `train` split; using only split present",
            repo_id=pick.repo_id, only_split=only,
        )
    return ds[only]

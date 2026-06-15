"""Seeded shard-pick + mod-offset for eval-data sampling.

Replaces the older `.shuffle(seed) + .skip(small)` pattern in
`get_tokenised_dataset` for the validator eval path, which can only
reach the head ~`shuffle_buffer + skip_max` rows of whichever shard a
seed permutes to position 0. With ~50 k + 50 k that's the first ~100 k
rows of each shard — i.e. ~70 % of every C4-en shard is unreachable
forever regardless of how long the network runs.

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
# `path_prefix`/`path_suffix` are matched against the dataset's
# `siblings.rfilename` list. The pair is intentionally narrow rather
# than a glob: silent over-matching (e.g. picking up `validation/`
# files or `noblocklist` variants) would not raise at config time and
# would only manifest as cross-validator consensus breaks. Keep these
# explicit per source.
#
# `revision` pins to a commit SHA when possible. `"main"` is a moving
# target; using it accepts a small consensus-break risk in exchange
# for not having to ship a config update every time HF re-uploads.
# Operators who care should override via `eval_source_revision_pin`
# in DataCfg.
#
# `row_count_source` selects how the in-shard offset bound is
# resolved:
#   - "constant": use `safe_floor_rows` directly. Required for
#     formats without a footer (json.gz). Lossy by construction —
#     rows past the floor are unreachable. Validated at module load
#     against `verified_shard_rows`.
#   - "parquet_footer": read `num_rows` from the parquet footer at
#     pick time. Cheap (~32 KB HTTP range). Bound becomes
#     `actual_rows − min_headroom_rows`.
#
# `safe_floor_rows`: the offset bound for constant-source policies.
# Must be set strictly less than `min(verified_shard_rows.values())
# − min_headroom_rows`, enforced at module load.
#
# `min_headroom_rows`: rows guaranteed to remain available in the
# source AFTER the skip lands. Sized for the downstream pipeline's
# per-round consumption — see `dataloader.py:281` comment:
# `~max_eval_batches × world_size / vali_fraction ≈ 5 000` total /
# 2 sources = ~2 500 per source. Default 10 000 gives a 4× safety
# margin for filter / split / interleave variance and unusual configs.
#
# `verified_shard_rows`: empirically-counted row counts for a small
# spot-check sample of shards (not the full set). Used at module load
# to validate `safe_floor_rows`, and at runtime as documentation of
# what the policy was sized against. Empty for footer-based sources.
@dataclass(frozen=True)
class _SourceShardPolicy:
    path_prefix: str
    path_suffix: tuple[str, ...]
    revision: str
    row_count_source: str  # "constant" | "parquet_footer"
    safe_floor_rows: int | None = None
    min_headroom_rows: int = 10_000
    verified_shard_rows: dict[str, int] = field(default_factory=dict)


# Known sources. Add new entries here, NOT via config — the consensus
# rules require every validator to use the same policy.
_KNOWN_SOURCES: dict[tuple[str, str | None], _SourceShardPolicy] = {
    ("allenai/c4", "en"): _SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",  # operator-overridable; see DataCfg
        row_count_source="constant",
        # Empirically all four spot-checked shards (0, 1, 500, 1023)
        # are 356 317 or 356 318 rows; C4 is publisher-balanced. The
        # safe floor (340 000) sits ~16 k below the minimum spot-check
        # and ~6 k above the (safe_floor + headroom) module-load
        # threshold of 350 000. Coverage loss per shard is
        # (356 317 − 340 000) / 356 317 ≈ 4.6 %, accepted in exchange
        # for not enumerating all 1 024 shards. If C4 is ever
        # re-uploaded with materially different shard sizes,
        # `verified_shard_rows` MUST be re-spot-checked and the floor
        # adjusted before flipping the gate on the new revision.
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
    if policy.row_count_source not in {"constant", "parquet_footer"}:
        raise ValueError(
            f"Policy {repo_id}/{name}: unknown row_count_source "
            f"{policy.row_count_source!r}"
        )
    if policy.min_headroom_rows <= 0:
        raise ValueError(
            f"Policy {repo_id}/{name}: min_headroom_rows must be > 0"
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
            # Not strictly an error — but flag the inconsistency so a
            # future reader doesn't wonder which value is used.
            logger.warning(
                "Policy has both row_count_source='parquet_footer' and "
                "safe_floor_rows set; safe_floor_rows will be ignored",
                repo_id=repo_id, name=name,
            )


# Validate every registered policy at import time. A misconfigured
# policy reaching production silently is exactly the kind of bug this
# system is designed to surface.
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
        # Fall back to the requested value verbatim. HF older versions
        # may not expose `sha` consistently — better to ship the request
        # string than to crash, and operators will see the consensus
        # divergence loudly via mismatched losses if it bites.
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
    info = HfApi().dataset_info(repo_id, revision=revision)
    filtered = []
    for f in info.siblings:
        rf = f.rfilename
        if not rf.startswith(policy.path_prefix):
            continue
        if not rf.endswith(policy.path_suffix):
            continue
        # Final-segment shape check guards against accidentally pulling
        # in unrelated files that happen to share the prefix/suffix
        # (e.g. metadata, sidecar files). The "train" / "part_" check
        # is per-source-format and intentionally narrow.
        leaf = rf.split("/")[-1]
        if not _SHARD_NAME_FILTER.search(leaf):
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
    # Lazy import — pyarrow is already a dependency of `datasets` but
    # importing it eagerly at module top-level slows test imports.
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

    # For constant policies we don't know the actual shard size;
    # surface `offset_bound` as `shard_rows` so older callers (notebook
    # / tests) that check "offset < shard_rows" still see the right
    # invariant. The parquet path can fill in the real count.
    if policy.row_count_source == "parquet_footer":
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
    # Lazy: `datasets` is already imported by the caller side, but
    # keeping this module importable without the full datasets stack
    # helps unit tests.
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {
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

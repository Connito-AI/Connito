"""Seeded shard-pick + mod-offset for eval-data sampling.

Replaces the older `.shuffle(seed) + .skip(small)` pattern in
`get_tokenised_dataset` for the validator eval path, which can only
reach the head ~`shuffle_buffer + skip_max` rows of whichever shard a
seed permutes to position 0. With ~50 k + 50 k that's the first ~100 k
rows of each shard — i.e. ~30 % of every C4-en shard is unreachable
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

The in-shard offset uses `h256_int(...) % rows_in_this_shard` rather
than `rng.randrange(0, fixed_ceiling)`. The hash → mod approach
defers the bound to pick time, so each shard contributes a uniform
distribution over its OWN size without the per-source code needing a
checked-in min(shard_sizes) constant. Modulo bias is ~rows / 2**256,
negligible.

Anti-memorization properties are unchanged from today:
    * Shard selection AND in-shard offset are both seed-derived, so a
      miner who can compute `combined_seed` can predict the exact rows.
    * The defense is therefore (i) `combined_seed` itself mixes the
      late-bound MinerCommit2 block hash (see
      `connito/shared/cycle.py:_get_minercommit2_block_hash`), and
      (ii) the per-round reachable pool (one shard ≈ ~350 k rows) is
      wide enough that overfitting within the remaining commit window
      is infeasible.

Consensus depends on every validator deriving an IDENTICAL shard list.
That requires:
    1. A pinned dataset revision (commit SHA, not `main`) per source.
       Without this, HF reordering / adding / replacing shards mid-
       rollout would cause two validators to pick different rows for
       the same seed and break weight consensus for that round.
    2. Deterministic file-list sort (plain lexicographic on the full
       `siblings.rfilename`).
    3. Shard row-counts agreed across validators. Parquet files
       carry `num_rows` in the footer (cheap HTTP-range read); json.gz
       files have no footer, so a checked-in JSON table is used.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from connito.shared.app_logging import structlog
from connito.shared.helper import h256_int


logger = structlog.get_logger(__name__)


# Per-source shard-list filter and row-count source.
#
# `path_prefix`/`path_suffix` are matched against the dataset's
# `siblings.rfilename` list. The pair is intentionally narrow rather
# than a glob: silent over-matching (e.g. picking up `validation/` files
# or `noblocklist` variants) would not raise at config time and would
# only manifest as cross-validator consensus breaks. Keep these
# explicit per source.
#
# `revision` pins to a commit SHA when possible. `"main"` is a moving
# target; using it accepts a small consensus-break risk in exchange
# for not having to ship a config update every time HF re-uploads.
# Operators who care should override via `eval_source_revision_pin`
# in DataCfg.
#
# `row_count_source` controls how rows-per-shard is resolved:
#   - "parquet_footer": read `pq.read_metadata(path).num_rows` from the
#     parquet file's footer. One HTTP-range fetch (~32 KB), no row
#     decode. Validators agree because parquet footers are
#     content-determined.
#   - "table": look up rows from a checked-in JSON file (for json.gz
#     sources where the format has no footer).
@dataclass(frozen=True)
class _SourceShardPolicy:
    path_prefix: str
    path_suffix: tuple[str, ...]
    revision: str
    row_count_source: str  # "parquet_footer" | "table"
    row_count_table_resource: str | None  # only used when source = "table"


# Known sources. Add new entries here, NOT via config — the consensus
# rules require every validator to use the same policy.
_KNOWN_SOURCES: dict[tuple[str, str | None], _SourceShardPolicy] = {
    ("allenai/c4", "en"): _SourceShardPolicy(
        path_prefix="en/",
        path_suffix=(".json.gz",),
        revision="main",  # operator-overridable; see DataCfg
        row_count_source="table",
        row_count_table_resource="data/c4_en_shard_rows.json",
    ),
    ("nvidia/Nemotron-CC-Math-v1", "4plus"): _SourceShardPolicy(
        path_prefix="4plus/",
        path_suffix=(".parquet",),
        revision="main",
        row_count_source="parquet_footer",
        row_count_table_resource=None,
    ),
}


_SHARD_NAME_FILTER = re.compile(r"(train|part_)", re.IGNORECASE)


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


def _load_row_count_table(resource_path: str) -> dict[str, int]:
    """Load a checked-in shard → rows table from package data.

    Resource paths are relative to `connito.shared`, e.g.
    `data/c4_en_shard_rows.json`.
    """
    # importlib.resources.files is the modern, package-friendly path
    # and works regardless of whether the package is installed or run
    # from source.
    pkg_data = resources.files("connito.shared").joinpath(resource_path)
    with pkg_data.open("rb") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _row_count_table(resource_path: str) -> dict[str, int]:
    return _load_row_count_table(resource_path)


def _shard_rows_via_parquet_footer(repo_id: str, revision: str, shard_path: str) -> int:
    """Read num_rows from the parquet footer without downloading the full file."""
    # Lazy import — pyarrow is already a dependency of `datasets` but
    # importing it eagerly at module top-level slows test imports.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id, shard_path, repo_type="dataset", revision=revision,
    )
    return int(pq.read_metadata(local).num_rows)


def _resolve_shard_rows(
    *, repo_id: str, name: str | None, revision: str, shard_path: str,
) -> int:
    policy = _policy_for(repo_id, name)
    if policy.row_count_source == "parquet_footer":
        return _shard_rows_via_parquet_footer(repo_id, revision, shard_path)
    elif policy.row_count_source == "table":
        assert policy.row_count_table_resource, \
            "row_count_source='table' requires row_count_table_resource"
        table = _row_count_table(policy.row_count_table_resource)
        if shard_path not in table:
            raise KeyError(
                f"Shard {shard_path!r} not in row-count table "
                f"{policy.row_count_table_resource!r}. Either the table is "
                f"out of date with the pinned revision, or the shard list "
                f"matched a file the table doesn't know about. Regenerate "
                f"the table or tighten the path filter."
            )
        return int(table[shard_path])
    else:  # pragma: no cover — guarded by dataclass typing in practice
        raise ValueError(f"Unknown row_count_source: {policy.row_count_source!r}")


@dataclass(frozen=True)
class ShardPick:
    """Result of one seed-driven pick for one source."""
    repo_id: str
    name: str | None
    revision: str
    shard_path: str
    shard_rows: int
    in_shard_offset: int

    @property
    def reachable_window_end(self) -> int:
        return self.shard_rows  # for logging / sanity checks


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
    shard_rows = _resolve_shard_rows(
        repo_id=repo_id, name=name, revision=revision, shard_path=chosen,
    )
    # `% shard_rows` lets the bound be the chosen shard's actual size,
    # so a smaller-than-average shard can't be over-skipped.
    offset = h256_int("eval_in_shard_offset", repo_id, str(name), int_seed) % shard_rows

    return ShardPick(
        repo_id=repo_id,
        name=name,
        revision=revision,
        shard_path=chosen,
        shard_rows=shard_rows,
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

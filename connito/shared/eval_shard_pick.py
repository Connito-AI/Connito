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
- The in-shard offset is bounded by ONE shard's row count instead of
  the dataset's (hundreds of millions).

That bound is only cheap while a shard holds a few hundred thousand
rows. `.skip(N)` on a streaming dataset is decode-and-discard — there
is no row-group seek — so the per-round cost is proportional to the
offset drawn, and the skip runs AFTER the character filter, making the
underlying decode `offset / pass_rate` rows. A source whose shards hold
millions of rows therefore decodes a large fraction of a shard every
round: measured at ~650 MB for a ~985 k-row shard, and ~12.5 GB for a
9.9 M-row one. `max_offset_rows` caps the draw for such sources; see
`_SourceShardPolicy`.

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
    3. Every validator must resolve the SAME policy for a source.
       The two registries satisfy this by different means. Built-in
       entries in `_KNOWN_SOURCES` live in code, so the discipline is
       "deploy the same commit" — which a mixed-version fleet can
       quietly violate. Policies served with the task bundle are
       fetched by every validator from the same locked `owner_url`
       and checked against the bundle hash, so they are identical by
       construction; the discipline there is that a policy-bearing
       task must not be published until the whole fleet can read one.
       Either way, operators MUST re-verify `verified_shard_rows` and
       bump `safe_floor_rows` together if the upstream dataset is
       ever re-uploaded with different shard sizes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    row_count_source: str  # "constant" | "parquet_footer" | "verified_table"
    safe_floor_rows: int | None = None
    min_headroom_rows: int = 10_000
    verified_shard_rows: dict[str, int] = field(default_factory=dict)
    # Override for `_SHARD_NAME_FILTER` when a source's data files don't
    # follow the `train`/`part_` leaf naming convention (e.g.
    # Multi_Legal_Pile ships `data/<lang>/<type>/<source>.jsonl.xz`).
    # Only consulted for listing-based policies; `verified_table`
    # policies take their shard list from the table and skip name
    # filtering entirely.
    leaf_name_pattern: str | None = None
    # When set (e.g. "json"), `load_streaming_shard` loads the shard via
    # the named generic builder against a resolved `hf_hub_url` instead
    # of `load_dataset(repo_id, data_files=...)`. Required for repos
    # that ship a custom loading script: passing the repo id would
    # execute the script (defeating both the shard pick and the
    # `trust_remote_code` opt-out), while the generic builder reads the
    # raw file directly.
    load_builder: str | None = None
    # Upper bound on the in-shard offset, independent of shard size.
    #
    # The offset is applied as `.skip(n)` on a streaming dataset, which
    # decodes and discards every row it passes — `datasets` has no
    # row-group seek, and the footer read that gives us the row count
    # throws its row-group offsets away. So the per-round cost scales
    # with the offset drawn, not with the shard pick.
    #
    # Unset means "bounded only by the shard", which is correct while
    # shards are a few hundred thousand rows: a ~985 k-row shard costs
    # ~650 MB of decode on an average draw. It stops being correct for
    # multi-million-row shards — a 9.9 M-row shard averages ~12.5 GB per
    # round, roughly 19x, against a Validate phase of ~2 minutes.
    #
    # Setting it trades reachable pool for round time: the pool per shard
    # becomes `max_offset_rows` instead of the whole shard. Breadth then
    # has to come from the shard pick rotating across seeds, which is the
    # same trade a many-shard source already makes.
    max_offset_rows: int | None = None


# Built-in sources. A task bundle may serve its own policy for a source
# (see `parse_served_policies`), and that takes precedence; entries here
# are the fallback for sources no bundle covers.
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
    ("joelniklaus/Multi_Legal_Pile", "all_all"): _SourceShardPolicy(
        # The repo's NATIVE data files. Deliberately NOT the `all_all`
        # builder-script mix: the script streams additional files from
        # external repos (joelito/eurlex_resources, legal-mc4, …) that
        # can't be pinned or row-counted here, and executing it requires
        # `trust_remote_code`. The eval pool is therefore a (large)
        # subset of the miner-training distribution — acceptable: eval ⊆
        # train, and the native corpus is tens of GB.
        path_prefix="data/",
        path_suffix=(".jsonl.xz",),
        # Pinned at registration time (2026-07-21). Bump together with a
        # re-count of the table if the dataset is ever re-uploaded.
        revision="911e1d214162fd11d2c78d3f1428cbfcbe07782c",
        row_count_source="verified_table",
        min_headroom_rows=10_000,
        # `.jsonl.xz` has no footer — rows counted once offline (full
        # decompress of every shard at the pinned revision, 2026-07-21)
        # and frozen here. The table doubles as the allowlist; four
        # shards with ≤10k rows (denmark_ddsc caselaw 4 442,
        # en switzerland_lexfind 147, belgium_jurportal 2 221,
        # it switzerland_lexfind 5 642) are deliberately left out — no
        # safe offset exists above the headroom floor.
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
    if policy.max_offset_rows is not None:
        if policy.max_offset_rows <= 0:
            raise ValueError(
                f"Policy {repo_id}/{name}: max_offset_rows must be > 0 when set "
                f"(got {policy.max_offset_rows})"
            )
        if policy.max_offset_rows < policy.min_headroom_rows:
            # Not wrong — the two bound different things, one cost and
            # one stream exhaustion — but a cap below the headroom means
            # the reachable window is narrower than the safety margin,
            # which is almost always a misplaced digit.
            logger.warning(
                "Policy has max_offset_rows below min_headroom_rows; the eval "
                "read window is narrower than the reserved tail. Intended?",
                repo_id=repo_id, name=name,
                max_offset_rows=policy.max_offset_rows,
                min_headroom_rows=policy.min_headroom_rows,
            )
    if policy.row_count_source == "verified_table":
        # The table IS the shard allowlist: every listed shard must have
        # a row count that leaves at least one valid offset after the
        # headroom is reserved. Shards too small for the eval pipeline's
        # per-round consumption must be left out of the table, not
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


# ----------------------------------------------------------- served policies
#
# A task bundle may carry a `shard_policy.json` beside its `config.yaml`
# (written by `task_sync.materialize_task`, covered by `bundle_sha256`).
# It lets a dataset's shard policy be registered by publishing a task
# rather than by cutting a validator release.
#
# Consensus is preserved by a different mechanism than the built-in
# registry's: every validator fetches the SAME bundle from the same
# locked `owner_url` and checks it against the server's hash, so the
# policy is identical fleet-wide by construction. That is strictly
# stronger than "deploy the same commit", which a mixed-version fleet
# can violate.

_SERVED_POLICY_SCHEMA_VERSION = 1

# Generic HF builders a served policy may name. `load_builder` exists to
# AVOID executing a repo's own loading script, so a publisher has to be
# able to set it; restricting it to file-format builders keeps a served
# value from naming something with side effects of its own.
_ALLOWED_LOAD_BUILDERS = frozenset({"arrow", "csv", "json", "parquet", "text"})


class ServedShardPolicy(BaseModel):
    """One source's policy as it arrives in `shard_policy.json`.

    `extra="forbid"`, deliberately against this repo's usual
    `extra="ignore"`: an ignored typo here does not fail, it silently
    samples the wrong rows. The cost is that a policy using a field
    added in a later release cannot be published until the whole fleet
    carries that release — the same discipline the rest of this rollout
    already requires.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str | None = None
    path_prefix: str
    path_suffix: tuple[str, ...]
    revision: str
    row_count_source: Literal["constant", "parquet_footer", "verified_table"]
    safe_floor_rows: int | None = None
    min_headroom_rows: int = 10_000
    verified_shard_rows: dict[str, int] = Field(default_factory=dict)
    leaf_name_pattern: str | None = None
    load_builder: str | None = None
    max_offset_rows: int | None = None

    @model_validator(mode="after")
    def _validate_load_builder(self) -> ServedShardPolicy:
        if self.load_builder is not None and self.load_builder not in _ALLOWED_LOAD_BUILDERS:
            raise ValueError(
                f"load_builder {self.load_builder!r} is not one of {sorted(_ALLOWED_LOAD_BUILDERS)}"
            )
        return self

    def to_policy(self) -> _SourceShardPolicy:
        """The in-code policy object this document describes.

        `path_suffix` lands as a tuple — the field type coerces it —
        because `str.endswith` raises `TypeError` on a list and JSON
        gives a list. Without that coercion every shard listing for a
        served source would fail.
        """
        return _SourceShardPolicy(
            path_prefix=self.path_prefix,
            path_suffix=tuple(self.path_suffix),
            revision=self.revision,
            row_count_source=self.row_count_source,
            safe_floor_rows=self.safe_floor_rows,
            min_headroom_rows=self.min_headroom_rows,
            verified_shard_rows=dict(self.verified_shard_rows),
            leaf_name_pattern=self.leaf_name_pattern,
            load_builder=self.load_builder,
            max_offset_rows=self.max_offset_rows,
        )


class ServedShardPolicyDoc(BaseModel):
    """A whole `shard_policy.json` document.

    Sources are a list rather than a mapping keyed by repo id because
    the registry key is `(path, name)` and `name` may be null, which a
    JSON object key cannot express.
    """

    model_config = ConfigDict(extra="forbid")

    version: int
    sources: list[ServedShardPolicy]

    @model_validator(mode="after")
    def _validate_version_and_keys(self) -> ServedShardPolicyDoc:
        # Checked rather than ignored so a future schema cannot be
        # half-understood by an older validator.
        if self.version != _SERVED_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported shard_policy.json version {self.version}; this validator "
                f"understands version {_SERVED_POLICY_SCHEMA_VERSION}"
            )
        seen: set[tuple[str, str | None]] = set()
        for source in self.sources:
            key = (source.path, source.name)
            if key in seen:
                raise ValueError(f"duplicate policy for source {key}")
            seen.add(key)
        return self


def parse_served_policies(doc: dict[str, Any]) -> dict[tuple[str, str | None], _SourceShardPolicy]:
    """Validate a served policy document into a policy registry.

    Raises on anything malformed, and callers must NOT soften that into
    a fallback. Every validator holds the identical bundle, so a bad
    document fails identically fleet-wide: one uniformly bad round,
    which is recoverable and loud. Falling back to `_KNOWN_SOURCES` on
    some nodes and not others would split weight consensus, which is
    neither.
    """
    parsed = ServedShardPolicyDoc(**doc)
    registry: dict[tuple[str, str | None], _SourceShardPolicy] = {}
    for source in parsed.sources:
        key = (source.path, source.name)
        policy = source.to_policy()
        # Held to exactly the invariants the built-in registry is held
        # to at import time.
        _validate_policy(key, policy)
        registry[key] = policy
    return registry


# The active task's served policies. Consulted by `_policy_for` ahead of
# `_KNOWN_SOURCES`. Module-level, like the built-in registry, because
# exactly one task is active per process.
_SERVED_POLICIES: dict[tuple[str, str | None], _SourceShardPolicy] = {}

# sha256 of the document `_SERVED_POLICIES` was parsed from, "" when none
# is active. This is a CACHE KEY, not bookkeeping: `_list_shards` is
# keyed on (repo_id, name, revision) but reads the policy inside its
# body, so without a fingerprint component a task switch that changes a
# policy while leaving the revision alone would be served a stale shard
# list — silently, and only on nodes that had warmed the cache. That is
# a consensus break, so the fingerprint travels with the cache key.
_SERVED_FINGERPRINT = ""

SHARD_POLICY_FILE = "shard_policy.json"


def activate_served_policies(task_dir: Path) -> None:
    """Install, or clear, the policies served with the active task.

    Called before each seeded pick rather than wired into the task
    switch: `config.task.path` is re-derived on switch and
    `materialize_task` has already replaced the directory, so reading
    it here picks the change up with no lifecycle plumbing in the
    validator or the miner. Re-reading the same document is a stat and
    a hash, and returns early.

    Absence of the file is the normal case — a task that serves no
    policy falls through to `_KNOWN_SOURCES`.

    Raises on a present-but-invalid document. See `parse_served_policies`
    for why that must not be softened into a fallback.
    """
    global _SERVED_POLICIES, _SERVED_FINGERPRINT

    path = Path(task_dir) / SHARD_POLICY_FILE
    raw = path.read_bytes() if path.is_file() else b""
    fingerprint = hashlib.sha256(raw).hexdigest() if raw else ""
    if fingerprint == _SERVED_FINGERPRINT:
        return

    policies = parse_served_policies(json.loads(raw)) if raw else {}
    _SERVED_POLICIES = policies
    _SERVED_FINGERPRINT = fingerprint
    # Deliberately no `_list_shards.cache_clear()` here. The fingerprint
    # is part of that cache's key, so entries from a previous document
    # are unreachable rather than stale, and the bounded LRU evicts them
    # on its own. Clearing as well would be a second mechanism for the
    # same guarantee, and a weaker one: a pick already inside
    # `_list_shards` when the clear ran would write its result back
    # afterwards, under a key the new policy still reads.
    # Source names are deliberately not logged: the count and
    # fingerprint are enough to diagnose a switch, and the task config
    # is where the identities belong.
    logger.info(
        "served shard policies activated",
        task_dir=str(task_dir),
        source_count=len(policies),
        fingerprint=fingerprint[:12] or None,
    )


def _policy_for(path: str, name: str | None) -> _SourceShardPolicy:
    key = (path, name)
    # Served policies win. The bundle is the source of truth for the
    # active task, which is what lets a wrong policy be corrected by
    # publishing a task instead of cutting a release.
    if key in _SERVED_POLICIES:
        return _SERVED_POLICIES[key]
    if key not in _KNOWN_SOURCES:
        raise KeyError(
            f"No shard-pick policy registered for source ({path!r}, {name!r}). "
            f"Serve one in the task bundle's `{SHARD_POLICY_FILE}`, or add an "
            f"entry to `_KNOWN_SOURCES` in `eval_shard_pick.py`, and verify that "
            f"data_files=[shard] yields the same rows as the canonical load path "
            f"before flipping the feature flag."
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
def _list_shards(
    repo_id: str, name: str | None, revision: str, policy_fingerprint: str = ""
) -> tuple[str, ...]:
    """Return the deterministically-sorted shard list for a source.

    Cached per (repo_id, name, revision, policy_fingerprint). Sort is
    plain lex over the full `rfilename` string so any validator
    computing this against the same revision gets the same tuple.

    `policy_fingerprint` is unread in the body and exists only to keep
    the cache honest. The body resolves `_policy_for(...)` and filters
    on `path_prefix`, `path_suffix`, `leaf_name_pattern` and the
    verified table, so two different policies for one
    (repo_id, name, revision) must not share a cache entry.
    """
    policy = _policy_for(repo_id, name)
    if policy.row_count_source == "verified_table":
        # The frozen table doubles as the shard allowlist. Listing from
        # the HF API here would re-introduce the consensus hazard the
        # table exists to remove (a re-uploaded repo changing the list
        # under our feet); the pinned revision + table are the source
        # of truth.
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
        # Final-segment shape check guards against accidentally pulling
        # in unrelated files that happen to share the prefix/suffix
        # (e.g. metadata, sidecar files). The "train" / "part_" check
        # is per-source-format and intentionally narrow.
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
    """Read num_rows from the parquet footer without downloading the full file.

    The row count lives in the footer, a few kilobytes at the end of the
    file, so this is a couple of range requests. Reaching it through
    `hf_hub_download` first — which is what this did — downloaded the
    whole shard to get at them, and cached it: one full shard per
    distinct shard a validator's seed picks, which is how an observer
    accumulated 58 GB for a single source on a host where disk pressure
    had already caused an incident. A source with 25 GB shards would
    have made that 118 GB.

    `HfFileSystem` serves the same bytes over range requests and writes
    nothing to the cache. Measured against a source with eight shards of
    10-25 GB: the largest answered in 1.0 s, and all eight (118 GB) in
    4.9 s. The revision is already resolved to a SHA by the caller, so
    the count returned is identical and the pinning story is unchanged.
    """
    # Lazy import — pyarrow is already a dependency of `datasets` but
    # importing it eagerly at module top-level slows test imports.
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    try:
        # `datasets/<repo>@<sha>/<path>` is HfFileSystem's revision
        # syntax; the caller has already resolved `revision` to a SHA.
        fs_path = f"datasets/{repo_id}@{revision}/{shard_path}"
        with HfFileSystem().open(fs_path, "rb") as fh:
            return int(pq.read_metadata(fh).num_rows)
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

    shards = _list_shards(repo_id, name, revision, _SERVED_FINGERPRINT)
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
    # For constant policies we don't know the actual shard size;
    # surface `offset_bound` as `shard_rows` so older callers (notebook
    # / tests) that check "offset < shard_rows" still see the right
    # invariant. The parquet and verified-table paths know the real count.
    #
    # Derived from the UNCAPPED bound on purpose: `shard_rows` is the
    # shard's true size and is logged as such, so capping first would
    # make the log understate the shard.
    if policy.row_count_source in {"parquet_footer", "verified_table"}:
        actual_shard_rows = offset_bound + policy.min_headroom_rows
    else:
        actual_shard_rows = offset_bound

    # Cost cap, applied here rather than inside `_resolve_offset_bound`
    # for the reason above: that function answers "how far into this
    # shard is it safe to skip", which is a property of the row count,
    # while this answers "how far is it affordable to skip", which is a
    # property of the decode budget.
    if policy.max_offset_rows is not None:
        offset_bound = min(offset_bound, policy.max_offset_rows)

    # `% offset_bound` lets the mod be the safe_floor, the
    # footer-derived actual_rows - headroom, or the cost cap; either
    # way, after `.skip` the source retains at least
    # `min_headroom_rows` for the downstream pipeline.
    offset = (
        h256_int("eval_in_shard_offset", repo_id, str(name), int_seed) % offset_bound
    )

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
    # Lazy: `datasets` is already imported by the caller side, but
    # keeping this module importable without the full datasets stack
    # helps unit tests.
    from datasets import load_dataset

    if pick.load_builder:
        # Script-bypass path: the repo ships a custom loading script, so
        # `load_dataset(repo_id, ...)` would execute it (and for
        # Multi_Legal_Pile, stream entirely different files from external
        # repos). Loading the raw file through a generic builder reads
        # exactly the picked shard and needs no `trust_remote_code`.
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

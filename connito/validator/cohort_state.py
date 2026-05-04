"""Per-validator cohort state for the round-group construction scheme.

A cohort is the 8-cycle hold window during which validation Groups B and C
(and weight Group 2) are kept constant. `CohortState` is what survives a
validator restart so a process coming back up at cycle `8k+5` rejoins the
existing cohort instead of starting a fresh window.

`cohort_epoch` is `cycle_index // 8 * 8` and is therefore reconstructable
from chain state alone, but Group C is per-validator (drawn from
`validator_miner_assignment` for this hotkey) and cannot be derived from
chain state, so it must be persisted.

Persisted alongside `score_aggregator.json` under the validator's
checkpoint directory; load/save mirrors `MinerScoreAggregator.persist_atomic`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class CohortState:
    """Per-validator cohort hold window and its election outputs.

    A `CohortState` is written exactly once per cohort epoch — at the
    cohort boundary, after the previous cohort's election ballots have
    produced the new validation Groups B and C and weight Group 2.

    Fields:
      - `cohort_epoch`: `cycle_index` at which this cohort window began.
        Equals `cycle_index // 8 * 8` for any cycle inside the window.
      - `expert_group`: validator's expert group when this state was
        written. Asserted on load so a config change cannot silently
        carry stale assignment forward.
      - `validation_group_b`, `validation_group_c`, `weight_group_2`:
        held constant for the cohort.
      - `last_election_round_id`: the `round_id` whose election produced
        these groups. `None` only on first-ever cohort.
      - `highest_seen_cycle_index`: clamp against rollback (chain reorg
        or owner-API misbehavior). Refuse to advance the cohort if a
        load brings back a value lower than this.
    """

    cohort_epoch: int
    expert_group: str
    weight_group_1: tuple[int, ...] = ()         # this validator's local ballot
    weight_group_2: tuple[int, ...] = ()         # this validator's local ballot
    validation_group_a: tuple[int, ...] = ()     # chain-set top-3 (cross-validator)
    validation_group_b: tuple[int, ...] = ()     # chain-set top-N (cross-validator)
    validation_group_c: tuple[int, ...] = ()     # per-validator assignment slice
    last_election_round_id: int | None = None
    highest_seen_cycle_index: int = 0
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        payload = asdict(self)
        # Tuples round-trip as lists in JSON; normalize at load.
        for key in (
            "weight_group_1",
            "weight_group_2",
            "validation_group_a",
            "validation_group_b",
            "validation_group_c",
        ):
            payload[key] = list(getattr(self, key))
        return json.dumps(payload)

    @classmethod
    def from_json(cls, data: str) -> "CohortState":
        raw = json.loads(data)
        version = int(raw.get("schema_version", 1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CohortState schema_version={version}; "
                f"expected {SCHEMA_VERSION}"
            )
        last_rid = raw.get("last_election_round_id")
        return cls(
            cohort_epoch=int(raw["cohort_epoch"]),
            expert_group=str(raw["expert_group"]),
            weight_group_1=tuple(int(u) for u in raw.get("weight_group_1", [])),
            weight_group_2=tuple(int(u) for u in raw.get("weight_group_2", [])),
            validation_group_a=tuple(int(u) for u in raw.get("validation_group_a", [])),
            validation_group_b=tuple(int(u) for u in raw.get("validation_group_b", [])),
            validation_group_c=tuple(int(u) for u in raw.get("validation_group_c", [])),
            last_election_round_id=int(last_rid) if last_rid is not None else None,
            highest_seen_cycle_index=int(raw.get("highest_seen_cycle_index", 0)),
            schema_version=version,
        )


def load(path: str | os.PathLike, expected_expert_group: str) -> CohortState | None:
    """Load `CohortState` from disk, or `None` if the file does not exist.

    Asserts `expected_expert_group` matches the persisted value; mismatch
    means the validator's expert group changed between runs and the held
    cohort is no longer applicable. Caller decides whether to discard.
    """
    p = Path(path)
    if not p.exists():
        return None
    state = CohortState.from_json(p.read_text(encoding="utf-8"))
    if state.expert_group != expected_expert_group:
        raise ValueError(
            f"CohortState.expert_group={state.expert_group!r} does not match "
            f"current expert_group={expected_expert_group!r}; refusing to load"
        )
    return state


def persist_atomic(path: str | os.PathLike, state: CohortState) -> None:
    """Write `state.to_json()` to `path` atomically (tmp file + os.replace).

    Mirrors `MinerScoreAggregator.persist_atomic` so a crash mid-write
    leaves the previous fully-flushed snapshot intact.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_json()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(p.parent),
        prefix=f".{p.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, p)

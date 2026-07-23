"""Per-round mutation journal for the validator.

Records every score / failure / freeze-zero event that lands on the live
``Round`` to disk, so a kill before ``finalize_round_scores`` runs (SIGKILL,
OOM, segfault, eval-window timeout) does not lose the bg-eval / foreground
work for that round. On the next clean startup, ``finalize_round_scores``
can be replayed against any unfinalized journal so the aggregator on disk
ends up identical to what it would have been without the kill.

File layout: ``<checkpoint_path>/round_journal/round_<round_id>.json``.
One file per round_id, kept after finalize as an audit log; pruned by
age (``8 × cycle_length``) at the same site that prunes the aggregator.

Persistence pattern mirrors ``connito/validator/cohort_state.py``:
schema-version envelope, atomic tmp-file + ``os.fsync`` + ``os.replace``,
``load()`` validates the schema version.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

# v2 adds `uid_to_commit` (uid -> (hf_repo_id, hf_revision)) so the
# journal-recovery finalize can re-publish evaluated-commit telemetry.
# `from_json` accepts v1 files (missing map -> empty) so leftover journals
# written by an older build still recover.
SCHEMA_VERSION = 2
JOURNAL_DIR_NAME = "round_journal"
JOURNAL_FILENAME_PREFIX = "round_"
JOURNAL_FILENAME_SUFFIX = ".json"


@dataclass
class RoundJournal:
    """On-disk snapshot of one ``Round``'s mutation state.

    Written atomically on every ``mark_scored`` / ``mark_failed`` /
    ``mark_validation_failed`` and once at ``Round.freeze`` time so the
    journal exists from the moment the round goes live.

    Fields are deliberately minimal — only what
    ``evaluator.finalize_round_scores`` reads off the round so the
    startup-recovery pass can hydrate a stub round and replay finalize.
    """

    round_id: int
    uid_to_hotkey: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)
    scored_uids: tuple[int, ...] = ()
    failed_uids: tuple[int, ...] = ()
    validation_failed_uids: tuple[int, ...] = ()
    freeze_zero_uids: tuple[int, ...] = ()
    freeze_zero_hotkeys: dict[int, str] = field(default_factory=dict)
    # v2: uid -> (hf_repo_id, hf_revision) evaluated for the round. Only
    # uids whose chain checkpoint carried BOTH values are recorded.
    uid_to_commit: dict[int, tuple[str, str]] = field(default_factory=dict)
    finalized: bool = False
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        payload = asdict(self)
        # Normalize tuples to lists, int keys to strings (JSON requirement).
        payload["uid_to_hotkey"] = {str(k): v for k, v in self.uid_to_hotkey.items()}
        payload["freeze_zero_hotkeys"] = {str(k): v for k, v in self.freeze_zero_hotkeys.items()}
        payload["scores"] = {str(k): float(v) for k, v in self.scores.items()}
        payload["scored_uids"] = list(self.scored_uids)
        payload["failed_uids"] = list(self.failed_uids)
        payload["validation_failed_uids"] = list(self.validation_failed_uids)
        payload["freeze_zero_uids"] = list(self.freeze_zero_uids)
        payload["uid_to_commit"] = {
            str(k): [str(v[0]), str(v[1])] for k, v in self.uid_to_commit.items()
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, data: str) -> "RoundJournal":
        raw = json.loads(data)
        version = int(raw.get("schema_version", 1))
        # Accept every version up to the current one: v1 files simply lack
        # `uid_to_commit` (defaults to empty). Reject only FUTURE versions —
        # fields this build doesn't understand could change recovery
        # semantics silently.
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported RoundJournal schema_version={version}; "
                f"this build supports <= {SCHEMA_VERSION}"
            )
        return cls(
            round_id=int(raw["round_id"]),
            uid_to_hotkey={int(k): str(v) for k, v in raw.get("uid_to_hotkey", {}).items()},
            scores={int(k): float(v) for k, v in raw.get("scores", {}).items()},
            scored_uids=tuple(int(u) for u in raw.get("scored_uids", [])),
            failed_uids=tuple(int(u) for u in raw.get("failed_uids", [])),
            validation_failed_uids=tuple(int(u) for u in raw.get("validation_failed_uids", [])),
            freeze_zero_uids=tuple(int(u) for u in raw.get("freeze_zero_uids", [])),
            freeze_zero_hotkeys={
                int(k): str(v) for k, v in raw.get("freeze_zero_hotkeys", {}).items()
            },
            uid_to_commit={
                int(k): (str(v[0]), str(v[1]))
                for k, v in raw.get("uid_to_commit", {}).items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            },
            finalized=bool(raw.get("finalized", False)),
            schema_version=version,
        )


def commit_map_from_checkpoints(
    uid_to_chain_checkpoint: dict[int, object],
) -> dict[int, tuple[str, str]]:
    """Extract the journal's `uid_to_commit` map from a round's
    `uid_to_chain_checkpoint`. Skips uids missing either field.
    """
    out: dict[int, tuple[str, str]] = {}
    for uid, ckpt in (uid_to_chain_checkpoint or {}).items():
        repo = getattr(ckpt, "hf_repo_id", None)
        rev = getattr(ckpt, "hf_revision", None)
        if repo and rev:
            out[int(uid)] = (str(repo), str(rev))
    return out


def journal_dir(checkpoint_path: str | os.PathLike) -> Path:
    """Directory holding all per-round journal files."""
    return Path(checkpoint_path) / JOURNAL_DIR_NAME


def journal_path_for(checkpoint_path: str | os.PathLike, round_id: int) -> Path:
    """Path of the journal file for a specific round_id."""
    return journal_dir(checkpoint_path) / f"{JOURNAL_FILENAME_PREFIX}{int(round_id)}{JOURNAL_FILENAME_SUFFIX}"


def write_atomic(path: str | os.PathLike, journal: RoundJournal) -> None:
    """Write ``journal.to_json()`` to ``path`` atomically (tmp file +
    ``os.replace``). Same shape as ``cohort_state.persist_atomic`` so a
    crash mid-write leaves the prior snapshot intact.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = journal.to_json()
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


def load(path: str | os.PathLike) -> RoundJournal | None:
    """Load a ``RoundJournal`` from disk, or ``None`` if the file is absent."""
    p = Path(path)
    if not p.exists():
        return None
    return RoundJournal.from_json(p.read_text(encoding="utf-8"))


def scan(checkpoint_path: str | os.PathLike) -> list[Path]:
    """List every ``round_<rid>.json`` under the journal directory.

    Sorted by round_id ascending so startup-recovery replays older
    rounds before newer ones.
    """
    d = journal_dir(checkpoint_path)
    if not d.exists():
        return []
    matches: list[tuple[int, Path]] = []
    for entry in d.iterdir():
        name = entry.name
        if not (name.startswith(JOURNAL_FILENAME_PREFIX) and name.endswith(JOURNAL_FILENAME_SUFFIX)):
            continue
        rid_str = name[len(JOURNAL_FILENAME_PREFIX) : -len(JOURNAL_FILENAME_SUFFIX)]
        try:
            rid = int(rid_str)
        except ValueError:
            continue
        matches.append((rid, entry))
    matches.sort()
    return [p for _, p in matches]


@dataclass
class _RecoveryRound:
    """Round-shaped stub used by the startup-recovery pass.

    `evaluator.finalize_round_scores` reads a small set of fields off
    its `round_obj` argument (`round_id`, `scores`, `scored_uids`,
    `validation_failed_uids`, `freeze_zero_uids`, `freeze_zero_hotkeys`,
    `uid_to_hotkey`, `_lock`, `journal_path`, plus
    `processed_uids_snapshot()`). This dataclass exposes exactly those
    so we can drive a finalize off a leftover journal at startup
    without rebuilding the full `Round` (which carries the model
    snapshot, chain checkpoints, etc. — none of which finalize needs).
    """
    round_id: int
    scores: dict[int, float]
    scored_uids: set[int]
    failed_uids: set[int]
    validation_failed_uids: set[int]
    freeze_zero_uids: set[int]
    freeze_zero_hotkeys: dict[int, str]
    uid_to_hotkey: dict[int, str]
    journal_path: Path
    # Same shape finalize reads off a live Round: objects exposing
    # `.hf_repo_id` / `.hf_revision`. Hydrated from the journal's v2
    # `uid_to_commit` map (empty for v1 journals) so a recovered finalize
    # re-publishes evaluated-commit telemetry too.
    uid_to_chain_checkpoint: dict[int, object] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_journal(cls, journal: "RoundJournal", journal_path: str | os.PathLike) -> "_RecoveryRound":
        from types import SimpleNamespace

        return cls(
            round_id=int(journal.round_id),
            scores=dict(journal.scores),
            scored_uids=set(journal.scored_uids),
            failed_uids=set(journal.failed_uids),
            validation_failed_uids=set(journal.validation_failed_uids),
            freeze_zero_uids=set(journal.freeze_zero_uids),
            freeze_zero_hotkeys=dict(journal.freeze_zero_hotkeys),
            uid_to_hotkey=dict(journal.uid_to_hotkey),
            journal_path=Path(journal_path),
            uid_to_chain_checkpoint={
                int(uid): SimpleNamespace(hf_repo_id=repo, hf_revision=rev)
                for uid, (repo, rev) in journal.uid_to_commit.items()
            },
        )

    def processed_uids_snapshot(self) -> tuple[set[int], set[int]]:
        with self._lock:
            return set(self.scored_uids), set(self.failed_uids)


def prune_before_round(checkpoint_path: str | os.PathLike, min_round_id: int) -> int:
    """Delete every journal whose round_id is below ``min_round_id``.

    Mirrors ``MinerScoreAggregator.prune_before_round`` — same call site
    in run.py uses both with the same cutoff. Returns the count deleted.
    """
    deleted = 0
    for entry in scan(checkpoint_path):
        rid_str = entry.name[len(JOURNAL_FILENAME_PREFIX) : -len(JOURNAL_FILENAME_SUFFIX)]
        try:
            rid = int(rid_str)
        except ValueError:
            continue
        if rid < int(min_round_id):
            try:
                entry.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass
    return deleted

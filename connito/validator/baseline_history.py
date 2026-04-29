"""Ring buffer of recent rounds' baseline_loss values.

Powers the 24-round trend tile on the leaderboard frontend (per
``observability/miner-validator-leaderboard.html``). Persisted to disk in
the same checkpoint directory as ``score_aggregator.json`` so the trend
populates immediately after a validator restart instead of starting empty.

Designed to mirror ``MinerScoreAggregator``: thread-safe, atomic on-disk
persistence (tmp + ``os.replace``), and a ``from_json`` recovery path
that tolerates corrupt files by falling back to an empty buffer.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class BaselineLossHistory:
    """Bounded chronological buffer of (round_id, baseline_loss, timestamp).

    ``add`` appends; on overflow, the oldest entry is dropped. ``snapshot``
    returns a JSON-serializable list — what the API hands to the frontend.
    """

    def __init__(self, max_points: int = 24) -> None:
        self.max_points = int(max_points)
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ---------------- mutation ----------------
    def add(self, round_id: int, baseline_loss: float) -> None:
        entry = {
            "round_id": int(round_id),
            "baseline_loss": float(baseline_loss),
            "timestamp": time.time(),
        }
        with self._lock:
            # Idempotent on round_id — re-running a round (e.g. after a
            # restart mid-round) updates in place rather than duplicating.
            for i, e in enumerate(self._entries):
                if e["round_id"] == entry["round_id"]:
                    self._entries[i] = entry
                    return
            self._entries.append(entry)
            if len(self._entries) > self.max_points:
                # Drop oldest. Stable order: the trim is from the head.
                self._entries = self._entries[-self.max_points:]

    # ---------------- read ----------------
    def snapshot(self) -> list[dict[str, Any]]:
        """Shallow copy of the entries list, oldest → newest."""
        with self._lock:
            return [dict(e) for e in self._entries]

    # ---------------- persistence ----------------
    def to_json(self) -> str:
        with self._lock:
            return json.dumps({
                "schema_version": SCHEMA_VERSION,
                "max_points": self.max_points,
                "entries": list(self._entries),
            })

    @classmethod
    def from_json(cls, data: str, max_points: int | None = None) -> "BaselineLossHistory":
        raw = json.loads(data)
        n = int(max_points if max_points is not None else raw.get("max_points", 24))
        h = cls(max_points=n)
        for e in raw.get("entries", []):
            try:
                h._entries.append({
                    "round_id": int(e["round_id"]),
                    "baseline_loss": float(e["baseline_loss"]),
                    "timestamp": float(e.get("timestamp", 0.0)),
                })
            except (KeyError, ValueError, TypeError):
                # Skip malformed entries; the rest of the file may still be good.
                continue
        # Cap if the on-disk file held more than max_points (e.g. capacity
        # was reduced between runs).
        if len(h._entries) > h.max_points:
            h._entries = h._entries[-h.max_points:]
        return h

    def persist_atomic(self, path: str | os.PathLike) -> None:
        """Atomically write ``to_json()`` to ``path``. Mirrors the
        ``MinerScoreAggregator.persist_atomic`` pattern."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)

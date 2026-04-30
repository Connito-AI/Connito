from __future__ import annotations

import bisect
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

# Score-aggregator on-disk format version. v2 wraps miners in a top-level
# envelope and tags each (ts, score) with the round_id it was recorded under,
# so a restart mid-round can drop the in-flight round's partial scores.
SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MinerSeries:
    """Holds a single miner's (timestamp, score, round_id) points, kept sorted by time."""

    points: list[tuple[datetime, float, int | None]] = field(default_factory=list)
    max_points: int = 8  # default mirrors config.evaluation.score_window

    def add(self, ts: datetime, score: float, round_id: int | None = None) -> None:
        if ts.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (e.g., UTC).")
        i = bisect.bisect_left(self.points, (ts, float("-inf"), -1))
        if i < len(self.points) and self.points[i][0] == ts:
            self.points[i] = (ts, score, round_id)  # overwrite same-ts
        else:
            self.points.insert(i, (ts, score, round_id))

        # Keep only the last max_points phases per miner to compute rolling average
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]

    def slice(self, start: datetime | None, end: datetime | None) -> list[tuple[datetime, float, int | None]]:
        if start and start.tzinfo is None:
            raise ValueError("start must be timezone-aware.")
        if end and end.tzinfo is None:
            raise ValueError("end must be timezone-aware.")
        lo = 0 if start is None else bisect.bisect_left(self.points, (start, float("-inf"), -1))
        hi = len(self.points) if end is None else bisect.bisect_right(self.points, (end, float("inf"), 2**63 - 1))
        return self.points[lo:hi]

    def prune_before(self, cutoff: datetime) -> None:
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware.")
        idx = bisect.bisect_left(self.points, (cutoff, float("-inf"), -1))
        if idx > 0:
            del self.points[:idx]

    def drop_round(self, round_id: int) -> int:
        """Remove every point tagged with the given round_id. Returns how many were dropped."""
        before = len(self.points)
        self.points = [p for p in self.points if p[2] != round_id]
        return before - len(self.points)

    def clear(self) -> None:
        self.points.clear()

    def latest(self) -> float:
        return float(self.points[-1][1]) if self.points else 0.0

    def _window(
        self, start: datetime | None, end: datetime | None,
    ) -> list[tuple[datetime, float, int | None]]:
        """Return the points used by every aggregation, hard-capped to
        the last ``max_points`` entries. ``add()`` already trims storage,
        so this is mostly belt-and-suspenders for callers that pass a
        wide ``start``/``end`` (e.g. the entire run history) — we still
        only ever look at the most recent ``score_window`` records.
        """
        pts = self.slice(start, end) if (start or end) else self.points
        if len(pts) > self.max_points:
            pts = pts[-self.max_points:]
        return pts

    def sum(self, start: datetime | None = None, end: datetime | None = None) -> float:
        pts = self._window(start, end)
        return float(sum(v for _, v, _ in pts))

    def avg(self, start: datetime | None = None, end: datetime | None = None) -> float:
        pts = self._window(start, end)
        return float(sum(v for _, v, _ in pts) / len(pts)) if pts else 0.0


from connito.shared.telemetry import VALIDATOR_MINER_SCORE

@dataclass
class MinerState:
    uid: int
    hotkey: str
    series: MinerSeries = field(default_factory=MinerSeries)


class MinerScoreAggregator:
    """
    Aggregates miners' scores over time keyed by uid (int), and tracks each miner's current hotkey.
    REQUIREMENT: if a uid's hotkey changes, that uid's score history resets.
    """

    def __init__(self, max_points: int = 8):
        self._miners: dict[int, MinerState] = {}  # uid -> MinerState
        self._lock = threading.RLock()
        self._max_points = max_points

    # ---------- Recording ----------
    def add_score(
        self,
        uid: int,
        hotkey: str,
        score: float,
        ts: datetime | None = None,
        round_id: int | None = None,
    ) -> None:
        """
        Add a score point for a uid at timestamp ts (UTC if omitted).
        If the provided hotkey differs from the stored hotkey for this uid,
        the uid's score history is RESET before recording this new point.

        round_id tags the score with the lifecycle round that produced it.
        Pass None for legacy callers; restart recovery uses the tag to drop
        partial in-flight rounds.
        """
        uid = int(uid)
        if ts is None:
            ts = _utc_now()
        if ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware (UTC recommended).")

        with self._lock:
            state = self._miners.get(uid)
            if state is None:
                state = MinerState(uid=uid, hotkey=hotkey, series=MinerSeries(max_points=self._max_points))
                self._miners[uid] = state
            elif state.hotkey != hotkey:
                # Hotkey changed -> reset scores for this uid
                state.hotkey = hotkey
                state.series.clear()

            state.series.add(ts, float(score), round_id)
            # Push metric update to prometheus
            try:
                VALIDATOR_MINER_SCORE.labels(miner_uid=str(uid)).set(float(score))
            except Exception:
                pass

    def drop_round(self, round_id: int) -> int:
        """Remove all score points tagged with the given round_id across every miner.

        Used at restart to drop the partial score set from a round whose
        weight submission never landed. Returns the total number of points
        dropped.
        """
        dropped = 0
        with self._lock:
            for state in self._miners.values():
                dropped += state.series.drop_round(round_id)
        return dropped

    def set_hotkey(self, uid: int, new_hotkey: str) -> None:
        """
        Explicitly change a uid's hotkey and reset its scores.
        """
        uid = int(uid)
        with self._lock:
            state = self._miners.get(uid)
            if state is None:
                # create empty series for new uid
                self._miners[uid] = MinerState(uid=uid, hotkey=new_hotkey, series=MinerSeries(max_points=self._max_points))
            else:
                if state.hotkey != new_hotkey:
                    state.hotkey = new_hotkey
                    state.series.clear()

    # ---------- Retrieval ----------
    def get_history(
        self, uid: int, start: datetime | None = None, end: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        """Return [(ts, score), ...] for the uid; round_id is dropped here so
        callers that predate schema v2 still see the legacy 2-tuple shape."""
        uid = int(uid)
        with self._lock:
            s = self._miners.get(uid)
            if not s:
                return []
            return [(ts, v) for ts, v, _rid in s.series.slice(start, end)]

    # ---------- Aggregates ----------
    def sum_over(self, uid: int, start: datetime | None = None, end: datetime | None = None) -> float:
        uid = int(uid)
        with self._lock:
            s = self._miners.get(uid)
            return s.series.sum(start, end) if s else 0.0

    def avg_over(self, uid: int, start: datetime | None = None, end: datetime | None = None) -> float:
        uid = int(uid)
        with self._lock:
            s = self._miners.get(uid)
            return s.series.avg(start, end) if s else 0.0

    def rolling_sum(self, uid: int, window: timedelta, now: datetime | None = None) -> float:
        if now is None:
            now = _utc_now()
        start = now - window
        return self.sum_over(uid, start, now)

    def rolling_avg(self, uid: int, window: timedelta, now: datetime | None = None) -> float:
        if now is None:
            now = _utc_now()
        start = now - window
        return self.avg_over(uid, start, now)

    def ema(
        self, uid: int, alpha: float = 0.2,
        start: datetime | None = None, end: datetime | None = None,
    ) -> float:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1].")
        pts = self.get_history(uid, start, end)
        # Mirror MinerSeries._window: cap to the last score_window entries
        # so ema never folds in records older than what avg/sum see.
        if len(pts) > self._max_points:
            pts = pts[-self._max_points:]
        if not pts:
            return 0.0
        ema_val = pts[0][1]
        for _, v in pts[1:]:
            ema_val = alpha * v + (1 - alpha) * ema_val
        return float(ema_val)

    # ---------- New: UID → score map ----------
    def uid_score_pairs(
        self,
        how: Literal["latest", "sum", "avg", "ema"] = "latest",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[int, float]:
        """
        Return {uid: score} for ALL miners.
        - how="latest" (default): most recent score per uid (0.0 if no points)
        - how="sum": sum of scores in [start, end] (or all time if not provided)
        - how="avg": average of scores in [start, end] (0.0 if no points)
        - how="ema": exponential moving average of scores in [start, end]
        """
        with self._lock:
            out: dict[int, float] = {}
            for uid, state in self._miners.items():
                if how == "latest":
                    out[uid] = state.series.latest()
                elif how == "sum":
                    out[uid] = state.series.sum(start, end)
                elif how == "avg":
                    out[uid] = state.series.avg(start, end)
                elif how == "ema":
                    out[uid] = self.ema(uid, start=start, end=end)
                else:
                    raise ValueError('how must be one of: "latest", "sum", "avg", "ema"')
            return out

    def is_in_top(
        self,
        uid: int,
        cutoff: int = 3,
        how: Literal["latest", "sum", "avg", "ema"] = "latest",
        among: list[int] | set[int] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> bool:
        """
        Return True if the given uid's score is in the top 'cutoff' miners.

        among: if provided, rank only against these uids instead of all
               tracked miners. Pass the set of uids from the current
               evaluation round to scope the ranking to this round only.

        how: the metric used for ranking ("latest", "sum", "avg", "ema").
        """
        uid = int(uid)
        scores = self.uid_score_pairs(how=how, start=start, end=end)
        if uid not in scores:
            return False

        if among is not None:
            scores = {u: s for u, s in scores.items() if u in among}
            if uid not in scores:
                return False

        # Sort all scores descending
        sorted_scores = sorted(scores.values(), reverse=True)

        # If there are fewer miners than the cutoff, all are "in top"
        if len(sorted_scores) <= cutoff:
            return True

        # Find the cutoff score for the top N
        cutoff_score = sorted_scores[cutoff - 1]
        return scores[uid] >= cutoff_score

    def last_evaluated_per_uid(self) -> dict[int, datetime]:
        """Return ``{uid: most_recent_timestamp}`` for every miner with at
        least one recorded score. Used by ``Round.freeze`` to prioritize
        the validation queue: miners that were last evaluated longer ago
        come first so every miner gets refreshed.
        """
        with self._lock:
            return {
                uid: state.series.points[-1][0]
                for uid, state in self._miners.items()
                if state.series.points
            }

    # ---------- Maintenance ----------
    def prune_older_than(self, older_than: timedelta, now: datetime | None = None) -> None:
        if now is None:
            now = _utc_now()
        cutoff = now - older_than
        with self._lock:
            for state in self._miners.values():
                state.series.prune_before(cutoff)

    # ---------- Persistence ----------
    def to_json(self) -> str:
        """Serialize miners to JSON (schema v2: envelope + round_id per point)."""
        with self._lock:
            miners_payload = {
                str(uid): {
                    "hotkey": state.hotkey,
                    "points": [
                        [ts.isoformat(), v, rid] for ts, v, rid in state.series.points
                    ],
                }
                for uid, state in self._miners.items()
            }
        return json.dumps({"schema_version": SCHEMA_VERSION, "miners": miners_payload})

    @classmethod
    def from_json(cls, data: str, max_points: int = 8) -> MinerScoreAggregator:
        raw = json.loads(data)
        # v2 has a top-level envelope; v1 was a bare {uid: {...}} mapping.
        if isinstance(raw, dict) and "miners" in raw and "schema_version" in raw:
            miners = raw["miners"]
        else:
            miners = raw  # legacy v1

        agg = cls(max_points=max_points)
        with agg._lock:
            for uid_str, body in miners.items():
                uid = int(uid_str)
                hotkey = body.get("hotkey", "")
                pts = body.get("points", [])
                state = MinerState(uid=uid, hotkey=hotkey, series=MinerSeries(max_points=max_points))
                for entry in pts:
                    # v2 entries are [ts, score, round_id]; v1 are [ts, score].
                    if len(entry) == 3:
                        ts_str, v, rid = entry
                    else:
                        ts_str, v = entry
                        rid = None
                    ts = datetime.fromisoformat(ts_str)
                    state.series.add(ts, float(v), rid if rid is None else int(rid))
                agg._miners[uid] = state
        return agg

    def persist_atomic(self, path: str | os.PathLike) -> None:
        """Write to_json() to `path` atomically (tmp file + os.replace).

        Concurrent writers serialize on the aggregator's RLock. The on-disk
        file therefore never reflects a half-written state, so a crash
        between writes leaves the most recent fully-flushed snapshot intact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
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
                tmp_name = tmp.name
            os.replace(tmp_name, path)

from __future__ import annotations

import bisect
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MinerSeries:
    """Holds a single miner's (timestamp, score) points, kept sorted by time."""

    points: list[tuple[datetime, float]] = field(default_factory=list)

    def add(self, ts: datetime, score: float) -> None:
        if ts.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (e.g., UTC).")
        i = bisect.bisect_left(self.points, (ts, float("-inf")))
        if i < len(self.points) and self.points[i][0] == ts:
            self.points[i] = (ts, score)  # overwrite same-ts
        else:
            self.points.insert(i, (ts, score))
            
        # Keep only the last 4 phases (points) per miner to compute rolling average
        if len(self.points) > 4:
            self.points = self.points[-4:]

    def slice(self, start: datetime | None, end: datetime | None) -> list[tuple[datetime, float]]:
        if start and start.tzinfo is None:
            raise ValueError("start must be timezone-aware.")
        if end and end.tzinfo is None:
            raise ValueError("end must be timezone-aware.")
        lo = 0 if start is None else bisect.bisect_left(self.points, (start, float("-inf")))
        hi = len(self.points) if end is None else bisect.bisect_right(self.points, (end, float("inf")))
        return self.points[lo:hi]

    def prune_before(self, cutoff: datetime) -> None:
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware.")
        idx = bisect.bisect_left(self.points, (cutoff, float("-inf")))
        if idx > 0:
            del self.points[:idx]

    def clear(self) -> None:
        self.points.clear()

    def latest(self) -> float:
        return float(self.points[-1][1]) if self.points else 0.0

    def sum(self, start: datetime | None = None, end: datetime | None = None) -> float:
        pts = self.slice(start, end) if (start or end) else self.points
        return float(sum(v for _, v in pts))

    def avg(self, start: datetime | None = None, end: datetime | None = None) -> float:
        pts = self.slice(start, end) if (start or end) else self.points
        return float(sum(v for _, v in pts) / len(pts)) if pts else 0.0


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

    def __init__(self):
        self._miners: dict[int, MinerState] = {}  # uid -> MinerState
        self._lock = threading.RLock()

    # ---------- Recording ----------
    def add_score(self, uid: int, hotkey: str, score: float, ts: datetime | None = None) -> None:
        """
        Add a score point for a uid at timestamp ts (UTC if omitted).
        If the provided hotkey differs from the stored hotkey for this uid,
        the uid's score history is RESET before recording this new point.
        """
        uid = int(uid)
        if ts is None:
            ts = _utc_now()
        if ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware (UTC recommended).")

        with self._lock:
            state = self._miners.get(uid)
            if state is None:
                state = MinerState(uid=uid, hotkey=hotkey)
                self._miners[uid] = state
            elif state.hotkey != hotkey:
                # Hotkey changed -> reset scores for this uid
                state.hotkey = hotkey
                state.series.clear()

            state.series.add(ts, float(score))
            # Push metric update to prometheus
            try:
                VALIDATOR_MINER_SCORE.labels(miner_uid=str(uid)).set(float(score))
            except Exception:
                pass

    def set_hotkey(self, uid: int, new_hotkey: str) -> None:
        """
        Explicitly change a uid's hotkey and reset its scores.
        """
        uid = int(uid)
        with self._lock:
            state = self._miners.get(uid)
            if state is None:
                # create empty series for new uid
                self._miners[uid] = MinerState(uid=uid, hotkey=new_hotkey)
            else:
                if state.hotkey != new_hotkey:
                    state.hotkey = new_hotkey
                    state.series.clear()

    # ---------- Retrieval ----------
    def get_history(
        self, uid: int, start: datetime | None = None, end: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        uid = int(uid)
        with self._lock:
            s = self._miners.get(uid)
            if not s:
                return []
            return s.series.slice(start, end)

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
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> bool:
        """
        Return True if the given uid's score is in the top 'cutoff' miners.
        The ranking is determined using the specified 'how' metric:
        - how="latest": most recent score
        - how="sum": total score over [start, end]
        - how="avg": average score over [start, end]
        - how="ema": exponential moving average over [start, end]
        """
        uid = int(uid)
        scores = self.uid_score_pairs(how=how, start=start, end=end)
        if uid not in scores:
            return False

        # Sort all scores descending
        sorted_scores = sorted(scores.values(), reverse=True)

        # If there are fewer than 20 miners, all are "in top 20"
        if len(sorted_scores) <= cutoff:
            return True

        # Find the cutoff score for the top N
        cutoff_score = sorted_scores[cutoff - 1]
        return scores[uid] >= cutoff_score

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
        """Serialize miners to JSON (timestamps as ISO 8601 strings).
        JSON keys are always strings, so uid is serialized as str and
        converted back to int in from_json."""
        with self._lock:
            payload = {
                str(uid): {
                    "hotkey": state.hotkey,
                    "points": [(ts.isoformat(), v) for ts, v in state.series.points],
                }
                for uid, state in self._miners.items()
            }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, data: str) -> MinerScoreAggregator:
        raw = json.loads(data)
        agg = cls()
        with agg._lock:
            for uid_str, body in raw.items():
                uid = int(uid_str)
                hotkey = body.get("hotkey", "")
                pts = body.get("points", [])
                state = MinerState(uid=uid, hotkey=hotkey)
                for ts_str, v in pts:
                    ts = datetime.fromisoformat(ts_str)
                    state.series.add(ts, float(v))
                agg._miners[uid] = state
        return agg

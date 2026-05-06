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
    """Holds a single miner's (timestamp, score, round_id) points, kept sorted by time.

    Two caps:
    - ``max_points``: rolling window for ``avg`` / ``sum`` / ``ema``. This
      is the metric that drives weight submission, so changing it changes
      *what* the validator rewards.
    - ``max_history_points``: on-disk retention. Defaults to ``max_points``
      when omitted. Set higher than ``max_points`` to keep extra
      historical points for diagnostics without changing scoring.
    """

    points: list[tuple[datetime, float, int | None]] = field(default_factory=list)
    max_points: int = 8  # default mirrors config.evaluation.score_window
    # On-disk retention. ``None`` falls back to ``max_points`` so behavior
    # is unchanged unless the caller opts in to a larger history.
    max_history_points: int | None = None

    def __post_init__(self) -> None:
        if self.max_history_points is not None and self.max_history_points < self.max_points:
            raise ValueError(
                f"max_history_points ({self.max_history_points}) must be >= "
                f"max_points ({self.max_points}); the rolling window cannot exceed retention."
            )

    @property
    def _retention_cap(self) -> int:
        return self.max_history_points if self.max_history_points is not None else self.max_points

    def add(self, ts: datetime, score: float, round_id: int | None = None) -> None:
        if ts.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (e.g., UTC).")
        i = bisect.bisect_left(self.points, (ts, float("-inf"), -1))
        if i < len(self.points) and self.points[i][0] == ts:
            self.points[i] = (ts, score, round_id)  # overwrite same-ts
        else:
            self.points.insert(i, (ts, score, round_id))

        # Trim to on-disk retention. The rolling-avg cap (max_points) is
        # applied later inside _window/ema, not here, so increasing
        # max_history_points beyond max_points keeps extra points around
        # without changing the metric `submit_weights` consumes.
        cap = self._retention_cap
        if len(self.points) > cap:
            self.points = self.points[-cap:]

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

    def prune_before_round(self, min_round_id: int) -> int:
        """Drop tagged points whose round_id is below ``min_round_id``.

        Untagged points (legacy schema v1) are left alone — their age
        cannot be derived from round_id and the on-disk retention cap
        already bounds them.
        """
        before = len(self.points)
        self.points = [
            p for p in self.points
            if p[2] is None or p[2] >= min_round_id
        ]
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

    def cycle_means(
        self,
        round_id_to_cycle_index: dict[int, int],
        cycles_in_window: list[int],
    ) -> tuple[float, float]:
        """Mean and min per-cycle score over `cycles_in_window`, zero-filling missing cycles.

        Used by the round-group construction scheme's election ballots
        (spec items 15, 18, 20). For each cycle in the window:
          - if this miner has one or more points tagged with a `round_id`
            that maps to that cycle, the cycle's score is the mean of those
            points (a miner can be evaluated more than once per cycle —
            foreground + background, or multiple validators in tests);
          - if the miner has no points in that cycle, the cycle's score is
            0.0 (sentinel "did not qualify" per spec item 18).
        Returns `(mean, min)` across the cycles. Tie-breaking in the
        ballot uses the min — penalizes a single great cycle among
        bad ones — per spec item 20.
        """
        if not cycles_in_window:
            return 0.0, 0.0
        per_cycle_sums: dict[int, float] = {c: 0.0 for c in cycles_in_window}
        per_cycle_counts: dict[int, int] = {c: 0 for c in cycles_in_window}
        for _ts, score, rid in self.points:
            if rid is None:
                continue
            cycle = round_id_to_cycle_index.get(rid)
            if cycle is None or cycle not in per_cycle_sums:
                continue
            per_cycle_sums[cycle] += float(score)
            per_cycle_counts[cycle] += 1
        per_cycle_means: list[float] = []
        for c in cycles_in_window:
            count = per_cycle_counts[c]
            per_cycle_means.append(
                per_cycle_sums[c] / count if count > 0 else 0.0
            )
        mean = sum(per_cycle_means) / len(per_cycle_means)
        return float(mean), float(min(per_cycle_means))


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

    def __init__(self, max_points: int = 8, max_history_points: int | None = None):
        self._miners: dict[int, MinerState] = {}  # uid -> MinerState
        self._lock = threading.RLock()
        self._max_points = max_points
        # On-disk retention. None -> match max_points (no behavior change).
        # Validated by MinerSeries.__post_init__ when each series is built.
        self._max_history_points = max_history_points

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
                state = MinerState(
                    uid=uid,
                    hotkey=hotkey,
                    series=MinerSeries(
                        max_points=self._max_points,
                        max_history_points=self._max_history_points,
                    ),
                )
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
                self._miners[uid] = MinerState(
                    uid=uid,
                    hotkey=new_hotkey,
                    series=MinerSeries(
                        max_points=self._max_points,
                        max_history_points=self._max_history_points,
                    ),
                )
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

    def scores_over_window(
        self,
        uids: list[int] | set[int],
        round_id_to_cycle_index: dict[int, int],
        cycles_in_window: list[int],
    ) -> dict[int, tuple[float, float]]:
        """Return `{uid: (cycle_mean, cycle_min)}` for the given uids.

        Used by the round-group election ballot to rank validation Groups
        A∪B and B∪C over the previous cohort's 8-cycle window.
        Missing UIDs (never scored) get `(0.0, 0.0)` so the ballot
        still ranks them deterministically against scored UIDs.
        """
        out: dict[int, tuple[float, float]] = {}
        with self._lock:
            for uid in uids:
                uid_int = int(uid)
                state = self._miners.get(uid_int)
                if state is None:
                    out[uid_int] = (0.0, 0.0)
                    continue
                out[uid_int] = state.series.cycle_means(
                    round_id_to_cycle_index, cycles_in_window
                )
        return out

    def all_round_ids(self) -> set[int]:
        """Every distinct `round_id` tag across every miner's point list.

        Used by the round-group election ballot to build a
        `round_id -> cycle_index` lookup over the previous cohort's
        history without scanning each miner separately.
        """
        out: set[int] = set()
        with self._lock:
            for state in self._miners.values():
                for _ts, _v, rid in state.series.points:
                    if rid is not None:
                        out.add(int(rid))
        return out

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

    def prune_before_round(self, min_round_id: int) -> int:
        """Drop history points tagged with ``round_id < min_round_id`` across all miners.

        Used to enforce an "8 cycles of history" cap based on round_id —
        the caller passes ``current_round_id - 8 * cycle_length``.
        Returns the total number of points dropped.
        """
        dropped = 0
        with self._lock:
            for state in self._miners.values():
                dropped += state.series.prune_before_round(min_round_id)
        return dropped

    def record_count(self, uid: int) -> int:
        """Number of recorded score points for ``uid`` (0 if unknown)."""
        uid = int(uid)
        with self._lock:
            state = self._miners.get(uid)
            return len(state.series.points) if state else 0

    def latest_round_id(self, uid: int) -> int | None:
        """Highest ``round_id`` recorded for ``uid``, or ``None`` if the uid
        has no points or every point predates the round_id tag (legacy v1)."""
        uid = int(uid)
        with self._lock:
            state = self._miners.get(uid)
            if state is None:
                return None
            best: int | None = None
            for _ts, _v, rid in state.series.points:
                if rid is None:
                    continue
                if best is None or rid > best:
                    best = rid
            return best

    def has_round_ids(self, uid: int, round_ids: list[int] | tuple[int, ...]) -> bool:
        """True iff ``uid`` has at least one recorded score for every
        ``round_id`` in ``round_ids``."""
        uid = int(uid)
        if not round_ids:
            return True
        targets = {int(r) for r in round_ids}
        with self._lock:
            state = self._miners.get(uid)
            if state is None:
                return False
            seen: set[int] = set()
            for _ts, _v, rid in state.series.points:
                if rid is None:
                    continue
                if rid in targets:
                    seen.add(int(rid))
                    if seen == targets:
                        return True
            return seen == targets

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
    def from_json(
        cls,
        data: str,
        max_points: int = 8,
        max_history_points: int | None = None,
    ) -> MinerScoreAggregator:
        raw = json.loads(data)
        # v2 has a top-level envelope; v1 was a bare {uid: {...}} mapping.
        if isinstance(raw, dict) and "miners" in raw and "schema_version" in raw:
            miners = raw["miners"]
        else:
            miners = raw  # legacy v1

        agg = cls(max_points=max_points, max_history_points=max_history_points)
        with agg._lock:
            for uid_str, body in miners.items():
                uid = int(uid_str)
                hotkey = body.get("hotkey", "")
                pts = body.get("points", [])
                state = MinerState(
                    uid=uid,
                    hotkey=hotkey,
                    series=MinerSeries(
                        max_points=max_points,
                        max_history_points=max_history_points,
                    ),
                )
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

"""Which retained baseline shard the validator currently runs.

The validator's model is a frozen backbone plus one expert shard
(`shared.model.freeze_parameters` leaves nothing else trainable), so this
pointer — not a checkpoint — is its persisted model state. `globalver_*` used
to record the same fact as a multi-GB copy written at every Merge; the shard
already exists under `baseline/`.

Two shards are retained at any time (the adopted one, and the one published
since), so something has to say which is current. This is it.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from connito.shared.app_logging import structlog
from connito.shared.checkpoints import ModelCheckpoint, select_best_checkpoint
from connito.shared.helper import (
    expert_group_shard_name,
    get_model_hash,
    load_state_dict_from_path,
)

logger = structlog.get_logger(__name__)

FILENAME = "adopted_baseline.json"


@dataclass(frozen=True)
class AdoptedBaseline:
    path: Path  # the immutable round_<id> name — never a mutable alias
    round_id: int
    global_ver: int  # global_opt_step at the adopting Merge; keeps boot precedence
    model_hash: str
    revision: str | None = None

    def as_checkpoint(self) -> ModelCheckpoint:
        """Shaped like an on-disk checkpoint so boot can rank it against the
        downloaded fleet cache with the existing `global_ver` ordering."""
        return ModelCheckpoint(
            global_ver=self.global_ver, model_hash=self.model_hash,
            path=self.path, role="validator", place="local",
        )


def pointer_path(config) -> Path | None:
    root = getattr(config.ckpt, "checkpoint_path", None)
    return Path(root) / FILENAME if root else None


def baseline_dir(config) -> Path:
    """Where `publish_round_baseline` retains shards (`distribute._retain_baseline`)."""
    return Path(config.ckpt.miner_submission_path).parent / "baseline"


def load(config) -> AdoptedBaseline | None:
    """None when absent or unreadable — boot then falls back, never crashes."""
    p = pointer_path(config)
    if p is None or not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        return AdoptedBaseline(
            path=Path(raw["path"]), round_id=int(raw["round_id"]),
            global_ver=int(raw["global_ver"]), model_hash=str(raw["model_hash"]),
            revision=raw.get("revision"),
        )
    except Exception as e:
        logger.warning("adopted_baseline: unreadable, ignoring", path=str(p), error=str(e))
        return None


def persist(config, ab: AdoptedBaseline) -> None:
    """tmp + os.replace, so a crash mid-write leaves the previous pointer."""
    p = pointer_path(config)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({**asdict(ab), "path": str(ab.path)}))
    os.replace(tmp, p)


def migrate_from_legacy_globalver(config) -> AdoptedBaseline | None:
    """One boot per validator: derive the pointer from the newest `globalver_*`.

    Links its shard into `baseline/` under the version as round id, so normal
    retention applies from then on. The legacy dir is deliberately left in
    place — a rollback to the previous image must still find it.
    """
    legacy = select_best_checkpoint(primary_dir=Path(config.ckpt.checkpoint_path))
    if legacy is None or legacy.path is None or legacy.global_ver is None:
        return None
    src = Path(legacy.path) / expert_group_shard_name(config.task.exp.group_id)
    if not src.is_file():
        return None
    dest = baseline_dir(config) / f"round_{int(legacy.global_ver)}{src.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
    ab = AdoptedBaseline(
        path=dest, round_id=int(legacy.global_ver), global_ver=int(legacy.global_ver),
        model_hash=get_model_hash(load_state_dict_from_path(str(dest)), hex=True),
    )
    persist(config, ab)
    logger.info(
        "adopted_baseline: migrated from legacy checkpoint",
        legacy=str(legacy.path), path=str(dest), global_ver=ab.global_ver,
    )
    return ab

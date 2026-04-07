from __future__ import annotations

import json
from pathlib import Path
from connito.shared.app_logging import configure_logging, structlog

configure_logging()
logger = structlog.get_logger(__name__)

def _read_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except Exception:
        return []
    return []


def _write_json(path: Path, peer_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp_{path.name}")
    tmp_path.write_text(json.dumps(peer_ids, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def get_init_peer_ids(path: Path) -> list[str]:
    return _read_json(path)


def add_init_peer_id(path: Path, peer_id: str) -> list[str]:
    peer_ids = _read_json(path)
    if peer_id not in peer_ids:
        peer_ids.append(peer_id)
        _write_json(path, peer_ids)
    logger.info("added peer_id to init_peer_ids", peer_id=peer_id, all_peer_ids=peer_ids)
    return peer_ids


def remove_init_peer_id(path: Path, peer_id: str) -> list[str]:
    peer_ids = _read_json(path)
    if peer_id in peer_ids:
        peer_ids = [p for p in peer_ids if p != peer_id]
        _write_json(path, peer_ids)
    return peer_ids

#!/usr/bin/env python3
import os
import time
from pathlib import Path

import requests
from requests import Response
from bittensor import Keypair

from connito.shared.app_logging import structlog
from connito.shared.schema import (
    SignedDownloadRequestMessage,
    construct_block_message,
    sign_message,
)

logger = structlog.get_logger(__name__)

CHUNK = 1024 * 1024  # 1 MiB


def human(n):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def download_model(
    url: str,
    my_hotkey: Keypair,
    target_hotkey_ss58: str,
    block: int,
    token: str,
    out_dir: str | Path,
    expert_group_id: int | str | None = None,
    resume: bool = False,
    timeout: int = 30,
):
    out_path = Path(out_dir)
    tmp_path = out_path.with_name(f".tmp_{out_path.name}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    mode = "wb"
    start_at = 0

    if not resume and tmp_path.exists():
        tmp_path.unlink()

    if resume:
        if tmp_path.exists():
            start_at = tmp_path.stat().st_size
        elif out_path.exists():
            out_path.replace(tmp_path)
            start_at = tmp_path.stat().st_size

        if start_at > 0:
            headers["Range"] = f"bytes={start_at}-"
            mode = "ab"

    data = SignedDownloadRequestMessage(
        target_hotkey_ss58=target_hotkey_ss58,
        origin_hotkey_ss58=my_hotkey.ss58_address,
        expert_group_id=expert_group_id,
        origin_block=block,
        signature=sign_message(my_hotkey, construct_block_message(target_hotkey_ss58, block=block)),
    ).to_dict()

    def _start_request() -> Response:
        return requests.get(url, headers=headers, stream=True, timeout=timeout, data=data)

    r = _start_request()
    try:
        logger.info("HTTP response received", status_code=r.status_code)

        if r.status_code in (401, 403):
            raise RuntimeError(f"Auth failed (HTTP {r.status_code}). Check your token.")
        if r.status_code == 416:
            logger.info("Nothing to resume; file already complete.")
            if tmp_path.exists():
                os.replace(tmp_path, out_path)
            return
        if resume and r.status_code not in (200, 206):
            logger.info(f"Server did not honor range request (HTTP {r.status_code}). Restarting full download.")
            headers.pop("Range", None)
            mode = "wb"
            start_at = 0
            r.close()
            if tmp_path.exists():
                tmp_path.unlink()
            r = _start_request()
            logger.info("HTTP response received", status_code=r.status_code)

        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total = int(total) + start_at if total is not None else None

        downloaded = start_at
        t0 = time.time()
        last_print = t0

        with open(tmp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 0.5:
                    if total:
                        bar = f"{human(downloaded)} / {human(total)} ({downloaded / total * 100:5.1f}%)"
                    else:
                        bar = f"{human(downloaded)}"
                    rate = (downloaded - start_at) / max(1e-6, (now - t0))
                    logger.info(f"\rDownloading: {bar} @ {human(rate)}/s", end="", flush=True)
                    last_print = now

        elapsed = max(1e-6, time.time() - t0)
        rate = (downloaded - start_at) / elapsed
        if total:
            bar = f"{human(downloaded)} / {human(total)} (100.0%)"
        else:
            bar = f"{human(downloaded)}"
        os.replace(tmp_path, out_path)
        logger.info(f"\rDone:       {bar} in {elapsed:.1f}s @ {human(rate)}/s", final_path=str(out_path))
    finally:
        r.close()



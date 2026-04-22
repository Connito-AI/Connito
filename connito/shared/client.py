#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from requests import Response
from requests.exceptions import (
    ConnectionError as ReqConnectionError,
)
from requests.exceptions import (
    RequestException,
    Timeout,
)
from bittensor import Keypair

from connito.shared.app_logging import structlog
from connito.shared.schema import (
    SignedModelSubmitMessage,
    construct_block_message,
    construct_model_message,
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


_CHUNK = 1024 * 1024  # 1 MiB


def _sha256_file(path: str, chunk_size: int = _CHUNK) -> str:
    """Stream the file to compute SHA256 without loading into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def submit_model(
    url: str,
    token: str,
    model_path: str,
    my_hotkey: Keypair,
    target_hotkey_ss58: str,
    block: int,
    timeout_s: int = 300,
    retries: int = 3,
    backoff: float = 1.8,
    expert_groups: list[int | str] | None = None,
    extra_form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Upload a model checkpoint with retries and robust error handling.

    Returns parsed JSON on success. Raises RuntimeError with context on failure.
    """
    logger.info(
        "Starting model submission",
        url=url,
        model_path=model_path,
        target_hotkey=target_hotkey_ss58,
        block=block,
        retries=retries,
    )

    # --- preflight checks ---
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")

    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {model_path}")

    logger.info("Model file validated", file_size=file_size, human_size=human(file_size))

    model_byte = construct_model_message(model_path=model_path, expert_groups=expert_groups)
    block_byte = construct_block_message(target_hotkey_ss58, block=block)
    
    data = SignedModelSubmitMessage(
        target_hotkey_ss58=target_hotkey_ss58,
        origin_hotkey_ss58=my_hotkey.ss58_address,
        origin_block=block,
        model_hex=model_byte.hex(),
        block_hex=block_byte.hex(),
        signature=sign_message(
            my_hotkey,
            model_byte + block_byte,
        ),
    ).to_dict()

    logger.info("Constructed signed submission message", data=data, model_hash=model_byte.hex()[:6])

    if extra_form:
        # stringify non-bytes for safety in form data
        for k, v in extra_form.items():
            data[k] = v if isinstance(v, str | bytes) else str(v)

    # --- retry loop for transient failures ---
    attempt = 0
    last_exc: Exception | None = None

    while attempt <= retries:
        try:
            with open(model_path, "rb") as fh:
                files = {"file": (os.path.basename(model_path), fh)}
                logger.info("Uploading file to server", attempt=attempt + 1, retries_left=retries - attempt)
                resp: Response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    files=files,
                    data=data,
                    timeout=timeout_s,
                )
            logger.info("HTTP response received", status_code=resp.status_code, attempt=attempt + 1)

            # Raise on non-2xx
            try:
                resp.raise_for_status()
            except requests.HTTPError as http_err:
                # Try to extract JSON error payload for context
                err_body = None
                try:
                    err_body = resp.json()
                except ValueError:
                    err_body = resp.text[:1000]  # truncate long HTML
                # Some 4xx are not retryable (auth, validation)
                non_retryable = {400, 401, 403, 404, 405, 409, 422}
                detail = f"HTTP {resp.status_code}: {err_body}"
                if resp.status_code in non_retryable or attempt == retries:
                    logger.error(
                        "Submission failed (non-retryable or final attempt)",
                        status_code=resp.status_code,
                        error_body=err_body,
                        attempt=attempt + 1,
                    )
                    raise RuntimeError(f"Upload failed: {detail}") from http_err
                else:
                    last_exc = RuntimeError(detail)
                    logger.warning(
                        "HTTP error, will retry",
                        status_code=resp.status_code,
                        error_body=err_body,
                        attempt=attempt + 1,
                    )
                    # fall through to retry
                    raise

            # Parse success JSON (server should return metadata)
            try:
                result = resp.json()
                logger.info("Submission successful", response=result)
                return result
            except ValueError:
                # Not JSON; return minimal info
                logger.info("Submission successful (non-JSON response)", status_code=resp.status_code)
                return {"status": "ok", "http_status": resp.status_code, "text": resp.text}

        except (Timeout, ReqConnectionError) as net_err:
            # Retry timeouts / connection errors unless we exhausted attempts
            last_exc = net_err
            logger.warning("Network error during submission, will retry", error=str(net_err), attempt=attempt + 1)
        except RequestException as req_err:
            # Generic requests error: retry unless final attempt
            last_exc = req_err
            logger.warning("Request error during submission, will retry", error=str(req_err), attempt=attempt + 1)
        except Exception as e:
            # File I/O during open/read already handled above; treat others as fatal
            logger.error("Unexpected error during upload", error=str(e))
            raise RuntimeError(f"Unexpected error during upload: {e}") from e

        # If we got here, we plan to retry
        attempt += 1
        if attempt <= retries:
            sleep_s = backoff**attempt
            logger.info("Retrying after backoff", sleep_seconds=sleep_s, attempt=attempt + 1)
            time.sleep(sleep_s)

    # Exhausted retries
    logger.error("Submission failed after all retries exhausted", total_attempts=retries + 1, last_error=str(last_exc))
    raise RuntimeError(f"Upload failed after {retries + 1} attempts: {last_exc}")



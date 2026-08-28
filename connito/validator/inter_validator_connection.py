# --- Authorizer --- 
import os
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Set
import threading

import bittensor as bt
from hivemind.utils.auth import AuthorizedRequestBase, AuthorizedResponseBase
from hivemind.utils.crypto import RSAPublicKey
from hivemind.utils.timed_storage import TimedStorage, get_dht_time
from connito.shared.cycle import get_init_peer_id, get_validator_whitelist_from_api
from connito.shared.app_logging import structlog
from connito.shared.config import ValidatorConfig
from connito.shared.schema import sign_message, verify_message
import traceback
from bittensor.core.async_subtensor import AsyncSubtensor
import asyncio
import multiprocessing as mp
import queue

logger = structlog.get_logger(__name__)


@dataclass
class NonceCache:
    max_bytes: int = 100 * 1024 * 1024

    def __post_init__(self) -> None:
        self._nonces: OrderedDict[bytes, None] = OrderedDict()
        self._bytes: int = 0
        self._lock = threading.Lock()

    def _nonce_size(self, nonce: bytes) -> int:
        return len(nonce)

    def _evict_oldest(self) -> None:
        while self._bytes > self.max_bytes and self._nonces:
            old_nonce, _ = self._nonces.popitem(last=False)
            self._bytes -= self._nonce_size(old_nonce)

    def contains(self, nonce: bytes) -> bool:
        with self._lock:
            return nonce in self._nonces

    def add(self, nonce: bytes) -> None:
        with self._lock:
            if nonce in self._nonces:
                return
            self._nonces[nonce] = None
            self._bytes += self._nonce_size(nonce)
            self._evict_oldest()

def get_init_peer_ids(config: ValidatorConfig):
    init_peer = get_init_peer_id(config)
    if isinstance(init_peer, (list, tuple, set)):
        peers = [str(peer) for peer in init_peer if peer]
    elif init_peer:
        peers = [str(init_peer)]
    else:
        peers = []

    return peers


# ---------------------------
# Hotkey authorizer
# ---------------------------
class AllowedHotkeyService:
    def __init__(self, config, allowed_hotkey_mp, refresh_every_s: float = 60 * 5):
        self.config = config
        self.refresh_every_s = refresh_every_s  # 0 => only refresh when forced

        # process/queue state
        self.job_queue = mp.Queue()
        self._process: mp.Process | None = None
        self._stop_flag = mp.Event()
        self._refreshed_once = mp.Event()
        self.allowed_hotkeys = allowed_hotkey_mp
        self._last_refresh_ts = mp.Value("d", 0.0)
        self._last_refresh_lock = mp.Lock()

    def refresh_allowed_hotkeys(self):

        allowed = get_validator_whitelist_from_api(self.config)
        if not allowed:
            logger.warning("No allowed hotkeys returned from whitelist API")
        else:
            logger.debug("Refreshed allowed hotkeys", count=len(allowed), allowed=list(allowed))

        self.allowed_hotkeys[:] = list(allowed)
        with self._last_refresh_lock:
            self._last_refresh_ts.value = time.time()
        return allowed

    # ----- thread worker -----
    def _worker_loop(self):
        try:
            self.refresh_allowed_hotkeys()
            self._refreshed_once.set()
            logger.debug("Initial hotkey refresh done", count=len(self.allowed_hotkeys))
        except Exception:
            logger.exception("initial refresh_allowed_hotkeys failed")

        while not self._stop_flag.is_set():
            try:
                try:
                    self.job_queue.get()
                except queue.Empty:
                    continue
                self.refresh_allowed_hotkeys()
                self._refreshed_once.set()
            except Exception:
                logger.exception("_thread_loop error")

    # ----- public API -----
    def start_refresh_thread(self):
        """
        Start background refresher thread. Safe to call from sync code.
        """
        self._stop_flag.clear()
        self._process = mp.Process(target=self._worker_loop, daemon=True)
        self._process.start()
    
    def stop_refresh_thread(self, timeout: float | None = None) -> bool:
        """
        Request stop and block until thread exits (or timeout).
        Returns True if stopped cleanly.
        """
        if not (self._process and self._process.is_alive()):
            return True
        
        self._stop_flag.set()
        self.job_queue.put("stop")
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()
            return False
        return True
    
    def refresh(self, timeout: float | None = None) -> bool:
        """
        Force a refresh and optionally block until it completes.
        Returns True if refresh succeeded, False on timeout or refresh failure.
        """
        if self.refresh_every_s > 0:
            with self._last_refresh_lock:
                last_ts = self._last_refresh_ts.value
            if (time.time() - last_ts) < self.refresh_every_s:
                return True

        self._refreshed_once.clear()
        self.job_queue.put("refresh")
        if timeout is None:
            self._refreshed_once.wait()
        else:
            if not self._refreshed_once.wait(timeout=timeout):
                return False

        return True
    
class HotkeyAuthorizer:
    """
    DHT request/response authorizer that only accepts messages signed by allowed hotkeys (SS58).

    You pass this into: hivemind.DHT(..., authorizer=HotkeyAuthorizer(...))
    """

    def __init__(
        self,
        my_hotkey: bt.Wallet.hotkey,
        subtensor: bt.AsyncSubtensor,
        config,
        max_time_skew_s: float = 30,
    ):
        
        allowed_hotkeys = mp.Manager().list()
        hotkey_service = None
        hotkey_service = AllowedHotkeyService(config, allowed_hotkeys)
        hotkey_service.start_refresh_thread()

        self.my_hotkey: bt.Keypair = my_hotkey
        self.max_time_skew_s: float = max_time_skew_s
        self._seen_nonces: Optional[NonceCache] = None
        self.subtensor: bt.AsyncSubtensor = subtensor
        self.config = config
        self._max_time_diff = timedelta(minutes=1)
        self.refresh_every_s = 60 * 30
        self.hotkey_service = hotkey_service
        self.allowed_hotkeys = allowed_hotkeys

    def __post_init__(self):
        if self._seen_nonces is None:
            self._seen_nonces = NonceCache()

    def get_allowed_hotkeys(self):
        self.hotkey_service.refresh()
        return list(self.allowed_hotkeys)
    
    @property
    def my_hotkey_ss58(self) -> str:
        return self.my_hotkey.ss58_address

    # ---- Core API ----
    async def sign_request(self, request: AuthorizedRequestBase, service_public_key: Optional[RSAPublicKey]) -> None:
        self.__post_init__()
        logger.debug("sign request - start", request.auth.service_public_key)
        auth = request.auth

        auth.service_public_key = self.my_hotkey_ss58.encode("utf-8")
        auth.time = get_dht_time()
        auth.nonce = secrets.token_bytes(8)

        auth.signature = b""
        auth.signature = sign_message(self.my_hotkey, request.SerializeToString()).encode("utf-8")
        logger.debug("sign request - complete", request.auth.nonce.hex())

    async def validate_request(self, request: AuthorizedRequestBase) -> bool:
        self.__post_init__()
        logger.debug("validate_request - start", request.auth.nonce.hex(), request.auth.service_public_key)

        auth = request.auth
        try:
            signer_ss58 = auth.service_public_key.decode("utf-8")
            sig_b64url = auth.signature.decode("utf-8")
        except Exception:
            logger.info("Request auth decode failed")
            return False

        allowed_hotkeys = self.get_allowed_hotkeys()
        if signer_ss58 not in allowed_hotkeys:
            logger.info("Request from unauthorized hotkey", signer_ss58, allowed_hotkeys)
            return False

        signature = auth.signature
        auth.signature = b""
        if not verify_message(signer_ss58, request.SerializeToString(), sig_b64url):
            logger.info("Request has invalid signature")
            auth.signature = signature
            return False

        auth.signature = signature

        current_time = get_dht_time()
        if abs(float(auth.time) - current_time) > self._max_time_diff.total_seconds():
            logger.info("Clocks are not synchronized or a previous request is replayed again")
            return False

        if self._seen_nonces.contains(auth.nonce):
            logger.info("Previous request is replayed again")
            return False

        self._seen_nonces.add(auth.nonce)

        logger.debug("validate_request - complete", request.auth.nonce.hex(), request.auth.service_public_key)
        return True

    async def sign_response(self, response: AuthorizedResponseBase, request: AuthorizedRequestBase) -> None:
        self.__post_init__()
        logger.debug("sign response - start", request.auth.nonce.hex(), request.auth.service_public_key)
        auth = response.auth
        auth.nonce = request.auth.nonce
        auth.signature = b""
        auth.signature = sign_message(self.my_hotkey, response.SerializeToString()).encode("utf-8")
        logger.debug("sign response - complete")

    async def validate_response(self, response: AuthorizedResponseBase, request: AuthorizedRequestBase) -> bool:
        self.__post_init__()
        logger.debug("validate_response - start", request.auth.nonce.hex(), request.auth.service_public_key)
        auth = response.auth
        if auth.nonce != request.auth.nonce:
            logger.info("Response is generated for another request")
            return False

        try:
            sig_b64url = auth.signature.decode("utf-8", errors="ignore")
        except Exception:
            logger.info("Response auth decode failed")
            return False

        signature = auth.signature
        auth.signature = b""
        msg = response.SerializeToString()
        auth.signature = signature

        return True
        return False

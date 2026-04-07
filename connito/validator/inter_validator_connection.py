# --- Authorizer --- 
import fnmatch
import os
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Set
import threading

import bittensor as bt
import hivemind
import torch
import torch.nn as nn
from hivemind.averaging import DecentralizedAverager
from hivemind.utils.auth import AuthorizedRequestBase, AuthorizedResponseBase
from hivemind.utils.crypto import RSAPublicKey
from hivemind.utils.timed_storage import TimedStorage, get_dht_time
from connito.shared.cycle import get_init_peer_id, get_validator_whitelist_from_api
from connito.shared.app_logging import structlog
from connito.shared.config import ValidatorConfig
from connito.shared.expert_manager import get_layer_expert_id
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

def connect_with_peers(config, wallet, subtensor: bt.Subtensor, max_retry: int = 15):
    
    initial_peer_ids: list[str] = get_init_peer_ids(config)

    authorizer = HotkeyAuthorizer(
        my_hotkey=wallet.hotkey,
        max_time_skew_s=30.0,
        subtensor=subtensor,
        config = config,
    )

    logger.info("DHT bootstrap starting", peer_count=len(initial_peer_ids), initial_peers=initial_peer_ids)
    last_err: Exception | None = None
    for attempt in range(1, max_retry):
        try:
            dht = hivemind.DHT(
                start=True,
                client_mode = False,
                initial_peers=initial_peer_ids,
                authorizer=authorizer,
                bootstrap_timeout=120,
                wait_timeout=30,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/{config.dht.port}", f"/ip4/0.0.0.0/udp/{config.dht.port}/quic"],
            )
            break
        except RuntimeError as e:
            last_err = e
            logger.exception(
                "DHT bootstrap failed",
                initial_peers=initial_peer_ids,
                attempt=attempt,
            )
        except Exception as e:
            last_err = e
            logger.error(e)

        if attempt < max_retry:
            logger.info("retry soon", initial_peers=initial_peer_ids)
            time.sleep(min(5 * attempt, 10))
    else:
        raise last_err if last_err is not None else RuntimeError("DHT bootstrap failed after 5 attempts")

    visible_addrs = [str(addr) for addr in dht.get_visible_maddrs()]
    if not visible_addrs:
        logger.warning("No visible peers after DHT bootstrap")
    else:
        logger.info("DHT bootstrap complete", visible_peers=visible_addrs)
    return dht

# --- expert group selection helpers ---
def names_for_expert(
    model: nn.Module, eid, expert_name_fmt: str, include_buffers: bool
) -> list[tuple[str, torch.Tensor]]:
    """Collect all tensors whose names start with the expert module prefix."""
    prefix = expert_name_fmt.format(eid=eid)
    out = []
    for name, tensor in iter_named_params(model):
        if name.startswith(prefix + ".") or name == prefix:
            out.append((name, tensor))
    return out


def iter_named_params(model: nn.Module):
    """
    Yield (name, parameter) for model parameters without allocating gradients.
    """
    for n, p in model.named_parameters():
        yield n, p


def iter_named_grads(model: nn.Module):
    """
    Yield (name, grad_tensor) for all model parameters that have gradients.
    """
    for n, p in model.named_parameters():
        yield n, p.grad


def name_selected(name, include_globs, exclude_globs):
    inc_ok = (not include_globs) or any(fnmatch.fnmatch(name, pat) for pat in include_globs)
    exc_ok = not any(fnmatch.fnmatch(name, pat) for pat in exclude_globs)
    return inc_ok and exc_ok


def select_tensors(model, include_globs=(), exclude_globs=()):
    # deterministic order across peers: sort by name!
    chosen = []
    for name, tensor in sorted(iter_named_params(model), key=lambda kv: kv[0]):
        if name_selected(name, include_globs, exclude_globs):
            chosen.append(tensor)
    return chosen


# --- packaging gradient buff ---
def build_buff_from_params(param_dict, buffer_dtype: torch.dtype = torch.float16):
    param_names = list(param_dict.keys())
    numels = [p.numel() for p in param_dict.values()]
    offsets = [0]
    for n in numels[:-1]:
        offsets.append(offsets[-1] + n)
    total = sum(numels)
    flat_grad = torch.zeros(total, device="cpu", dtype=buffer_dtype)

    return {
        "param_names": param_names,
        "numels": numels,
        "offsets": offsets,
        "buff": flat_grad,
    }


def pack_grads(buff_meta, model):
    with torch.no_grad():
        param_map = dict(model.named_parameters())
        for param_name, off, n in zip(buff_meta["param_names"], buff_meta["offsets"], buff_meta["numels"], strict=False):
            p = param_map.get(param_name)
            buff_slice = buff_meta["buff"][off : off + n]
            if p is None or p.grad is None:
                buff_slice.zero_()
            else:
                buff_slice.copy_(p.grad.view(-1).to(device=buff_slice.device, dtype=buff_slice.dtype))


def unpack_to_grads(buff_meta, model):
    with torch.no_grad():
        param_map = dict(model.named_parameters())
        for param_name, off, n in zip(buff_meta["param_names"], buff_meta["offsets"], buff_meta["numels"], strict=False):
            p = param_map.get(param_name)
            if p is None:
                continue
            view = buff_meta["buff"][off : off + n].view_as(p).to(device=p.device, dtype=p.dtype)
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(view)


# --- getting averager ---
def build_grad_buff_from_model(
    model: nn.Module,
    expert_group_assignment: dict[int, dict[int, list[int]]],
    include_shared: bool = False,
    buffer_dtype: torch.dtype = torch.float16,
) -> dict[str | int, dict]:
    """
    Returns:
      - group_averagers: dict[group_id] -> DecentralizedAverager averaging *all* experts in that group
      - non_expert_averager: DecentralizedAverager averaging all non-expert tensors
    Notes:
      - All peers that should meet in the same group MUST use the same prefix (we derive it from group_id).
      - We sort tensor names to keep a deterministic order across peers.
    """
    # 1) Index tensors by name and prepare expert buckets
    all_named = [(name, param) for name, param in iter_named_params(model) if param.requires_grad]
    all_named.sort(key=lambda kv: kv[0])  # deterministic order
    name_to_tensor = dict(all_named)
    expert_group_to_names = {group_id: [] for group_id in list(expert_group_assignment.keys())}

    # Detect naming mode: per-expert (expert index embedded in name) vs stacked
    uses_per_expert_names = any(
        get_layer_expert_id(name)[1] is not None
        for name in name_to_tensor
    )

    for name, _ in name_to_tensor.items():
        layer_id, expert_id = get_layer_expert_id(name)

        if uses_per_expert_names:
            # ── per-expert mode: match by explicit expert_id ─────────────────
            if layer_id is not None and expert_id is not None:
                for group_id, layer_to_expert_ids in expert_group_assignment.items():
                    allowed_expert_ids = {int(a) for a, _ in layer_to_expert_ids.get(layer_id, [])} | {
                        int(b) for _, b in layer_to_expert_ids.get(layer_id, [])
                    }
                    if int(expert_id) in allowed_expert_ids:
                        expert_group_to_names[group_id].append(name)
        else:
            # ── stacked mode: match by layer_id, exclude shared_experts ──────
            is_routed_expert = "experts" in name and "shared_experts" not in name and layer_id is not None
            if is_routed_expert:
                for group_id, layer_to_expert_ids in expert_group_assignment.items():
                    if layer_id in layer_to_expert_ids:
                        expert_group_to_names[group_id].append(name)

    # 2) Build gradient buffer per expert group
    group_buff_metas: dict[str | int, Any] = {}
    buffer_element_size = torch.empty((), dtype=buffer_dtype).element_size()
    for group_id in expert_group_to_names.keys():
        tensors_for_group = {name: name_to_tensor[name] for name in expert_group_to_names[group_id]}
        if len(tensors_for_group) == 0:
            logger.warning(
                "No tensors found for expert group",
                group_id=group_id,
            )
        group_buff_metas[group_id] = build_buff_from_params(
            param_dict=tensors_for_group,
            buffer_dtype=buffer_dtype,
        )
        total_numel = sum(t.numel() for t in tensors_for_group.values())
        logger.info(
            f"Built expert group grad buffer - {group_id}",
            tensor_count=len(tensors_for_group),
            buffer_dtype=str(buffer_dtype),
            total_numel=total_numel,
            approx_buffer_mb=round(total_numel * buffer_element_size / (1024 * 1024), 2),
        )

    if include_shared:
        expert_owned_names = {name for names in expert_group_to_names.values() for name in names}
        non_expert_names = [n for n, _t in all_named if n not in expert_owned_names]
        non_expert_tensors = {n: name_to_tensor[n] for n in non_expert_names}
        group_buff_metas["shared"] = build_buff_from_params(
            param_dict=non_expert_tensors,
            buffer_dtype=buffer_dtype,
        )
        shared_numel = sum(t.numel() for t in non_expert_tensors.values())
        logger.info(
            "Built shared grad buffer",
            tensor_count=len(non_expert_tensors),
            total_param_count=len(all_named),
            buffer_dtype=str(buffer_dtype),
            total_numel=shared_numel,
            approx_buffer_mb=round(shared_numel * buffer_element_size / (1024 * 1024), 2),
        )
    else:
        logger.debug(
            "Skipping shared grad buffer",
            reason="include_shared=False",
            total_param_count=len(all_named),
        )

    return group_buff_metas


def build_averagers_from_buff(
    group_buff_metas: dict[int | str, dict[str, torch.Tensor]],
    dht: hivemind.DHT,
    prefix_base: str = "expert_averaging",
    target_group_size: int = 8,
    min_group_size: int = 2,
    averaging_alpha: float = 1.0,
) -> dict[str | int, DecentralizedAverager]:
    """
    Returns:
      - group_averagers: dict[group_id] -> DecentralizedAverager averaging *all* experts in that group
      - non_expert_averager: DecentralizedAverager averaging all non-expert tensors
    Notes:
      - All peers that should meet in the same group MUST use the same prefix (we derive it from group_id).
      - We sort tensor names to keep a deterministic order across peers.
    """

    group_averagers: dict[str | int, DecentralizedAverager] = {}
    try:
        visible_maddrs = [str(addr) for addr in dht.get_visible_maddrs()]
    except Exception:
        visible_maddrs = []
    if not visible_maddrs:
        logger.warning("No visible peers for averager matchmaking", prefix_base=prefix_base)
    else:
        logger.info(
            "Averager matchmaking context",
            prefix_base=prefix_base,
            visible_peers=len(visible_maddrs),
            target_group_size=target_group_size,
            min_group_size=min_group_size,
        )
        logger.debug("Visible peer addresses", addrs=visible_maddrs)
    for group_id, buff_meta in group_buff_metas.items():
        prefix = f"{prefix_base}-group{group_id}"
        group_averagers[group_id] = DecentralizedAverager(
            averaged_tensors=[buff_meta["buff"]],
            dht=dht,
            start=True,
            prefix=prefix,
            target_group_size=target_group_size,
            min_group_size=min_group_size,
            # allreduce_timeout = 60 * 5,
            # min_matchmaking_time = 60 * 5,
            # request_timeout = 60 * 2
        )
        avg = group_averagers[group_id]
        logger.info(
            "Averager ready",
            group=group_id,
            prefix=prefix,
            mode=avg.mode,
            peers=avg.total_size,
            target_group_size=target_group_size,
            min_group_size=min_group_size,
        )

    return group_averagers


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

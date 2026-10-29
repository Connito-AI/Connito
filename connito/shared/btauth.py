"""btauth/1 request signing with a wallet's hotkey, as bittensor 11's `http_auth.sign` with no receiver."""
from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable


def btauth_payload(method: str, target: str, body: bytes, nonce_ns: int, hotkey_ss58: str) -> bytes:
    """The signed bytes: `http_auth.build_payload` for an sr25519 key, bound to no receiver ("-")."""
    lines = ("btauth/1", "sr25519", method.upper(), target, hashlib.sha256(body).hexdigest(), str(nonce_ns),
             hotkey_ss58, "-")
    return "\n".join(lines).encode()


def wallet_signer(wallet) -> Callable[[str, str, bytes], dict[str, str]]:
    """sign(method, target, body) -> its auth headers, with a nonce above the last (btauth/1 refuses a reused one)."""
    lock, last = threading.Lock(), [0]

    def sign(method: str, target: str, body: bytes) -> dict[str, str]:
        with lock:
            last[0] = max(time.time_ns(), last[0] + 1)
            nonce = last[0]
        hotkey = wallet.hotkey
        signature = hotkey.sign(btauth_payload(method, target, body, nonce, hotkey.ss58_address))
        return {
            "X-Bittensor-Version": "1",
            "X-Bittensor-Hotkey": hotkey.ss58_address,
            "X-Bittensor-Nonce": str(nonce),
            "X-Bittensor-Signature": "0x" + bytes(signature).hex(),
        }

    return sign

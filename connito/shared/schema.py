import base64

import bittensor as bt
from bittensor import Keypair



def b64url_decode_nopad(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def sign_message(origin_hotkey: Keypair, message: bytes) -> str:
    sig = origin_hotkey.sign(message)  # bytes (likely 64 bytes)
    # URL-safe Base64, no padding (=)
    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")


def verify_message(origin_hotkey_ss58: str, message: bytes, signature_hex: str) -> bool:
    """
    Verify the signature for the message: pubkey || block
    signed by the hotkey at `my_hotkey_ss58_address`.
    """
    # 1. Rebuild signer keypair from their SS58
    signer_kp = bt.Keypair(ss58_address=origin_hotkey_ss58)

    # 2. Decode signature
    signature = b64url_decode_nopad(signature_hex)

    # 3. Verify
    return signer_kp.verify(message, signature)

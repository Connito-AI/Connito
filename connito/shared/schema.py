import base64
from dataclasses import asdict, dataclass
from pathlib import Path

import bittensor as bt
from substrateinterface import Keypair

from connito.shared.checkpoint_helper import compile_full_state_dict_from_path
from connito.shared.helper import get_model_hash


@dataclass
class SignedMessage:
    target_hotkey_ss58: str
    origin_hotkey_ss58: str
    origin_block: int
    signature: str  # hex string or raw bytes

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

@dataclass
class SignedDownloadRequestMessage(SignedMessage):
    expert_group_id: int | str | None = None


@dataclass()
class SignedModelSubmitMessage(SignedMessage):
    model_hex: str
    block_hex: str


def construct_model_message(model_path: str | Path, expert_groups: list[int | str] | None = None) -> bytes:
    """
    Sign:
        model_hash(32 bytes) || construct_block_message(...)
    """
    # 1. Get model hash
    model_hash = get_model_hash(compile_full_state_dict_from_path(model_path))

    return model_hash


def construct_block_message(target_hotkey_ss58: str, block: int) -> bytes:
    """
    Construct message: pubkey(32 bytes) || block(u64 big-endian)
    """
    # Convert SS58 → raw 32-byte pubkey
    target_kp = bt.Keypair(ss58_address=target_hotkey_ss58)
    pubkey_bytes = target_kp.public_key

    if len(pubkey_bytes) != 32:
        raise ValueError("Public key must be 32 bytes!")

    # Convert block → 8 bytes big-endian
    block_bytes = block.to_bytes(8, "big")

    # Final message
    return pubkey_bytes + block_bytes


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

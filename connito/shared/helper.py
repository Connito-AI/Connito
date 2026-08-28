import hashlib
import importlib
import os
from pathlib import Path
from typing import Any

from ipaddress import ip_address
from typing import Iterable, List

import torch
import torch.nn.functional as F


# File extensions accepted as miner-checkpoint formats. `.safetensors` is the
# preferred path — no pickle, no code-execution surface. `.pt` is still
# accepted for backwards compatibility with miners that haven't migrated.
MINER_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".pt")


def expert_group_shard_name(group_id: int | str, suffix: str = ".safetensors") -> str:
    """The filename every participant reads and writes for one expert group.

    Centralised because the writer moved to `.safetensors` while two readers
    kept asking for `.pt`, which silently broke miner distribution and
    validator peer-sync — `download_checkpoint_from_hf` has no fallback.
    """
    return f"model_expgroup_{group_id}{suffix}"


def load_state_dict_from_path(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Load a miner-checkpoint state_dict from `.safetensors` or `.pt`.

    `.safetensors` is loaded directly (no pickle path). `.pt` is loaded with
    `weights_only=True` so a malicious miner cannot execute code via a
    crafted `__reduce__` payload. Returned dict is always a flat
    {param_name: Tensor}; `.pt` files that wrap weights in
    `{"model_state_dict": ...}` are unwrapped here.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format at {path}: {type(obj).__name__}")


def sum_model_gradients(model):
    """
    Returns the sum of absolute gradients of all model parameters.
    Assumes backward() has already been called.
    """
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total += param.grad.detach().abs().to(torch.float64).sum().item()
        return total
    
    
def route_tokens_to_experts(router_logits):
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, 10, dim=-1)
    if True:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)
    return selected_experts, routing_weights


def convert_to_str(obj):
    """
    Recursively convert Path/PosixPath objects to strings
    inside any dict, list, or tuple.
    """

    if isinstance(obj, dict):
        return {k: convert_to_str(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_to_str(i) for i in obj]

    if isinstance(obj, tuple):
        return tuple(convert_to_str(i) for i in obj)

    if not isinstance(obj, int) and not isinstance(obj, float) and obj is not None:
        return str(obj)

    return obj


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def import_from_string(path: str) -> type:
    """
    Import a class from a string like 'package.module:ClassName'.
    """
    module_path, class_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_nested_attr(obj, attr_chain, default=None):
    for attr in attr_chain.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return default
    return obj


def parse_dynamic_filename(filename: str) -> dict:
    """
    Parse filenames like key_val_key_val... into a dictionary.
    Example:
        uid_13_hotkey_5FnRrH_block_5759026.pt
    → {"uid": 13, "hotkey": "5FnRrH", "block": 5759026}
    """
    # Remove .pt extension
    name = Path(filename).stem

    parts = name.split("_")
    meta = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        value = parts[i + 1]

        # Handle potential composite keys (non-even splits)
        # Example: if filename has uneven underscores
        if key in meta:  # duplicate key, skip
            i += 1
            continue

        # Try to cast numeric values to int
        try:
            value = int(value)
        except ValueError:
            pass

        meta[key] = value
        i += 2

    meta["filename"] = Path(filename)

    return meta


def find_submission_for_hotkey(
    submission_dir: Path,
    hotkey: str,
    submission_block_range: tuple[int, int] | None,
) -> Path | None:
    """Return the on-disk submission for `hotkey` that belongs to this round.

    Only the *start* of `submission_block_range` is meaningful. The block in
    the filename is stamped by `BackgroundDownloadWorker` at download time,
    not by the miner at submission time, and downloads run for the whole
    round — so anything fetched after the ~60-block submission phase closes
    fails an upper-bound test. That is every file but the first few minutes'
    worth, and it silently broke both callers: bg-download re-fetched shards
    it already had, and `publish_round_baseline` never once found its winner.

    `block >= start` is the correct predicate and needs no cycle length (the
    configured one disagrees with the chain's). Nothing can carry a future
    block, and the next freeze prunes anything below the new round id.

    If the range is None (legacy path / round without a window) fall back to
    hotkey-only match.
    """
    candidates = [
        p for suffix in MINER_CHECKPOINT_SUFFIXES
        for p in submission_dir.glob(f"*{suffix}")
    ]
    for path in candidates:
        if path.name.startswith(".tmp"):
            continue
        meta = parse_dynamic_filename(path.name)
        if not meta or meta.get("hotkey") != hotkey:
            continue
        if submission_block_range is not None:
            block = meta.get("block")
            if not isinstance(block, int):
                continue
            if block < submission_block_range[0]:
                continue
        return path
    return None


def h256_int(*parts: Any) -> int:
    """Deterministic 256-bit hash -> int."""
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"\x00")  # separator
    return int.from_bytes(m.digest(), "big")


def serialize_torch_model_path(state) -> bytes:
    """
    Load a torch model from disk and serialize its state_dict
    deterministically into raw bytes.
    """
    # If it's a full model, extract state_dict
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    elif not isinstance(state, dict):
        raise ValueError("Model file must contain a state_dict or nn.Module")

    buffer = []
    for key, tensor in sorted(state.items(), key=lambda item: item[0]):
        buffer.append(key.encode())
        if tensor.dtype == torch.bfloat16:
            buffer.append(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        else:
            buffer.append(tensor.cpu().numpy().tobytes())

    return b"".join(buffer)


def hash_model_bytes(model_bytes: bytes) -> bytes:
    """
    Blake2b-256 hash (32 bytes) of the model.
    """
    return hashlib.blake2b(model_bytes, digest_size=24).digest()


def get_model_hash(state, hex=False):
    """
    Create a model hash from model mocated at specified path.
    """
    # 1. Serialize model → bytes
    model_bytes = serialize_torch_model_path(state)

    # 2. Hash model to 32 bytes
    model_hash = hash_model_bytes(model_bytes)
    if hex:
        return model_hash.hex()
    else:
        return model_hash

def hex_to_byte(hex_str: str) -> bytes:
    """
    Convert hex string to raw bytes.
    """
    return bytes.fromhex(hex_str)

def public_multiaddrs(maddrs: Iterable) -> List:
    """
    Keep only multiaddrs whose /ip4 or /ip6 component is globally routable.
    Works with hivemind Multiaddr objects or strings.
    """
    out = []
    for ma in maddrs:
        s = str(ma)

        # multiaddr tokens look like: /ip4/1.2.3.4/tcp/1234/p2p/...
        parts = s.strip("/").split("/")
        ip = None
        for i in range(len(parts) - 1):
            if parts[i] in ("ip4", "ip6"):
                ip = parts[i + 1]
                break

        if ip is None:
            continue

        addr = ip_address(ip)

        # "public" => globally routable (not private/loopback/link-local/multicast/etc.)
        if addr.is_global:
            out.append(ma)

    return out
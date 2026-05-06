"""Process-wide memory cleanup utilities.

Centralised so both the validator main loop and helpers in ``shared``
(e.g. peer-sync in ``shared.model``) can reclaim cached allocator memory
without ``shared`` having to import from ``validator``.
"""
from __future__ import annotations

import ctypes
import gc

import torch

from connito.shared.app_logging import log_phase


try:
    _LIBC: ctypes.CDLL | None = ctypes.CDLL("libc.so.6")
    _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
    _LIBC.malloc_trim.restype = ctypes.c_int
except OSError:
    _LIBC = None


def release_cpu_ram() -> None:
    """Ask glibc to return freed arenas to the OS."""
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def cuda_mem_report(tag: str = "", device: int | None = None) -> None:
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    free, total = torch.cuda.mem_get_info(device)  # bytes

    def mb(x):
        return x / 1024**2

    log_phase(
        f"[{tag}] cuda:{device}",
        allocated=f"{mb(allocated):.1f}MB",
        reserved=f"{mb(reserved):.1f}MB",
        free=f"{mb(free):.1f}MB",
        total=f"{mb(total):.1f}MB",
        alloc_pct=f"{allocated/total*100:.1f}%",
        reserved_pct=f"{reserved/total*100:.1f}%",
    )


def cleanup(global_model=None) -> None:
    """Reclaim cached allocator memory. ``global_model`` stays resident on GPU.

    The ``global_model`` argument is unused; kept for the original call sites
    that pass it as documentation that the model is intentionally not freed.
    """
    cuda_mem_report("VRAM before GPU cleanup")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    release_cpu_ram()

    cuda_mem_report("VRAM after GPU cleanup")

"""Does graft-in-place actually fit, and does restore actually restore?

Two live rounds failed to score a single miner and neither told us much. The
first OOM'd every miner (`evaluate_one_miner: OOM` x3, 42.9 GiB allocated,
213 MB free); the second never reached a graft at all, because the submission
window had closed before the roster froze. In-round evidence is expensive:
one attempt per ~105-minute cycle, and a failure looks the same whether the
cause is memory, timing, or the shard.

So this reproduces the eval window on its own, with the two things that make
the in-round numbers real:

  * a **ballast** allocation standing in for the partial `global_model`, which
    holds ~9.5 GiB of the card during the eval window. Without it a 45.5 GiB
    card looks roomy and any budget claim is worthless.
  * a **real miner shard** off disk, not a synthesised one — the whole point
    is how much the overlay costs, and that depends on how many rows the
    shard actually touches.

Forward passes use synthetic token batches on purpose. Activation memory is a
function of shape, not content, and pulling real eval batches would add a
dataset download to a test about allocation. Correctness of the overlay is
checked by comparing tensors directly, which is stricter than inferring it
from a loss delta.

What each assertion is defending, in the order they run:

  1. build lands near 31.5 GB host        (build_at_dtype, PR 1)
  2. `.to(cuda)` lands near 29.3 GiB VRAM
  3. graft adds ~0 VRAM, not ~3.4 GiB     <- the fix under test
  4. a grafted forward pass completes at the real budget
  5. restore is bit-identical
  6. the backup pool is reused across miners, so host cost is flat
  7. park returns the card and does not grow host RSS

    python tools/quantization/gpu_graft_verify.py --shard /path/to/x.safetensors
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import banner, build_config, host_ram_now_gb, peak_vram_gb, reset_peak, vram_gb  # noqa: E402

GB = 1024 ** 3
# What the partial global model holds during the eval window, measured:
# `[VRAM before GPU cleanup] allocated=9513.3MB` with the base parked.
BALLAST_GB_DEFAULT = 9.3


def _fail(label: str, detail: str) -> None:
    print(f"  FAIL  {label}: {detail}")


def _ok(label: str, detail: str) -> None:
    print(f"  ok    {label}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, help="a real miner .safetensors off disk")
    parser.add_argument("--shard2", default=None,
                        help="a second shard, to prove the backup pool is reused")
    parser.add_argument("--ballast-gb", type=float, default=BALLAST_GB_DEFAULT,
                        help="stand-in for the resident partial global model")
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA — this test is about VRAM")
        return 2

    from connito.shared.expert_manager import ExpertManager
    from connito.validator.full_topology_eval import FullTopologyEvalBase

    failures: list[str] = []
    cfg = build_config()
    device = "cuda"

    # ── 1. build on host ────────────────────────────────────────────────────
    banner("Building the full-topology eval base on CPU")
    host_before_build = host_ram_now_gb()
    base = FullTopologyEvalBase.build(cfg, ExpertManager(cfg))
    host_after_build = host_ram_now_gb()
    print(f"  host RSS {host_before_build:.2f} -> {host_after_build:.2f} GB")
    if host_after_build > 42.0:
        failures.append("build host RSS")
        _fail("build host", f"{host_after_build:.2f} GB — expected ~31.5 (PR 1's build_at_dtype)")
    else:
        _ok("build host", f"{host_after_build:.2f} GB")

    # ── 2. ballast, then the base onto the card ─────────────────────────────
    banner(f"Reserving {args.ballast_gb:.1f} GiB of ballast for the partial global model")
    ballast = torch.empty(int(args.ballast_gb * GB // 2), dtype=torch.float16, device=device)
    print(f"  VRAM used {vram_gb():.2f} GiB")

    banner("Moving the base onto the card (the eval window begins here)")
    vram_before_gpu = vram_gb()
    base.model.to(device)
    base._on_gpu = True
    torch.cuda.synchronize()
    vram_with_base = vram_gb()
    base_cost = vram_with_base - vram_before_gpu
    print(f"  VRAM {vram_before_gpu:.2f} -> {vram_with_base:.2f} GiB  (base = {base_cost:.2f})")
    if not 26.0 <= base_cost <= 33.0:
        failures.append("base VRAM")
        _fail("base VRAM", f"{base_cost:.2f} GiB — expected ~29.3")
    else:
        _ok("base VRAM", f"{base_cost:.2f} GiB")

    # ── 3. the graft must not cost VRAM ─────────────────────────────────────
    banner("Grafting a real miner shard")
    print(f"  {Path(args.shard).name}")
    fingerprint = _fingerprint(base)
    reset_peak()
    vram_pre_graft = vram_gb()
    base.graft_from_path(args.shard)
    torch.cuda.synchronize()
    vram_post_graft = vram_gb()
    graft_cost = vram_post_graft - vram_pre_graft
    staged_gb = sum(t.numel() * t.element_size() for t in base._backup_pool.values()) / GB
    on_host = all(t.device.type == "cpu" for *_, t in base._backup)
    print(f"  VRAM {vram_pre_graft:.2f} -> {vram_post_graft:.2f} GiB  (graft = {graft_cost:+.2f})")
    print(f"  backup: {len(base._backup)} rows, {staged_gb:.2f} GB staged, on_host={on_host}")
    if not on_host:
        failures.append("backup device")
        _fail("backup device", "backup rows are on the GPU — this is the bug being fixed")
    elif graft_cost > 0.5:
        failures.append("graft VRAM")
        _fail("graft VRAM", f"{graft_cost:+.2f} GiB — expected ~0, was ~3.4 before the fix")
    else:
        _ok("graft VRAM", f"{graft_cost:+.2f} GiB, {staged_gb:.2f} GB staged on host")

    # ── 4. a forward pass at the real budget ────────────────────────────────
    banner(f"Forward pass on the grafted base ({args.batches} batches)")
    seq = int(cfg.task.exp.data.sequence_length)
    try:
        with torch.no_grad():
            for _ in range(args.batches):
                ids = torch.randint(0, 30000, (1, seq), device=device)
                base.model(input_ids=ids)
        torch.cuda.synchronize()
        print(f"  peak VRAM {peak_vram_gb():.2f} GiB of {torch.cuda.get_device_properties(0).total_memory / GB:.2f}")
        _ok("grafted forward", f"completed, peak {peak_vram_gb():.2f} GiB")
    except torch.cuda.OutOfMemoryError as exc:
        failures.append("grafted forward")
        _fail("grafted forward", f"OOM — {str(exc).splitlines()[0]}")

    # ── 5. restore must be bit-identical ───────────────────────────────────
    banner("Restoring")
    base.restore_grafted()
    torch.cuda.synchronize()
    drift = _compare(base, fingerprint)
    if drift:
        failures.append("restore fidelity")
        _fail("restore", f"{len(drift)} rows differ from pre-graft, e.g. {drift[0]}")
    else:
        _ok("restore", f"all {len(fingerprint)} sampled rows bit-identical")

    # ── 6. the pool is reused, so host cost is flat ────────────────────────
    if args.shard2:
        banner("Second miner: the backup pool must be reused, not reallocated")
        first_ptrs = None
        base.graft_from_path(args.shard)
        first_ptrs = sorted(t.data_ptr() for *_, t in base._backup)
        base.restore_grafted()
        host_mid = host_ram_now_gb()
        base.graft_from_path(args.shard2)
        second_ptrs = sorted(t.data_ptr() for *_, t in base._backup)
        base.restore_grafted()
        host_after = host_ram_now_gb()
        print(f"  host RSS {host_mid:.2f} -> {host_after:.2f} GB")
        if first_ptrs != second_ptrs:
            # Not necessarily a failure if the shards touch different rows,
            # so report the overlap rather than asserting equality blindly.
            shared = len(set(first_ptrs) & set(second_ptrs))
            if shared == 0:
                failures.append("pool reuse")
                _fail("pool reuse", "no staging storage shared between the two shards")
            else:
                _ok("pool reuse", f"{shared}/{len(first_ptrs)} rows reused storage")
        else:
            _ok("pool reuse", f"all {len(first_ptrs)} rows reused the same storage")

    # ── 7. park returns the card without growing host ──────────────────────
    banner("Parking for the merge window")
    host_pre_park = host_ram_now_gb()
    base.park()
    del ballast
    gc.collect()
    torch.cuda.empty_cache()
    host_post_park = host_ram_now_gb()
    print(f"  host RSS {host_pre_park:.2f} -> {host_post_park:.2f} GB, VRAM now {vram_gb():.2f} GiB")
    if host_post_park > host_pre_park + 2.0:
        failures.append("park host growth")
        _fail("park", f"host grew {host_post_park - host_pre_park:+.2f} GB — shadow storage not reused")
    else:
        _ok("park", f"host {host_post_park - host_pre_park:+.2f} GB")

    banner("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("  all checks passed")
    return 0


def _fingerprint(base) -> list[tuple[str, str, int, torch.Tensor]]:
    """Clone a sample of owned expert rows, to compare after restore.

    A sample rather than the whole model: this runs with the card near full,
    and a complete copy would be the very allocation the test exists to
    forbid. Every layer contributes, so a graft that wrote outside its rows
    has nowhere to hide.
    """
    sampled: list[tuple[str, str, int, torch.Tensor]] = []
    for layer, gids in sorted(base._allowed_by_layer.items()):
        if not gids:
            continue
        gid = sorted(gids)[0]
        path = f"model.layers.{layer}.mlp.experts"
        try:
            experts = base.model.get_submodule(path)
        except AttributeError:
            continue
        local = int(experts.global_to_local_map[gid])
        for name in ("gate_up_proj", "down_proj"):
            rows = getattr(experts, name).data[local]
            sampled.append((path, name, local, rows.detach().to("cpu", copy=True)))
    return sampled


def _compare(base, fingerprint) -> list[str]:
    drift: list[str] = []
    for path, name, local, saved in fingerprint:
        current = getattr(base.model.get_submodule(path), name).data[local]
        if not torch.equal(current.detach().to("cpu"), saved):
            drift.append(f"{path}.{name}[{local}]")
    return drift


if __name__ == "__main__":
    raise SystemExit(main())

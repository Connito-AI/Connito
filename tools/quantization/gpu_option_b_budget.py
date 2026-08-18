"""Is option B — `global_model` on CPU, eval model on GPU — workable?

Option B avoids ever grafting miner weights into the model the validator
submits on-chain, by keeping `global_model` on the host and giving the GPU a
dedicated eval model. The question is whether the host can hold what that
implies.

What the merge actually needs resident, from `validator/run.py`:

    outer_optimizer = torch.optim.SGD(
        [p for p in global_model.parameters() if p.requires_grad],
        momentum=config.opt.outer_momentum, nesterov=True,
    )
    ...
    populate_global_grads_from_local(global_model, miner_model, weight=weight)
    outer_optimizer.step()

So three tensors per trainable parameter, not one: the parameter, its `.grad`
(written by `populate_global_grads_from_local`, freed afterwards by
`_release_global_model_grads`) and SGD's `momentum_buffer`, which is allocated
on the first `step()` and persists for the process lifetime.

This script reports the arithmetic from *real* parameter counts rather than
estimates, and times an SGD step on a representative slice to extrapolate the
CPU cost. It deliberately does NOT try to allocate the full optimizer state:
on a 62 GB host that would be a deliberate OOM, and this box is shared.

    python tools/quantization/gpu_option_b_budget.py [--partial]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import banner, build_config, host_ram_hwm_gb  # noqa: E402

GB = 1024 ** 3
HOST_RAM_GB = 62.0
L40S_VRAM_GB = 44.4  # usable, per the OOM report — not the 46 on the box sticker


def _gb(num_bytes: float) -> float:
    return num_bytes / GB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true",
                        help="measure the partial topology instead (today's fleet)")
    args = parser.parse_args()

    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.mycelia import get_base_model

    cfg = build_config()
    cfg.model.device = "cpu"  # option B's premise: the global model is not on the GPU
    partial = bool(args.partial)

    banner(f"Building the {'partial' if partial else 'FULL'} model on CPU")
    expert_manager = ExpertManager(cfg)
    model = get_base_model(
        cfg,
        expert_manager=expert_manager,
        group_ids_trainable=[cfg.task.exp.group_id] if partial else None,
        group_ids_helper=[cfg.task.helper_group_id] if partial else None,
        partial=partial,
    )
    print(f"built, host hwm {host_ram_hwm_gb():.1f} GB", flush=True)

    banner("Parameter census")
    total_numel = trainable_numel = 0
    element_size = 2
    for param in model.parameters():
        total_numel += param.numel()
        if param.requires_grad:
            trainable_numel += param.numel()
            element_size = param.element_size()

    params_gb = _gb(trainable_numel * element_size)
    print(f"total params      {total_numel/1e9:.2f} B")
    print(f"trainable params  {trainable_numel/1e9:.2f} B  "
          f"({100*trainable_numel/max(total_numel,1):.1f}%)")
    print(f"bytes/element     {element_size}")

    banner("What the merge needs resident, on the host")
    grads_gb = params_gb
    momentum_gb = params_gb
    print(f"parameters                    {params_gb:6.1f} GB")
    print(f"+ .grad (transient)           {grads_gb:6.1f} GB   "
          f"populate_global_grads_from_local")
    print(f"+ SGD momentum (persistent)   {momentum_gb:6.1f} GB   "
          f"allocated on first step(), never freed")
    merge_peak_gb = params_gb + grads_gb + momentum_gb
    print(f"{'':30}{'-'*8}")
    print(f"peak during outer step        {merge_peak_gb:6.1f} GB")
    print(f"steady state after it         {params_gb + momentum_gb:6.1f} GB")

    banner("Verdict against this host")
    print(f"host RAM                      {HOST_RAM_GB:6.1f} GB")
    headroom = HOST_RAM_GB - merge_peak_gb
    print(f"headroom at merge peak        {headroom:6.1f} GB  "
          f"{'OK' if headroom > 0 else '<-- DOES NOT FIT'}")
    print("  (before the grad buffer, the per-round model_snapshot_cpu,")
    print("   the torch runtime, or anything the eval side needs)")

    banner("CPU cost of one outer step")
    # Timed on a slice rather than the whole model: allocating momentum for
    # every parameter is the very thing this script is arguing will not fit.
    sample = [p for p in model.parameters() if p.requires_grad][:24]
    sample_numel = sum(p.numel() for p in sample)
    for p in sample:
        p.grad = torch.zeros_like(p)
    opt = torch.optim.SGD(sample, lr=0.7, momentum=0.9, nesterov=True)
    opt.step()  # allocates momentum; excluded from the timing below
    started = time.time()
    opt.step()
    elapsed = time.time() - started
    per_billion = elapsed / max(sample_numel / 1e9, 1e-9)
    print(f"sampled {len(sample)} tensors, {sample_numel/1e9:.3f} B params")
    print(f"step {elapsed*1000:.0f} ms  ->  {per_billion:.1f} s per B params")
    print(f"extrapolated full step: {per_billion * trainable_numel/1e9:.1f} s")

    print("\nRESULT " + json.dumps({
        "topology": "partial" if partial else "full",
        "trainable_params_b": round(trainable_numel / 1e9, 3),
        "params_gb": round(params_gb, 1),
        "merge_peak_gb": round(merge_peak_gb, 1),
        "steady_after_merge_gb": round(params_gb + momentum_gb, 1),
        "host_ram_gb": HOST_RAM_GB,
        "fits_host": merge_peak_gb < HOST_RAM_GB,
        "extrapolated_step_sec": round(per_billion * trainable_numel / 1e9, 1),
        "build_host_hwm_gb": round(host_ram_hwm_gb(), 2),
    }))


if __name__ == "__main__":
    main()

"""fp8 on a FULL-experts validator — the configuration the experiment repo uses.

Everything measured on this box so far ran the *partial* model: group 4 trainable
+ group 2 helper, 430 of 1664 routed experts. This exercises `partial=False`,
where `get_base_model` builds every routed expert (64 per MoE layer, 26 layers)
and routes `full_topk` instead of `partial_topk`. That path has never been run
here, so the first question is whether it builds at all — the full loader
materialises a CPU model *and* a full pretrained state_dict before copying, and
this box has 62 GB of RAM and no swap.

Arms, one process each so VRAM figures are not contaminated:

    fp16      full model, no quantization                (the baseline)
    fp8       full model, experts quantized, backbone fp16 (the validator config)
    deepcopy  full fp16 model + `copy.deepcopy` + quantize — the production
              foreground path (`resolve_foreground_eval_model`), which copies
              first and quantizes second, so peak is two full fp16 models

Reports resident/peak VRAM, host-RAM high-water mark, eval loss on seeded
batches, and the state_dict invariants that the submission hash depends on
(checked per-module on two layers rather than model-wide: a full-model
`state_dict()` dequantizes 28.8 GB of experts onto the host).

    python tools/quantization/gpu_full_experts.py --arm fp16|fp8|deepcopy [--batches 8]
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import (  # noqa: E402
    banner,
    build_config,
    drop_gated_sources,
    eval_batches,
    host_ram_hwm_gb,
    host_ram_now_gb,
    peak_vram_gb,
    reset_peak,
    score,
    vram_gb,
)

# Layers whose expert module gets the full state_dict invariant check. Layer 1
# is the first MoE layer (`first_k_dense_replace=1`), 26 the last.
PROBE_LAYERS = (1, 26)


def build_full_model(cfg):
    """The validator's full-experts model: `partial=False`, every routed expert.

    Mirrors `get_model_from_checkpoint(partial=False)`, which passes no trainable
    and no helper group ids — in full mode every expert is present, so there is
    no trainable/frozen split and no helper group to speak of.
    """
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.mycelia import get_base_model

    expert_manager = ExpertManager(cfg)
    started = time.time()
    model = get_base_model(
        cfg,
        expert_manager,
        group_ids_trainable=None,
        group_ids_helper=None,
        partial=False,
    )
    print(f"  built on CPU, host RSS {host_ram_now_gb():.1f} GB (hwm {host_ram_hwm_gb():.1f})", flush=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    gc.collect()
    model.eval()
    return model, expert_manager, time.time() - started


def expert_modules(model) -> list[tuple[str, torch.nn.Module]]:
    """Every stacked-expert holder, in layer order."""
    found = []
    for name, module in model.named_modules():
        if hasattr(module, "quantize_") and hasattr(module, "global_to_local_map"):
            found.append((name, module))
    return found


def probe_metadata(model) -> dict[str, dict]:
    """key -> {shape, dtype} for the probe layers' expert modules only."""
    out: dict[str, dict] = {}
    for name, module in expert_modules(model):
        layer = name.split(".")[2] if name.startswith("model.layers.") else name
        if layer not in {str(n) for n in PROBE_LAYERS}:
            continue
        for key, value in module.state_dict().items():
            out[f"{name}.{key}"] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["fp16", "fp8", "deepcopy"], required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--c4-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    from connito.shared.modeling.mycelia import get_base_tokenizer
    from connito.shared.modeling.quantization import (
        all_finite,
        is_quantized,
        quantize_model_,
    )

    banner(f"FULL EXPERTS — arm: {args.arm}")
    reset_peak()

    quantization = "off" if args.arm == "fp16" else "fp8"
    cfg = build_config(seq_len=args.seq_len, quantization=quantization)
    if args.c4_only:
        print(f"dropped gated sources: {drop_gated_sources(cfg)}")

    model, _, build_s = build_full_model(cfg)
    experts = expert_modules(model)
    routed = sum(m.num_local_experts for _, m in experts)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"build              {build_s:.0f}s")
    print(f"MoE layers         {len(experts)}")
    print(f"routed experts     {routed} ({routed // max(len(experts), 1)} per layer)")
    print(f"total params       {total_params / 1e9:.2f} B")
    print(f"vram after build   {vram_gb():.2f} GB")
    print(f"host ram hwm       {host_ram_hwm_gb():.1f} GB")

    meta_before = probe_metadata(model)

    result: dict = {
        "arm": args.arm,
        "build_s": round(build_s, 1),
        "moe_layers": len(experts),
        "routed_experts": routed,
        "total_params_b": round(total_params / 1e9, 3),
        "vram_after_build_gb": round(vram_gb(), 3),
        "host_ram_hwm_gb": round(host_ram_hwm_gb(), 2),
        "probe_keys": len(meta_before),
    }

    # ── the arm's own work ───────────────────────────────────────────────────
    eval_model = model
    if args.arm == "fp8":
        model.requires_grad_(False)
        started = time.time()
        converted = quantize_model_(model, include_experts=True, include_linears=False)
        torch.cuda.empty_cache()
        result["quantize_s"] = round(time.time() - started, 1)
        result["converted_modules"] = len(converted)
        result["vram_after_quant_gb"] = round(vram_gb(), 3)
        print(f"quantize           {result['quantize_s']:.1f}s -> {len(converted)} modules")
        print(f"vram after quant   {vram_gb():.2f} GB")
        assert is_quantized(model)

    elif args.arm == "deepcopy":
        # `resolve_foreground_eval_model` deepcopies `global_model` on-device and
        # quantizes the copy, so both exist in full fp16 at the same instant.
        print("attempting copy.deepcopy(global_model) — the production path", flush=True)
        try:
            started = time.time()
            eval_model = copy.deepcopy(model)
            result["deepcopy_s"] = round(time.time() - started, 1)
            result["vram_after_deepcopy_gb"] = round(vram_gb(), 3)
            print(f"deepcopy ok        {result['deepcopy_s']:.1f}s, vram {vram_gb():.2f} GB")
            eval_model.requires_grad_(False)
            converted = quantize_model_(eval_model, include_experts=True, include_linears=False)
            torch.cuda.empty_cache()
            result["converted_modules"] = len(converted)
            result["vram_after_quant_gb"] = round(vram_gb(), 3)
            result["deepcopy_oom"] = False
            print(f"quantized copy     {len(converted)} modules, vram {vram_gb():.2f} GB")
        except torch.cuda.OutOfMemoryError as exc:
            result["deepcopy_oom"] = True
            result["oom_message"] = str(exc).splitlines()[0]
            result["vram_at_oom_gb"] = round(vram_gb(), 3)
            print(f"\nOOM during the production copy-then-quantize path:\n  {result['oom_message']}")
            print(f"\nRESULT {json.dumps(result)}")
            return

    # ── invariants ───────────────────────────────────────────────────────────
    meta_after = probe_metadata(eval_model)
    result["probe_keys_identical"] = sorted(meta_before) == sorted(meta_after)
    result["probe_shapes_dtypes_identical"] = meta_before == meta_after
    if meta_before != meta_after:
        drift = [k for k in meta_before if meta_after.get(k) != meta_before[k]]
        result["probe_drift_sample"] = drift[:5]
    print(f"probe keys ({len(meta_before)}) identical: {result['probe_keys_identical']}")
    print(f"probe shape+dtype identical:  {result['probe_shapes_dtypes_identical']}")

    bad = [n for n, p in eval_model.named_parameters() if not all_finite(p)]
    bad += [
        n for n, b in eval_model.named_buffers()
        if b.is_floating_point() and not all_finite(b)
    ]
    result["non_finite"] = len(bad)
    print(f"non-finite tensors {len(bad)}{'' if not bad else ' ' + str(bad[:5])}")

    # ── eval loss on seeded batches ──────────────────────────────────────────
    if not args.skip_eval:
        tokenizer = get_base_tokenizer(cfg)
        batches = eval_batches(cfg, tokenizer, args.batches)
        reset_peak()
        started = time.time()
        val_loss = score(eval_model, batches, torch.device("cuda"))
        result["val_loss"] = val_loss
        result["eval_s"] = round(time.time() - started, 1)
        result["vram_peak_eval_gb"] = round(peak_vram_gb(), 3)
        print(f"\nval_loss           {val_loss:.6f}  ({len(batches)} batches, {result['eval_s']:.0f}s)")
        print(f"vram peak (eval)   {peak_vram_gb():.2f} GB")

    result["vram_resident_gb"] = round(vram_gb(), 3)
    result["host_ram_hwm_gb"] = round(host_ram_hwm_gb(), 2)
    print(f"host ram hwm       {host_ram_hwm_gb():.1f} GB")
    print(f"\nRESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()

"""fp8 on the STOCK HuggingFace DeepSeek-V2-Lite — the experiment repo's base model.

Our own full-model loader leaves the routed experts randomly initialised (fused
`gate_up_proj` keys never meet the checkpoint's `gate_proj.weight`), so a loss
delta measured there is noise. The experiment repo sidesteps that entirely by
loading the stock model — `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)`, `model.py:72` — whose module tree matches the
checkpoint key-for-key.

That makes this the cleanest available comparison against her numbers: same
model, same weights, and *our* `FP8Linear` applied exactly where her
`quantize_expert_fp8` applies hers — the three projections of every routed
expert, shared experts and backbone untouched.

Both arms run in one process against one set of seeded batches, so the loss
delta is paired and attributable to quantization alone.

What this can answer: fp8 weight error, loss delta on correctly-loaded weights,
resident VRAM. What it cannot: the rank gate — miner shards are keyed to our
fused stacked layout and expert-group assignment, neither of which exists here.

    python tools/quantization/gpu_hf_stock_fp8.py [--batches 8] [--no-trust-remote-code]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import (  # noqa: E402
    EVAL_SEED,
    MODEL_PATH,
    banner,
    build_config,
    drop_gated_sources,
    eval_batches,
    host_ram_hwm_gb,
    peak_vram_gb,
    reset_peak,
    score,
    vram_gb,
)

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_stock_model(trust_remote_code: bool):
    """Her loader, verbatim in the parts that matter: bf16, low_cpu_mem_usage,
    no device_map (that branch is 4-bit only), then `.to(device)`."""
    from transformers import AutoModelForCausalLM

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    print(f"  loaded on CPU, host hwm {host_ram_hwm_gb():.1f} GB", flush=True)
    model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)
    return model, time.time() - started


def routed_expert_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Every routed-expert MLP: has all three projections as `nn.Linear`, and
    sits under `.experts.` rather than `.shared_experts.`."""
    blocks = []
    for name, module in model.named_modules():
        if ".experts." not in f"{name}." or "shared_experts" in name:
            continue
        if all(isinstance(getattr(module, p, None), nn.Linear) for p in PROJECTIONS):
            blocks.append((name, module))
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--c4-only", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    from connito.shared.modeling.mycelia import get_base_tokenizer
    from connito.shared.modeling.quantization import (
        FP8Linear,
        all_finite,
        dequantize_for_compute,
        quantize_last_dim,
    )

    device = torch.device("cuda")
    reset_peak()

    banner("Loading stock HuggingFace DeepSeek-V2-Lite")
    model, load_s = load_stock_model(not args.no_trust_remote_code)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"load               {load_s:.0f}s")
    print(f"total params       {total_params / 1e9:.2f} B")
    print(f"vram resident      {vram_gb():.2f} GB")
    print(f"host ram hwm       {host_ram_hwm_gb():.1f} GB")

    blocks = routed_expert_blocks(model)
    print(f"routed expert MLPs {len(blocks)}")
    if not blocks:
        # Newer transformers packs experts into `gate_up_proj`/`down_proj`
        # tensors; her wrapper handles that case by unpacking, we would need to
        # as well. Report rather than silently measuring nothing.
        packed = [
            n for n, m in model.named_modules()
            if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")
        ]
        raise SystemExit(
            "No per-expert nn.Linear blocks found. This transformers version "
            f"packs experts ({len(packed)} packed modules, e.g. {packed[:2]}); "
            "unpack before quantizing."
        )

    banner("Baseline: bf16, seeded batches")
    cfg = build_config(seq_len=args.seq_len)
    if args.c4_only:
        print(f"dropped gated sources: {drop_gated_sources(cfg)}")
    tokenizer = get_base_tokenizer(cfg)
    batches = eval_batches(cfg, tokenizer, args.batches)
    print(f"{len(batches)} batches, seed {EVAL_SEED[:16]}...", flush=True)

    reset_peak()
    loss_bf16 = score(model, batches, device)
    peak_bf16 = peak_vram_gb()
    vram_bf16 = vram_gb()
    print(f"val_loss (bf16)    {loss_bf16:.6f}")
    print(f"vram peak          {peak_bf16:.2f} GB")

    banner("Quantizing the three projections of every routed expert")
    # Mirrors `quantize_expert_fp8` (partial_moe.py:65-75) — same modules, same
    # per-output-row symmetric scaling, our FP8Linear instead of hers.
    rel_errors: list[float] = []
    converted = 0
    started = time.time()
    for _, block in blocks:
        for projection in PROJECTIONS:
            linear = getattr(block, projection)
            weight = linear.weight.data
            values, scale = quantize_last_dim(weight)
            # Compute-path dequant (activation dtype), so the error reported is
            # the error the forward pass actually sees.
            round_trip = dequantize_for_compute(values, scale, weight.dtype)
            denominator = weight.float().norm()
            if denominator > 0:
                rel_errors.append(
                    ((round_trip.float() - weight.float()).norm() / denominator).item()
                )
            setattr(block, projection, FP8Linear.from_linear(linear))
            converted += 1
    torch.cuda.empty_cache()
    quantize_s = time.time() - started

    errors = torch.tensor(rel_errors)
    print(f"converted          {converted} projections in {quantize_s:.0f}s")
    print(f"vram resident      {vram_gb():.2f} GB")
    print(
        f"relative weight error (Frobenius, per projection): "
        f"mean {errors.mean().item() * 100:.3f}%  "
        f"min {errors.min().item() * 100:.3f}%  "
        f"max {errors.max().item() * 100:.3f}%"
    )

    bad = [n for n, p in model.named_parameters() if not all_finite(p)]
    bad += [n for n, b in model.named_buffers() if b.is_floating_point() and not all_finite(b)]
    print(f"non-finite tensors {len(bad)}{'' if not bad else ' ' + str(bad[:5])}")

    banner("fp8: identical batches")
    reset_peak()
    loss_fp8 = score(model, batches, device)
    peak_fp8 = peak_vram_gb()
    vram_fp8 = vram_gb()
    print(f"val_loss (fp8)     {loss_fp8:.6f}")
    print(f"vram peak          {peak_fp8:.2f} GB")

    banner("RESULT")
    print(f"val_loss  bf16 {loss_bf16:.6f} -> fp8 {loss_fp8:.6f}   delta {loss_fp8 - loss_bf16:+.6f}")
    print(f"vram      bf16 {vram_bf16:.2f} GB -> fp8 {vram_fp8:.2f} GB   saved {vram_bf16 - vram_fp8:.2f} GB")

    result = {
        "model": MODEL_PATH,
        "trust_remote_code": not args.no_trust_remote_code,
        "total_params_b": round(total_params / 1e9, 3),
        "routed_expert_mlps": len(blocks),
        "converted_projections": converted,
        "quantize_s": round(quantize_s, 1),
        "batches": len(batches),
        "val_loss_bf16": loss_bf16,
        "val_loss_fp8": loss_fp8,
        "val_loss_delta": loss_fp8 - loss_bf16,
        "rel_weight_error_mean_pct": errors.mean().item() * 100,
        "rel_weight_error_min_pct": errors.min().item() * 100,
        "rel_weight_error_max_pct": errors.max().item() * 100,
        "vram_bf16_gb": round(vram_bf16, 3),
        "vram_fp8_gb": round(vram_fp8, 3),
        "vram_saved_gb": round(vram_bf16 - vram_fp8, 3),
        "vram_peak_bf16_gb": round(peak_bf16, 3),
        "vram_peak_fp8_gb": round(peak_fp8, 3),
        "non_finite": len(bad),
        "host_ram_hwm_gb": round(host_ram_hwm_gb(), 2),
    }
    print(f"\nRESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()

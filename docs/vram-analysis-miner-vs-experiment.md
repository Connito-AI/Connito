# VRAM analysis — Connito miner (2Fnat) vs. experiment repo (2Fnat)

Why the Connito miner needed `MINER_ADAMW_OPTIM_BITS=8` (bnb `AdamW8bit`) to fit on a 47 GB A6000, while the experiment repo's **same-paradigm** 2Fnat runs (e.g. `scripts/tier5_2Fnat_code_lr1e-4.sh` — DeepSeek-V2-Lite, `--routing-mode natural_with_fallback`, `--frozen-kept-assignment-path` c4-p02 helpers) fit comfortably with `torch.optim.AdamW` at full fp32 state.

Both runs use the same base model, the same routing rule (2Fnat), and the same c4-p02 helper set — so helper cost is a wash, not a delta. Both run on the same GPU (A6000 47.4 GB), same host, and both set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## TL;DR

Two independent knobs push the miner ~15–25 GB above the 2Fnat experiment's footprint. Both need to be fixed (or one of them plus 8-bit AdamW) for the miner to fit in 47.4 GB:

| # | Delta | Approx cost |
|---|---|---|
| 1 | Miner runs at **`sequence_length=4096`**, 2Fnat experiment at **`1024`** | ~3–8 GB activations (grad-ckpt on both, but attention scratch scales with seq²) |
| 2 | Miner uses **fp16-mixed + `freeze_parameters(upcast_trainable=True)` + fp32 grads for `GradScaler.unscale_`**. 2Fnat experiment uses bf16 native — no shadow fp32 param copy, grads stay bf16 | ~3 GB shadow fp32 params + ~3.7 GB fp32-vs-bf16 grad delta = ~7 GB |

A third knob was initially suspected to matter — the **model rebuild every inner-opt step** in `reload_model_inplace` (before `ckpt.enable_peer_resync=false`). Empirical test confirms it wasn't the fit delta: disabling the reload buys back only ~470 MiB of fragmentation (`exp_avg` alloc goes from failing to succeeding, but `exp_avg_sq` still OOMs on the same run). fp32 AdamW state is simply too big for this setup — the reload path was just adding noise on top of an already-over-capacity fit.

2Fnat routing itself contributes 0 GB — it's a compute-only scatter/gather + topk. The helper group loading is present in *both* runs, so it cancels out of the comparison.

## Component-by-component footprint

DeepSeek-V2-Lite: `hidden_size=2048`, `moe_intermediate=1408`, `n_routed_experts=64`, 26 MoE layers, 1 dense layer, vocab=102400.

Per expert (fp16): `gate_up_proj` = `2 × 1408 × 2048 × 2 B` ≈ 11.5 MB, `down_proj` = `1408 × 2048 × 2 B` ≈ 5.8 MB → **~17.3 MB / expert / layer**.

| Component | 2Fnat experiment (`tier5_2Fnat_code_lr1e-4.sh` style) | Connito miner (exp_math + exp_c4_p02 helper) |
|---|---|---|
| Routing rule | `natural_with_fallback` (2Fnat) | `natural_with_fallback` (2Fnat) — same |
| Trainable expert set | codealpaca-p02 (or metamath-p02): 146 experts, ~5.6/layer | exp_math: 182 experts, 7/layer |
| Helper expert set | c4-p02: ~243 experts, ~9.4/layer | exp_c4_p02: ~312 experts, ~12/layer — same source, slight assignment difference |
| Loaded experts (all layers) | ~389 | 494 (~27% more, minor delta) |
| Sequence length | **1024** | **4096** — 4× |
| Batch × grad-accum | 1 × 128 (effective 128) | 1 × 4 |
| Precision | **bf16 autocast, no GradScaler** | fp16-mixed autocast + GradScaler |
| Trainable param dtype | bf16 (native) | **fp32** (`freeze_parameters(upcast_trainable=True)`) |
| Grad dtype | bf16 | fp32 (required by `GradScaler.unscale_`) |
| Model params in VRAM (fp16/bf16) | backbone ~2 GB + ~389 experts × 17 MB ≈ **~8.7 GB** | backbone ~2 GB + 494 experts × 17 MB ≈ **~10.5 GB** |
| Trainable params | 146 × 8.65 M ≈ **1.26 B** | 182 × 8.65 M ≈ **1.57 B** |
| fp32 shadow of trainable params | — (bf16 native) | **~3 GB** (fp32 upcast) |
| Gradients | bf16 ~2.5 GB | fp32 **~6.2 GB** |
| fp32 AdamW state | ~10 GB | **~12.4 GB** (this is the alloc that OOM'd) |
| 8-bit AdamW state (bnb `optim_bits=8`) | not used | **~3.1 GB** |
| Activations (grad-ckpt on) | ~0.5–1 GB (seq 1024) | ~4–8 GB (seq 4096) |
| Model rebuild per inner-opt step | no | yes (before `enable_peer_resync=false` fix) — buys back ~470 MiB when disabled, but not the fit delta |
| **Total (fp32 AdamW)** | **~24 GB** ✅ | **~46–50 GB → OOM** ❌ (observed 47.36 / 47.4 GB) |
| **Total (AdamW8bit)** | (not needed) | **~37–41 GB** ✅ |

## Where the ~22–26 GB delta actually goes

Starting from the 2Fnat experiment's ~24 GB and getting to the miner's ~46–50 GB, holding the paradigm (2Fnat + c4 helpers) constant:

```
  24.0  2Fnat experiment baseline (bf16, seq 1024, fp32 AdamW, same helpers)
+  ~2   ~27% more loaded experts (miner's exp_math 7/layer vs experiment's ~5.6)
+  3.1  fp32 upcast shadow of trainable experts (freeze_parameters(upcast_trainable=True))
+  3.7  grads fp32 vs bf16 delta (1.57B × 2 bytes)
+  2.4  fp32 AdamW state grows with the trainable count (1.57B vs 1.26B)
+  3.0–7.0  activation delta from seq_len 4096 vs 1024 (attention scratch scales with seq², FFN with seq)
+  ~0.5  fragmentation / reserved-but-unallocated on the pre-fix reload path (empirically ~470 MiB)
= ~40 – 44 GB with fp32 AdamW … which is exactly where the OOM lands (47.36/47.4 observed)
```

**Empirical confirmation** (run 7, `MINER_ADAMW_OPTIM_BITS=32` + `enable_peer_resync=false`): fp32 AdamW `_init_group` gets `exp_avg` (352 MiB) through cleanly this time, but still OOMs on `exp_avg_sq` (176 MiB) with 31 MiB free + 549 MiB reserved-but-unallocated. Disabling the reload alone doesn't buy enough headroom — the precision + seq-len deltas dominate.

## The compounding effect

The three "precision + seq_len + reload" deltas each cost 3–7 GB — none of them individually push you over the ceiling. But they compound: halving `sequence_length` alone would save 3–8 GB of activations and let fp32 AdamW fit; switching to bf16 alone would save ~7 GB. Any *one* of those fixes would probably let you keep fp32 AdamW state; picking 8-bit AdamW is the least invasive because it doesn't change the training paradigm (miner code path, sequence length, or the 2Fnat helper set).

## What would let the miner match the 2Fnat experiment's headroom on fp32 AdamW

None of these are strictly needed for training to work (8-bit works), but any one of them would bring parity if you want fp32 state back:

1. **Move backbone to bf16, drop fp16-mixed + GradScaler + `upcast_trainable=True`.** bf16 has enough dynamic range that no shadow fp32 copy is needed, and grads can stay in bf16 during backward. This is what the 2Fnat experiment does. Saves ~7 GB (upcast shadow + fp32-grad delta).
2. **`sequence_length: 2048`** (or match the experiment's 1024). Saves 3–8 GB.
3. **Both of the above** → miner behaves like the experiment on VRAM and fp32 AdamW state is comfortably in-budget.

Doing *neither* is why the miner needs `MINER_ADAMW_OPTIM_BITS=8`.

## Fix summary in this branch

`connito/miner/train.py` now selects the optimizer via `MINER_ADAMW_OPTIM_BITS`:

- `"32"` (or unset): `torch.optim.AdamW`, fp32 state (~12.4 GB) — **OOMs on the current miner config on a 47 GB A6000**
- `"8"`: `bnb.optim.AdamW(optim_bits=8)`, 8-bit state (~3.1 GB) — **fits**

Combined with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the `gc.collect()` + `torch.cuda.empty_cache()` call added right before `inner_scaler.step(inner_optimizer)`, the miner survives AdamW's first `_init_group` allocation and the training loop runs to completion.

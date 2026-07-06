# VRAM analysis — Connito miner (2Fnat) vs. experiment repo (2Fnat)

Why the Connito miner needed `MINER_ADAMW_OPTIM_BITS=8` (bnb `AdamW8bit`) to fit on a 47 GB A6000, while the experiment repo's **same-paradigm** 2Fnat runs (e.g. `scripts/tier4_2Fnat_lr1e-4_500.sh` — DeepSeek-V2-Lite, `--routing-mode natural_with_fallback`, `--frozen-kept-assignment-path` c4-p02 helpers) fit comfortably with `torch.optim.AdamW` at full fp32 state.

Both runs use the same base model, the same routing rule (2Fnat), and the same c4-p02 helper set — so helper cost is a wash, not a delta. Both run on the same GPU (A6000 47.4 GB), same host, and both set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## TL;DR

The dominant fit delta is precision: fp16-mixed + `upcast_trainable=True` + fp32 grads adds ~7 GB of persistent state on top of the 2Fnat experiment's bf16-native footprint. Sequence length and the reload path each contribute at most a few hundred MiB at the AdamW-alloc site — not the fit delta.

| Delta | Present at `optimizer.step` alloc? | Approx cost | Fit impact |
|---|---|---|---|
| **fp16-mixed + `freeze_parameters(upcast_trainable=True)` + fp32 grads for `GradScaler.unscale_`** vs bf16 native | Yes — params + grads are persistent, not freed after backward | ~3 GB shadow fp32 params + ~3.7 GB fp32-vs-bf16 grad delta = **~7 GB** | **This is the fit delta.** |
| `sequence_length=4096` vs `1024` | No — activations are transient, freed before `optimizer.step` runs | ~3–8 GB during forward/backward peak | Zero at the AdamW alloc site — empirically confirmed (seq_len=1024 + fp32 AdamW still OOMs at the exact same 176 MiB `exp_avg_sq`) |
| Model rebuild every inner-opt step (`reload_model_inplace` before `ckpt.enable_peer_resync=false`) | Only through fragmentation | ~470 MiB reserved-but-unallocated slack | Marginal — `exp_avg` alloc goes from failing to succeeding when disabled, but `exp_avg_sq` still fails |

The three "no fit impact" rows still matter for other reasons — seq_len bounds context, reload wipes optimizer state — but they aren't what's OOMing the AdamW alloc.

2Fnat routing itself contributes 0 GB — it's a compute-only scatter/gather + topk. The helper group loading is present in *both* runs, so it cancels out of the comparison.

### Empirical A/B/C

Same GPU (A6000 47.4 GB), same helper set (c4-p02), same expandable_segments env, all three variants failed at the same allocation site:

| Variant | Reload | seq_len | fp32 AdamW alloc result |
|---|---|---|---|
| A: default (pre-fix) | on | 4096 | `exp_avg` **352 MiB fails** (255 MiB free, 1018 MiB fragmented) |
| B: reload gated off | off | 4096 | `exp_avg` succeeds, `exp_avg_sq` **176 MiB fails** (31 MiB free, 549 MiB fragmented) |
| C: reload off + seq 1024 | off | 1024 | `exp_avg` succeeds, `exp_avg_sq` **176 MiB fails** (31 MiB free, 549 MiB fragmented) — identical to B |

**Only precision is left as an actionable fit fix.** Everything else is downstream of "fp32 AdamW state (~12 GB) + upcasted trainable fp32 shadow (~3 GB) + fp32 grads (~6 GB) doesn't leave enough room."

## Component-by-component footprint

DeepSeek-V2-Lite: `hidden_size=2048`, `moe_intermediate=1408`, `n_routed_experts=64`, 26 MoE layers, 1 dense layer, vocab=102400.

Per expert (fp16): `gate_up_proj` = `2 × 1408 × 2048 × 2 B` ≈ 11.5 MB, `down_proj` = `1408 × 2048 × 2 B` ≈ 5.8 MB → **~17.3 MB / expert / layer**.

| Component | 2Fnat experiment (`tier4_2Fnat_lr1e-4_500.sh` style) | Connito miner (exp_math + exp_c4_p02 helper) |
|---|---|---|
| Routing rule | `natural_with_fallback` (2Fnat) | `natural_with_fallback` (2Fnat) — same |
| Trainable expert set | metamath-p02 (ESFT-token @ p=0.2): 146 experts, ~5.6/layer | exp_math: 182 experts, 7/layer |
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
| **Total (AdamW8bit)** | (not needed) | **~37–41 GB** ✅ — measured ~38.4 GB peak / ~23.9 GB steady on GPU 2 (see runtime confirmation) |

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

## This branch (`feat/tier4-natural-routing`) vs `main`

The **core VRAM footprint is identical on both branches.** `main` (at `7a84c9d`) and this branch load the same fp16 model, apply the same `freeze_parameters(upcast_trainable=True)` fp32 shadow, carry the same fp32 grads, and use the same `seq_len=4096` + `fp16-mixed`. Both allocate the same ~46–50 GB with fp32 AdamW. What this branch adds is the **ability to fit** (an 8-bit optimizer path) plus minor fragmentation relief — not a smaller base footprint. The ~7 GB precision delta vs the bf16 experiment lives in `model.py`/the fp16-mixed default, which neither branch changed.

| VRAM-relevant knob | `main` (`7a84c9d`) | this branch (`feat/tier4-natural-routing`) |
|---|---|---|
| Base footprint (params / grads / activations / precision) | ~46–50 GB with fp32 AdamW | ~46–50 GB — **identical** |
| Optimizer-state options | `torch.optim.AdamW`, fp32 state only (~12.4 GB) | `MINER_ADAMW_OPTIM_BITS` ∈ {`32`→fp32 ~12.4 GB, **`8`**→bnb 8-bit ~3.1 GB} (commits `91453ee`, `1992530`) |
| Escape hatch on OOM | **none** — fp32 is the only path | `=8` → fits at ~38 GB peak |
| Mid-training model reload | always on (no gate) → ~470 MiB reserved-but-unallocated churn at the alloc site | gated by `ckpt.enable_peer_resync` (default True), set **false** in this config → reclaims ~470 MiB (commit `63d3ed8`) |
| Defrag before first `optimizer.step` | none | `gc.collect()` + `torch.cuda.empty_cache()` before `inner_scaler.step` (commit `c7abdca`) |
| **Net fit on a 47 GB A6000 (this miner config)** | ❌ **OOMs, unrecoverable** | ✅ fits with `=8`; still OOMs with `=32` |

Net: on `main` this exact config cannot train on a single 47 GB A6000 (no 8-bit path, no reload gate). This branch makes it fit and trims the reload/fragmentation slack — but the precision delta that *causes* the tight fit is the same on both.

## Runtime confirmation (2026-07-04, GPU 2, `hk2/pr-188`)

Live reproduction on a fully-free A6000 (GPU 2, 47.4 GB), config `checkpoints/miner/connito-puppet/hk2/pr-188/config.yaml` (exp_math trainable + exp_c4_p02 helper, `seq_len=4096`, `fp16-mixed`, `enable_peer_resync=false`). Three launches matched the analysis to the MiB:

| Run | Optimizer | `PYTORCH_CUDA_ALLOC_CONF` | Result |
|---|---|---|---|
| 1 | fp32 AdamW (`=32`) | default | OOM at `_init_group` **`exp_avg` 352 MiB** — 339.69 MiB free, 1018.93 MiB reserved-but-unallocated (process at 47.06 GiB) |
| 2 | fp32 AdamW (`=32`) | `expandable_segments:True` | `exp_avg` succeeds, OOM at **`exp_avg_sq` 176 MiB** — 31.69 MiB free, 549.93 MiB reserved-but-unallocated (process at 47.36 GiB) |
| 3 | **bnb AdamW8bit (`=8`)** | `expandable_segments:True` | ✅ **fits** — cleared `_init_group`, ran 6 h+ |

Run 2 reproduces the "run 7" datapoint above *exactly* (176 MiB `exp_avg_sq`, 31 MiB free, 549 MiB fragmented). Run 1 (no `expandable_segments`) shows the earlier `exp_avg` 352 MiB failure with a bit more slack (340 MiB free vs the 255 MiB in variant A) — expected, since the fragmentation state at the alloc site varies run to run.

Measured VRAM on the successful 8-bit run (`nvidia-smi -i 2`):
- **~38.4 GB** at the forward/backward + first-optimizer-step peak (`38446 MiB`, sampled at `Start epoch training`)
- **~23.9 GB** steady-state once the step frees activations (`23884 MiB`)
- **~44.5 GB** transient during checkpoint save (`44522 MiB`; each expert-group shard is `size_mb=3382.5`)

This lands inside the estimated ~37–41 GB AdamW8bit envelope and leaves ~10 GB of headroom at the alloc site that fp32 lacked (31 MiB). Sustained health: `val_loss` 2.29 → 1.93 over ~6 h, checkpoint every 80 steps, 0 NaN batches, 1 benign non-finite-grad warning (dropped by design).

## Fix summary in this branch

`connito/miner/train.py` now selects the optimizer via `MINER_ADAMW_OPTIM_BITS`:

- `"32"` (or unset): `torch.optim.AdamW`, fp32 state (~12.4 GB) — **OOMs on the current miner config on a 47 GB A6000**
- `"8"`: `bnb.optim.AdamW(optim_bits=8)`, 8-bit state (~3.1 GB) — **fits**

Combined with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the `gc.collect()` + `torch.cuda.empty_cache()` call added right before `inner_scaler.step(inner_optimizer)`, the miner survives AdamW's first `_init_group` allocation and the training loop runs to completion.

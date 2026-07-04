# VRAM analysis — Connito miner vs. experiment repo

Why the Connito miner needed `MINER_ADAMW_OPTIM_BITS=8` (bnb `AdamW8bit`) to fit on a 47 GB A6000, while the experiment repo's runs of the same base model (DeepSeek-V2-Lite) fit comfortably with `torch.optim.AdamW` at full fp32 state.

Both runs are on the same GPU (A6000 47.4 GB), same host, same PyTorch, and both use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## TL;DR

Four independent knobs together push the miner ~30 GB above the experiment's footprint:

| # | Delta | Approx cost |
|---|---|---|
| 1 | Miner loads a **helper expert group** (2Fnat), experiment loads none | ~5–6 GB (~312 extra fp16 experts across 26 MoE layers) |
| 2 | Miner runs at **`sequence_length=4096`**, experiment at **`1024`** | ~3–8 GB activations (grad-ckpt on both, but attention scratch scales) |
| 3 | Miner upcasts trainable params to **fp32** and runs fp16 mixed with GradScaler; experiment keeps params in **bf16** natively | ~3 GB shadow fp32 copy of trainable params |
| 4 | Miner rebuilds the model + fresh AdamW state **on every inner-opt step** (reload_model_inplace, before `enable_peer_resync=false`), so the fp32 alloc happens against fragmented VRAM every time | Not a size delta but a fit delta — makes the same allocation OOM that would otherwise succeed |

None of these are 2Fnat's fault. 2Fnat routing is compute-only (a scatter/gather + topk), it adds zero VRAM to the forward path. The five-GB helper cost is the *helper group being loaded*, which is orthogonal to which routing rule you pick.

## Component-by-component footprint

DeepSeek-V2-Lite: `hidden_size=2048`, `moe_intermediate=1408`, `n_routed_experts=64`, 26 MoE layers, 1 dense layer, vocab=102400.

Per expert (fp16): `gate_up_proj` = `2 × 1408 × 2048 × 2 B` ≈ 11.5 MB, `down_proj` = `1408 × 2048 × 2 B` ≈ 5.8 MB → **~17.3 MB / expert / layer**.

### Experiment (metamath ESFT-token @ p=0.2, `--esft-classic`)

- Loaded experts: 146 across 26 layers (avg ~5.6/layer, from the assignment JSON)
- Backbone (attn + shared MLPs + embed + lm_head) bf16: ~2 GB
- **Model VRAM (bf16): ~4.5 GB**
- Trainable: 146 experts × ~8.65 M = ~1.26 B params
- Grads (bf16): ~2.5 GB
- fp32 AdamW state (`exp_avg` + `exp_avg_sq`): ~10 GB
- Activations at `seq=1024, batch=1, grad-ckpt`: ~0.5–1 GB
- **Total ≈ 17 GB** (nvidia-smi shows current usage ~6.7 GB — headroom includes CUDA workspace/reservations not yet touched)

### Connito miner (exp_math trainable + exp_c4_p02 helper)

- Loaded experts: 7 (exp_math) + 12 (exp_c4_p02) = **19 per layer** × 26 = **494 experts** — **3.4× more than experiment**
- Backbone (fp16): ~2 GB
- **Model VRAM (fp16): ~10.5 GB** (~8.5 GB experts + 2 GB backbone)
- Trainable params: 7 × 26 = 182 experts × ~8.65 M ≈ **1.57 B** (comparable to experiment)
- `freeze_parameters(upcast_trainable=True)` promotes these 1.57 B params to fp32 → **+3 GB shadow** on top of the fp16 copy in the model
- Grads (fp32 for unscale_): ~6.2 GB
- fp32 AdamW state: ~12.4 GB (this is the alloc that OOM'd)
- 8-bit AdamW state (bnb): ~3.1 GB (`optim_bits=8` uses int8 blockwise-quantized `exp_avg` + `exp_avg_sq` + a few bytes/block of quantization stats)
- Activations at `seq=4096, batch=1, grad-ckpt`: ~4–8 GB
- **Total with fp32 AdamW ≈ ~46–50 GB → OOM** (observed: 47.36 GB / 47.4 GB total)
- **Total with AdamW8bit ≈ ~37–41 GB → fits comfortably**

## Where the ~30 GB delta actually goes

Starting from the experiment's ~17 GB and getting to the miner's ~46 GB:

```
  17.0  experiment baseline
+  5.4  helper group loaded (exp_c4_p02, ~312 extra fp16 experts)
+  3.1  fp32 upcast shadow of trainable experts (freeze_parameters(upcast_trainable=True))
+  3.7  grads fp32 vs bf16 delta (1.57B × 2 bytes)
+  2.4  fp32 AdamW state grows with the trainable count
+  3.0–7.0  activation delta from seq_len 4096 vs 1024 (attention scratch scales with seq², FFN with seq)
+  ~2  fragmentation / reserved-but-unallocated on the reload path
= ~37 – 42 GB with 8-bit AdamW state (fits)
+  ~9  fp32 AdamW state instead of 8-bit
= ~46 – 51 GB (OOMs)
```

## The compounding effect

The four deltas each cost 3–6 GB — none of them individually push you over the ceiling. But they compound: dropping helpers alone would save 5.4 GB and let fp32 AdamW fit; dropping fp32 upcast alone would save 3 GB but grads still balloon; halving seq_length would save 3–8 GB of activations. Any *two* of these fixes would let you keep fp32 AdamW state; picking 8-bit AdamW is the least invasive because it doesn't change the training paradigm (miner code path, sequence length, or 2Fnat helpers).

## What would let the miner match experiment's headroom on fp32 AdamW

None of these are strictly needed for training to work (8-bit works), but they'd bring parity if you want fp32 state back:

1. **Move backbone to bf16, drop fp16-mixed + GradScaler + upcast_trainable=True.** bf16 has enough dynamic range that no shadow fp32 copy is needed, and grads can stay in bf16 during backward. Saves ~6 GB (upcast shadow + fp32-grad delta).
2. **`sequence_length: 2048`** (or match the experiment's 1024). Saves 3–8 GB.
3. **Turn off the helper group** (`task.helper_group_id: null`) — but this disables 2Fnat routing which is the whole point of PR 188.

If (1) and (2) both land, fp32 AdamW should fit again without changing the 2Fnat semantics.

## Fix summary in this branch

`connito/miner/train.py` now selects the optimizer via `MINER_ADAMW_OPTIM_BITS`:

- `"32"` (or unset): `torch.optim.AdamW`, fp32 state (~12.4 GB) — **OOMs on the current miner config on a 47 GB A6000**
- `"8"`: `bnb.optim.AdamW(optim_bits=8)`, 8-bit state (~3.1 GB) — **fits**

Combined with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the `gc.collect()` + `torch.cuda.empty_cache()` call added right before `inner_scaler.step(inner_optimizer)`, the miner survives AdamW's first `_init_group` allocation and the training loop runs to completion.

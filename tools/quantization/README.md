# fp8 and full-topology verification harness

Scripts that measure the fp8 toggle and the full-experts model on a real GPU
against the real model. They exist here rather than in `connito/test/` because
they need an L40S-class card, the 30 GB DeepSeek-V2-Lite checkpoint and the
production eval datasets — none of which belong in the unit suite. The CPU-only
invariants are covered by `connito/test/` instead.

Run from the repo root on a GPU box:

```bash
python tools/quantization/gpu_full_experts.py --arm fp16
```

`gpu_common.py` builds an offline config (no wallet, no chain) and pins a fixed
eval seed so both arms of every comparison draw identical batches. Absolute
losses therefore do not match a production round's `val_loss`; the paired
fp16-vs-fp8 delta is the number to read.

| script | question it answers |
|---|---|
| `gpu_full_experts.py` | Does the full-experts (`partial=False`) validator config build, quantize and evaluate? What does it cost in VRAM, and does the production copy-then-quantize path fit? |
| `gpu_full_fix_verify.py` | Does `get_base_model(partial=False)` load the real pretrained experts, and what does the build cost in host RAM? |
| `gpu_hf_stock_fp8.py` | On the stock HuggingFace model — the experiment repo's base — what is fp8's weight error and loss impact? |

`gpu_full_fix_verify.py` reads ground truth with `safe_open`, one tensor at a
time, instead of loading a second full `state_dict`. Comparing four tensors does
not justify a 30 GB allocation, and on a 62 GB host it is the difference between
a check that runs and one that competes with the thing it is checking.

## Results on record (L40S, 2026-08-17)

DeepSeek-V2-Lite, 15.71 B params, 1664 routed experts, seq 1024, production eval
mix (C4 + Nemotron-CC-Math), 9 batches.

**`gpu_hf_stock_fp8.py` — fp8 on correctly loaded weights.** Quantizing the
three projections of every routed expert, backbone untouched:

| | bf16 | fp8 | delta |
|---|---|---|---|
| val_loss | 1.449180 | 1.448177 | −0.001003 |
| resident VRAM | 29.36 GiB | 16.98 GiB | −12.38 |
| peak VRAM | 30.56 GiB | 17.76 GiB | −12.80 |

Relative weight error across all 4992 projections: mean 2.654%, min 2.645%,
max 2.660%. That reproduces the reference implementation's figure
(`partial_moe.py:quantize_expert_fp8`, experiment repo `17c878d`, "~2.7% on a
single projection"), which is what makes loss deltas transfer between the two
codebases. The sign of the loss delta is not meaningful at 9 batches.

**`gpu_full_fix_verify.py` — the full-model loader, after #210.** Every probe is
an exact match against the checkpoint shards:

```
expert gate_proj      MATCH  max|delta|=0
expert up_proj        MATCH  max|delta|=0
expert down_proj      MATCH  max|delta|=0
backbone (control)    MATCH  max|delta|=0
RESULT {"experts_loaded": true, "backbone_loaded": true, ...}
```

`loaded_counts={'full': 299, 'sliced': 3328}` — 3328 is 26 layers x 64 experts x
2 stacked params, i.e. every routed expert. The forward-pass loss printed
alongside is a loaded-vs-random discriminator only: the probe text is one
paragraph repeated, so its absolute value means nothing. The weight match is the
result; it is stronger evidence than a loss, which dtype, routing width or data
can all move.

**Host RAM during the full build: 59.6 GB -> 38.0 GB.** The full branch built the
model at torch's fp32 default and cast afterwards, so the fp32 storage for every
parameter stayed live until that parameter was individually replaced — a peak of
~2x the final size, within a couple of GB of exhausting a 62 GB host before the
gradient buffers or any eval copy existed. `mycelia.build_at_dtype` constructs at
the target dtype instead. Loaded weights and eval loss are bit-identical either
way (0.30255791544914246 both runs); only the peak moves.

**`gpu_full_experts.py` — full-experts memory.** fp16 29.33 GiB resident /
30.69 GiB peak; fp8-experts 15.96 / 17.31; conversion 0.8 s for 26 modules; zero
non-finite tensors; state_dict keys, shapes and dtypes unchanged. The production
foreground path OOMs: `resolve_foreground_eval_model` deepcopies the fp16 model
*before* quantizing, and dies at 43.69 of 44.39 GiB.

## Historical: how the full-model loader was broken (fixed in #210)

Kept because the diagnosis explains the shape of the fix, and because the
failure was silent — there was no error to search for.

`partial=False` left every routed expert at `_init_weights` random:

- 4992 unexpected keys, all expert (64 experts x 3 projections x 26 layers)
- 52 missing keys, all expert (26 layers x 2 stacked params)

The checkpoint names experts `...experts.{N}.gate_proj.weight`, while
`CustomDeepseekV2Experts` serialises the fused `...experts.{N}.gate_up_proj`. The
partial path translates between them in `_apply_pretrained_tensor_to_partial`;
the full path was a bare `load_state_dict(strict=False)`, which swallowed both
sides of the mismatch. The values were provably random rather than partially
loaded: all three tensors had mean magnitude 0.01595, and `initializer_range=0.02`
gives `E|x| = 0.02*sqrt(2/pi) = 0.01596`.

Same weights and batches through the stock loader scored **1.449**; through ours,
**9.956**.

`gpu_full_load_check.py` used to reproduce this by calling
`load_state_dict(strict=False)` directly. It was removed once the fix landed: it
exercised a code path the loader no longer takes, so running it would report a
mismatch that no longer exists. `gpu_full_fix_verify.py` replaces it and tests
the real entry point.

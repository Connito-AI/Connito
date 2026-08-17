# fp8 quantization verification harness

Scripts that measure the fp8 toggle on a real GPU against the real model. They
exist here rather than in `connito/test/` because they need an L40S-class card,
the 30 GB DeepSeek-V2-Lite checkpoint and the production eval datasets — none of
which belong in the unit suite. The CPU-only invariants are covered by
`connito/test/` instead.

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
| `gpu_full_load_check.py` | Does the full-model loader actually load the pretrained routed experts? |
| `gpu_hf_stock_fp8.py` | On the stock HuggingFace model — the experiment repo's base — what is fp8's weight error and loss impact? |

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

**`gpu_full_load_check.py` — the full-model loader is broken.** `partial=False`
leaves every routed expert at `_init_weights` random:

- 4992 unexpected keys, all expert (64 experts x 3 projections x 26 layers)
- 52 missing keys, all expert (26 layers x 2 stacked params)
- expert `gate_proj`/`up_proj`/`down_proj` vs checkpoint: mismatch
- backbone `kv_a_proj_with_mqa` vs checkpoint: exact match (the control)

Cause: the checkpoint names experts `...experts.{N}.gate_proj.weight`, while
`CustomDeepseekV2Experts` serialises the fused `...experts.{N}.gate_up_proj`.
The partial path translates between them in
`_apply_pretrained_tensor_to_partial`; the full path is a bare
`load_state_dict(strict=False)`, which swallows both sides of the mismatch. The
values are provably random rather than partially loaded: all three tensors have
mean magnitude 0.01595, and `initializer_range=0.02` gives
`E|x| = 0.02*sqrt(2/pi) = 0.01596`.

Same weights and batches through the stock loader score **1.449**; through ours,
**9.956**. A fixed full loader must score ~1.45 on these batches.

**`gpu_full_experts.py` — full-experts memory.** fp16 29.33 GiB resident /
30.69 GiB peak; fp8-experts 15.96 / 17.31; conversion 0.8 s for 26 modules; zero
non-finite tensors; state_dict keys, shapes and dtypes unchanged. The production
foreground path OOMs: `resolve_foreground_eval_model` deepcopies the fp16 model
*before* quantizing, and dies at 43.69 of 44.39 GiB. Host RAM peaks at 59.6 GiB
of 62 during the build, because the full branch materialises a CPU model and a
complete state_dict at once — the stock loader's `low_cpu_mem_usage=True` peaks
at 18.9 GiB for the same result.

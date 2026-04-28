# Why am I getting a zero score despite low local loss?

This is one of the most common issues miners encounter. Low training loss on your
machine does **not** guarantee a positive score from the validator. This guide
explains the gap and how to close it.

---

## How scoring works

The validator evaluates your submitted checkpoint with this formula:

```
score = max(0, baseline_loss − validator_eval_loss) ^ 1.2
```

`baseline_loss` is the validator's current global-model loss measured on its **own
seeded evaluation stream**. You only earn a positive score when your submission's
loss on that **same stream** is strictly lower than the baseline.

---

## Three reasons you can see low train loss but score zero

### 1. Training loss ≠ validator evaluation loss

Your local training loss is measured on batches your model has **already seen and
been optimised on**. The validator uses a completely independent, seeded data stream
it constructs at evaluation time. These two streams are not the same.

Experimentally (confirmed by `notebook/diagnose_eval_parity.py`):

| Measurement | Value |
|---|---|
| Miner local eval loss after 500 SGD steps | 11.964 |
| Validator eval loss for the same model | 11.964 |
| Baseline loss | 11.964 |
| **Score** | **0.0** |

Both losses are essentially equal — but because neither beats the baseline by a
meaningful margin, the score is zero.

### 2. The baseline updates every cycle

Every time the global model merges in good submissions it improves. The baseline you
are scored against is the current global model, not the one you started training
from. If the global model improved since your last run, your previously-competitive
submission may now fall above the new (higher) baseline.

---

## What a successful submission looks like

A submission that earns a positive score needs:

1. **Loss strictly below baseline** on the validator's seeded eval stream — not just
   locally.
2. **Enough training steps with AdamW** to push the expert weights meaningfully below
   the baseline (hundreds to low thousands of steps, depending on LR and data).
3. **A fresh cycle checkpoint** — a submission whose `global_ver` matches the
   current cycle's baseline `global_ver`. Stale commits (from a previous cycle's
   baseline) fail the signature check and never reach the evaluator.

---

## How to diagnose your own submission

Use `notebook/diagnose_eval_parity.py` to run the full pipeline offline:

```bash
# Step 1: train and save a checkpoint
./.venv/bin/python notebook/diagnose_eval_parity.py \
  --config <path/to/config.yaml> \
  --baseline-checkpoint <path/to/validator_baseline_checkpoint> \
  --seed <combined_validator_seed> \
  --run-offline-cycle \
  --train-steps 200 \
  --train-optimizer adamw \
  --use-lr-schedule \
  --grad-clip 1.0 \
  --max-eval-batches 10 \
  --save-trained-checkpoint /tmp/my-trained-ckpt \
  --output-json /tmp/cycle-result.json

# Step 2: inspect the result
python3 -c "
import json
d = json.load(open('/tmp/cycle-result.json'))['offline_cycle']
print('train loss final :', d['train_loss_final'])
print('validator eval   :', d['validator_eval_reloaded']['val_loss'])
print('baseline         :', d['validator_baseline_eval']['val_loss'])
print('score            :', d['discrepancy']['validator_score_from_reloaded'])
"
```

The JSON output includes:
- `train_loss_final` — what you see as your local training loss
- `validator_eval_reloaded.val_loss` — what the validator would measure on your checkpoint
- `validator_baseline_eval.val_loss` — the baseline you need to beat
- `discrepancy.validator_score_from_reloaded` — your predicted score

If this predicted score is zero, your checkpoint will score zero on-chain.

---

## Script fidelity vs production `train.py`

`notebook/diagnose_eval_parity.py` intentionally differs from `train.py` in two memory-constrained
ways so it can run alongside the loaded baseline on a single GPU:

| Setting | `train.py` | `diagnose_eval_parity.py` |
|---|---|---|
| `upcast_trainable` | `True` (fp32 master weights) | `False` (stays fp16) |
| AdamW `foreach` | default (`True`, batched ops) | `False` (scalar ops, lower peak VRAM) |
| Gradient accumulation | configurable | not implemented (1 step per batch) |

These differences reduce memory pressure but slightly reduce training quality.
For production mining, always follow `train.py` exactly.

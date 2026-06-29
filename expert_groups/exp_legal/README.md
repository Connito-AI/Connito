# exp_legal — legal expert group

Expert group that trains on
[joelniklaus/Multi_Legal_Pile](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile)
(English-only, full document-type mix) at 50/50 weight against
[allenai/c4](https://huggingface.co/datasets/allenai/c4) for general-text
grounding.

**Chain `group_id`: `2`** (parallel to `exp_math=0`, `exp_dummy=1`).

Plan and rationale: [docs/exp-legal-migration-plan.md](../../docs/exp-legal-migration-plan.md).

## ⚠️ Pre-launch follow-ups

This group ships with two **placeholder** artifacts that must be replaced
before declaring the migration complete.

### 1. `expert_assignment.json` — placeholder copied from `exp_math/`

The current file is a verbatim copy of `exp_math/expert_assignment.json`.
The MoE router routes legal text through whatever experts the math group
selected, which is wrong semantically: training will work, but the routing
will not be specialized for legal text.

Regenerate on a machine with the model loaded:

```bash
python expert_groups/build_expert_assignment.py \
  --task exp_legal \
  --output expert_groups/exp_legal/expert_assignment.json \
  --num-batches 50 \
  --experts-per-layer 8
```

Runtime: ~30 min on an A6000. Inspect the resulting per-layer expert
distribution before committing — if a layer collapses to <3 distinct experts,
re-run with `--num-batches 100`.

### 2. `eval_source_seeded_shard_pick: false` in `config.yaml`

Set to disable the validator-consensus seeded shard-pick path, which requires
each HF source to be registered in
`connito/shared/eval_shard_pick.py:_KNOWN_SOURCES`. MultiLegalPile isn't
registered yet.

To flip back to `true`:

1. Add `(joelniklaus/Multi_Legal_Pile, en_all)` to `_KNOWN_SOURCES` with
   a `_SourceShardPolicy` entry:
   - `row_count_source="constant"` (MultiLegalPile is `.jsonl.xz`, no
     parquet footer)
   - `verified_shard_rows` populated by an offline script that streams
     every shard and counts lines (~2-4 hours of wall time)
   - `safe_floor_rows` = smallest verified shard count
   - `min_headroom_rows` = 5000
2. Pin a known-good HF revision SHA (e.g., via `huggingface-cli api`)
3. Add a unit test that imports `eval_shard_pick` and asserts
   `pick_shard_for_source("joelniklaus/Multi_Legal_Pile", "en_all", ...)`
   returns a valid pick.
4. Flip this config field to `true` (or remove the line; `true` is the
   default).

Until then, validators on `exp_legal` use the legacy head-of-stream eval
path. Consensus determinism across validators is weaker than the
seeded-pick guarantee `exp_math` enjoys.

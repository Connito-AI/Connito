---
name: connito-miner-docs
description: Use when writing or editing files under `docs/miner-faq/` in this repo (architecture.md, scoring.md, eval-status.md, phases.md, miner-config.md, troubleshooting.md, faq.md). These docs are RAG-ingested by the Connito Discord bot at github.com/Connito-AI/DiscordBot and shown to miners on Bittensor subnet 102.
---

# Writing Connito Miner-Facing Docs

Files under `docs/miner-faq/` are the knowledge base for a Discord support bot
that answers questions from miners and operators on Bittensor subnet 102. The
bot RAG-ingests them, chunked on H2 headings, and feeds matching chunks to
Claude as context. **The structure of your doc directly shapes whether the
bot finds and uses it correctly.**

## Audience

A miner running a GPU box on subnet 102. They know Bittensor basics (UIDs,
validators, weights) but not this subnet's internals. They are NOT
contributors to this codebase. Plain English first; code-accurate second;
define jargon on first use.

## File responsibilities

| File | Covers |
|---|---|
| `architecture.md` | Cycle phases, validator/miner roles, expert-group sharding, foundation/global checkpoint flow, baseline model |
| `scoring.md` | val_loss → delta_loss → score → chain_weight_stake_weighted chain; no-winner fallback to UID 0; conditions where fields go null |
| `eval-status.md` | Every `EVAL_STATUS_CODES` value (0–11, 99): code condition + plain meaning + miner-actionable remediation |
| `phases.md` | Every phase name, default length, what miners/validators do during it, where the state machine lives in code |
| `miner-config.md` | Miner-side flags/env vars that matter, defaults, pitfalls |
| `troubleshooting.md` | Symptom → cause → fix table |
| `faq.md` | Top miner-Discord confusions, ~15 Q&A |

## RAG conventions — non-negotiable

- **H2 headings are chunk boundaries.** Each `## Heading` starts a new chunk;
  everything until the next `## ` is the body. Chunks are retrieved out of
  order — **make each H2 section self-contained**.
- **No "see above" / "as mentioned in faq.md".** Repeat the key context
  inline. A chunk may be retrieved without its neighbors.
- **One H2 section ≈ 150–600 words.** Too short = no context; too long =
  diluted relevance.
- **H3+ stay nested inside an H2** — they're fine and don't start new chunks.
  Reserve H1 for the file title (optional, since the bot prefixes chunks with
  the filename anyway).
- **Code references in `path/to/file.py:funcname` form.** Plain text, no
  rendering. e.g. `connito/validator/evaluator.py:build_submission_uid_weights`.

## Writing rules

- One claim per paragraph; skim-friendly.
- Cite code for every load-bearing claim. Future maintainers can verify.
- No marketing copy ("we are excited", roadmap teasers). Functional only.
- No time-sensitive content (release dates, "next week", "currently broken").
  Use Discord for that; these docs should stay valid for months.

## The six rules the bot enforces in its system prompt

These six disambiguations are baked into the bot's system prompt as fixed
responses to common miner confusion. **Your docs are the upstream source of
truth for them.** If you change underlying behavior, update the doc AND
notify the bot maintainer so the system prompt stays in sync:

1. **"Burning"** = the empty-G1 fallback to UID 0, not destruction. Triggered
   when no miner clears the 2-round recency gate.
2. **`val_loss` finite with `eval_status_label = "no_chain_commit"`** is
   legitimate — gauge isn't cleared at round rollover. `eval_status_label`
   is authoritative for the current round.
3. **`score_latest = null` with finite `val_loss`** is legitimate — two
   different Prometheus gauges with different cadences.
4. **`no_chain_commit` + `chain_weight_stake_weighted > 0`** means other
   validators ARE weighting them. Never say "you're not registered."
5. **`round.stats.pending`** monotonicity is unverified upstream. Use
   `scored`/`failed` for progress.
6. **Top-level `score` field** = caller is on v1. v2 uses
   `chain_weight_stake_weighted`.

## Flagging uncertainty

If the code doesn't match the claim you'd write — or you can't fully verify
it — insert a blockquote TODO rather than papering over the gap:

```
> TODO: `connito/validator/aggregator.py:234-237` resets history on hotkey
> change, contradicting the community claim that newly registered miners
> inherit history. Confirm with reproducible repro.
```

The bot surfaces TODOs to users as "this is unverified — please file a
reproducible issue" instead of confirming folklore.

## Don't

- Don't reorder existing H2 sections without notifying the bot maintainer.
  Chunk IDs (`<file>::<index>::<heading>`) shift when sections reorder, which
  churns the RAG index unnecessarily.
- Don't write a section whose H2 body is empty / just a single link. The
  chunker emits an empty chunk that pollutes retrieval results.
- Don't omit code citations on the assumption "readers can grep". Future
  readers won't have the same context you do.

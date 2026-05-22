"""Eval-pool characterization tool for the Connito SN102 validator.

The validator scores miners on a streaming HF dataset built by
`connito/shared/dataloader.py:get_tokenised_dataset` from
`config.task.exp.data.dataset_sources` (default: 50% allenai/c4 (en)
+ 50% nvidia/Nemotron-CC-Math-v1 (4plus)). The selection pipeline is:

    shuffle(seed, buffer=50000) -> skip(rng.randrange(0, 50000)) ->
    interleave(probabilities=weights, seed) ->
    filter(fractional_index_filter(seed, fraction=0.1)) ->
    split_dataset_by_node(world_size=10, rank=0) ->
    tokenize(max_length=4096, padding="max_length", truncation=True)

The `combined_seed` is sha256(sorted validator-committed seeds +
last-MinerCommit2 block_hash). Two implications for tuning training
data:

  * Miners cannot predict the seed in advance, so they cannot
    memorize a fixed subset.
  * Across seeds the reachable pool spans the *body* of each
    source's shards, not just the head — but the per-round window
    is still narrow (50 batches * batch_size positions of a 50k
    shuffle buffer + 50k skip per source).

This tool draws N samples from the *exact* same loader the validator
uses (same shuffle/skip/interleave/filter knobs, same seed
semantics) and writes a JSON report covering:

  * Token-length histogram (raw text and tokenized): mean/median/
    pX quantiles.
  * Source distribution: C4 vs Nemotron fractions.
  * Token frequency: top-K token IDs.
  * Top-20 frequent bi/trigrams of raw text words.
  * Unique vocabulary size in the sample.
  * Math-vs-prose classification via simple regex heuristic.
  * Per-source loss-difficulty proxy (token-distribution entropy).
  * Multi-seed comparison (run 3-5 seeds) to confirm pool
    stability across draws.

It is purely diagnostic — does not modify model state, training
config, or on-chain weights. Lets the team see what training data
distribution would lower `miner_val_loss`.

Run:

    python -m bench.eval_pool_explorer --seed <hex> --samples 1000 \\
        --out report.json [--dump-text 50] \\
        [--tokenizer deepseek-ai/DeepSeek-V2-Lite]

With `--seed` omitted, a random seed is drawn. With `--multi-seed N`
the tool runs N seeds and emits per-seed stats plus an aggregate
stability summary.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import secrets
import statistics
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


# Mirror production constants. Keep in sync with
# `connito/shared/config.py:DataCfg` defaults and
# `connito/validator/evaluator.py:EVAL_MAX_BATCHES`.
DEFAULT_EVAL_MAX_BATCHES = 50
DEFAULT_SHUFFLE_BUFFER = 50_000
DEFAULT_SKIP_MAX = 50_000
DEFAULT_VALI_FRACTION = 0.1
DEFAULT_WORLD_SIZE = 10
DEFAULT_RANK = 0
DEFAULT_SEQUENCE_LENGTH = 4096
DEFAULT_PER_DEVICE_BATCH_SIZE = 1
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V2-Lite"

# The built-in default source mix the validator falls back to when no
# `dataset_sources` are configured. Matches
# `connito/shared/dataloader.py:get_tokenised_dataset`'s fallback.
DEFAULT_SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {"path": "allenai/c4", "name": "en", "weight": 0.5, "text_column": "text", "label": "c4"},
    {
        "path": "nvidia/Nemotron-CC-Math-v1",
        "name": "4plus",
        "weight": 0.5,
        "text_column": "text",
        "label": "nemo",
    },
)

# Simple word tokenizer for n-gram extraction. We deliberately do not
# use the HF tokenizer here because n-grams over BPE pieces are hard
# to interpret; word-level n-grams give a quick "what does the
# validator's data talk about?" picture.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Math/symbol heuristic. Counts of these substrings per sample
# determine math-vs-prose. Tuned on a small Nemotron-CC-Math sample
# — TeX/LaTeX dollar-delimited math, common operators, fraction/sum/
# integral macros, and unicode math symbols all push the score up.
_MATH_TOKENS = (
    "$$",
    "\\frac",
    "\\sum",
    "\\int",
    "\\sqrt",
    "\\begin{equation",
    "\\begin{align",
    "\\mathbf",
    "\\mathcal",
)
_MATH_CHARS = set("=+-*/^<>≤≥≠±∑∫∂∞≈∝√∇")


@dataclass
class ExplorerConfig:
    """Knobs that drive the dataloader replication.

    Defaults mirror the validator's production config so the report
    reflects the data miners are actually scored on.
    """

    seed: str
    sources: list[dict[str, Any]] = field(default_factory=lambda: [dict(s) for s in DEFAULT_SOURCE_SPECS])
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER
    skip_max: int = DEFAULT_SKIP_MAX
    vali_fraction: float = DEFAULT_VALI_FRACTION
    world_size: int = DEFAULT_WORLD_SIZE
    rank: int = DEFAULT_RANK
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    samples: int = 1000
    max_batches: int | None = None  # if set, overrides samples
    per_device_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE
    tokenizer_name: str = DEFAULT_TOKENIZER

    def effective_samples(self) -> int:
        if self.max_batches is not None:
            return self.max_batches * self.per_device_batch_size
        return self.samples


def _h256_int(prefix: str, idx: str, seed: str) -> int:
    """Reference of `connito.shared.helper.h256_int` — copied here so
    the bench module remains importable without pulling the full
    helper chain. The pattern is fixed (prefix, str(idx), seed) for
    the eval-pool filter."""
    m = hashlib.sha256()
    for part in (prefix, idx, seed):
        m.update(part.encode("utf-8"))
        m.update(b"\x00")
    return int.from_bytes(m.digest(), "big")


def _keep_fn(_example: dict[str, Any], idx: int, seed: str, threshold: int) -> bool:
    """Reference of `connito.shared.dataloader._fractional_index_filter`."""
    return _h256_int("dataset_selection", str(idx), seed) <= threshold


def _ensure_text_and_src(example: dict[str, Any], text_column: str, label: str) -> dict[str, Any]:
    """Coerce per-source text into a stable shape with a source label.

    The production dataloader does the same coercion but DROPS the
    source label. We keep `_src` so the report can attribute each
    sample to its origin — that's the whole point of this tool.
    """
    return {"text": str(example.get(text_column, "")), "_src": label}


def _build_stream(cfg: ExplorerConfig) -> Iterator[dict[str, Any]]:
    """Replicate `get_tokenised_dataset` with `train=False`, but yield
    raw `{"text", "_src"}` dicts instead of tokenized batches.

    Keeping the stream un-tokenized here lets us decide tokenization
    once for the whole report (instead of paying the per-batch cost
    twice — collator + ours).
    """
    from datasets import Features, Value, interleave_datasets, load_dataset
    from datasets.distributed import split_dataset_by_node

    # Production code (`get_tokenised_dataset`) does
    # `int(str(seed)[:8], 16)` and would crash on a non-hex seed
    # passed in by a library caller. Mirror its tolerance: if the
    # first 8 chars aren't hex, hash the whole seed into a 64-hex
    # digest first. The CLI's `_resolve_seed` already produces a
    # 64-hex string, so this is a belt for direct library use.
    seed_hex = cfg.seed
    try:
        int_seed = int(seed_hex[:8], 16)
    except ValueError:
        seed_hex = hashlib.sha256(seed_hex.encode()).hexdigest()
        int_seed = int(seed_hex[:8], 16)

    sources = []
    weights = []
    for source in cfg.sources:
        ds = load_dataset(
            source["path"],
            name=source.get("name"),
            streaming=True,
            revision="main",
        )
        split_name = "validation" if "validation" in ds else "train"
        ds = ds[split_name]
        ds = ds.select_columns([source.get("text_column", "text")])
        feats = Features({"text": Value("string"), "_src": Value("string")})
        ds = ds.map(
            partial(
                _ensure_text_and_src,
                text_column=source.get("text_column", "text"),
                label=source.get("label", source["path"]),
            ),
            features=feats,
        )
        sources.append(ds)
        weights.append(float(source.get("weight", 1.0)))

    if cfg.shuffle_buffer > 0:
        sources = [ds.shuffle(seed=int_seed, buffer_size=cfg.shuffle_buffer) for ds in sources]

    if cfg.skip_max > 0:
        skip_rng = random.Random(int_seed)
        offsets = [skip_rng.randrange(0, cfg.skip_max) for _ in sources]
        sources = [ds.skip(off) for ds, off in zip(sources, offsets, strict=True)]

    if len(sources) == 1:
        stream = sources[0]
    else:
        total = sum(weights)
        probs = [w / total for w in weights]
        stream = interleave_datasets(sources, probabilities=probs, seed=int_seed)

    if 0.0 < cfg.vali_fraction < 1.0:
        max_int = 2**256 - 1
        threshold = int(max_int * cfg.vali_fraction)
        stream = stream.filter(
            # Use the normalized hex seed so the keep_fn matches the
            # production filter exactly even when the caller passed a
            # non-hex seed.
            partial(_keep_fn, seed=seed_hex, threshold=threshold),
            with_indices=True,
        )

    if cfg.world_size > 1:
        try:
            stream = split_dataset_by_node(stream, world_size=cfg.world_size, rank=cfg.rank)
        except Exception as e:
            logger.warning(
                "split_dataset_by_node failed; continuing unsharded",
                error=str(e),
            )

    return iter(stream)


def _quantiles(xs: list[int | float]) -> dict[str, float]:
    if not xs:
        return {
            "mean": 0.0, "median": 0.0,
            "p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "p99": 0.0,
            "max": 0.0, "min": 0.0,
        }
    s = sorted(xs)

    def _pct(p: float) -> float:
        if not s:
            return 0.0
        # Linear interpolation between order statistics. `len-1` so
        # p=100 maps to the last element; matches numpy.percentile's
        # default "linear" method.
        if len(s) == 1:
            return float(s[0])
        k = (len(s) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(s[int(k)])
        return float(s[f]) + (s[c] - s[f]) * (k - f)

    return {
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
        "p5": _pct(5),
        "p25": _pct(25),
        "p75": _pct(75),
        "p95": _pct(95),
        "p99": _pct(99),
        "max": float(max(xs)),
        "min": float(min(xs)),
    }


def _math_score(text: str) -> int:
    """Return a positive integer score for how 'math-heavy' a text is.

    Score is the sum of:
      * count of single `$` (TeX inline math delimiter)
      * count of `=`
      * count of `\\` (TeX command marker)
      * 3 * count of each token in `_MATH_TOKENS`
      * count of distinct chars in `_MATH_CHARS` seen anywhere

    A sample is classified as `math` if this exceeds the threshold
    `MATH_THRESHOLD` (10 — empirically separates Nemotron math
    samples from C4 prose). The `_MATH_TOKENS` carry weight 3 because
    each occurrence is a strong math signal compared to a single `=`.
    """
    score = text.count("$") + text.count("=") + text.count("\\")
    for tok in _MATH_TOKENS:
        score += 3 * text.count(tok)
    score += sum(1 for ch in _MATH_CHARS if ch in text)
    return score


MATH_THRESHOLD = 10


def _ngrams(words: list[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield tuple(words[i : i + n])


def _shannon_entropy(counts: collections.Counter) -> float:
    """Shannon entropy in bits of the empirical distribution."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _analyze_samples(
    samples: list[dict[str, Any]],
    tokenizer,
    *,
    top_k_tokens: int = 1000,
    top_k_ngrams: int = 20,
) -> dict[str, Any]:
    """Compute the report dict for a single (seed) draw.

    `samples` are raw {"text", "_src"} dicts. Tokenization happens
    here on the assembled list, not in the streaming loop, so we can
    detect zero-length samples cleanly.
    """
    per_source_token_counters: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    source_counts: collections.Counter = collections.Counter()
    overall_token_counter: collections.Counter = collections.Counter()
    raw_char_lengths: list[int] = []
    raw_word_lengths: list[int] = []
    tokenized_lengths: list[int] = []
    bigram_counter: collections.Counter = collections.Counter()
    trigram_counter: collections.Counter = collections.Counter()
    math_flags = 0
    math_scores: list[int] = []
    seq_len = getattr(tokenizer, "model_max_length", DEFAULT_SEQUENCE_LENGTH)
    if seq_len is None or seq_len > 10**6:
        # Some tokenizers report `model_max_length` as a sentinel
        # (1e30) when no truncation is recommended; clamp to the
        # validator's actual sequence_length.
        seq_len = DEFAULT_SEQUENCE_LENGTH

    for sample in samples:
        text = sample.get("text", "") or ""
        label = sample.get("_src", "unknown")
        source_counts[label] += 1
        raw_char_lengths.append(len(text))

        words = _WORD_RE.findall(text)
        raw_word_lengths.append(len(words))

        # Use the same truncation/padding semantics as the production
        # dataloader so token counts here correspond to what the
        # model actually sees during eval.
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            padding=False,
        )["input_ids"]
        tokenized_lengths.append(len(tokens))
        overall_token_counter.update(tokens)
        per_source_token_counters[label].update(tokens)

        for bg in _ngrams(words, 2):
            bigram_counter[bg] += 1
        for tg in _ngrams(words, 3):
            trigram_counter[tg] += 1

        score = _math_score(text)
        math_scores.append(score)
        if score >= MATH_THRESHOLD:
            math_flags += 1

    total = max(1, len(samples))

    source_dist = {
        label: {
            "count": cnt,
            "fraction": cnt / total,
        }
        for label, cnt in source_counts.items()
    }

    top_tokens = [
        {
            "token_id": int(tid),
            "count": int(cnt),
            "fraction": cnt / max(1, sum(overall_token_counter.values())),
            "decoded": tokenizer.decode([int(tid)], skip_special_tokens=False),
        }
        for tid, cnt in overall_token_counter.most_common(top_k_tokens)
    ]

    top_bigrams = [
        {"ngram": " ".join(bg), "count": int(cnt)}
        for bg, cnt in bigram_counter.most_common(top_k_ngrams)
    ]
    top_trigrams = [
        {"ngram": " ".join(tg), "count": int(cnt)}
        for tg, cnt in trigram_counter.most_common(top_k_ngrams)
    ]

    per_source_entropy = {
        label: {
            "shannon_bits": _shannon_entropy(cnt),
            "unique_tokens": len(cnt),
            "total_tokens": sum(cnt.values()),
        }
        for label, cnt in per_source_token_counters.items()
    }

    return {
        "samples": total,
        "raw_text": {
            "char_length": _quantiles(raw_char_lengths),
            "word_length": _quantiles(raw_word_lengths),
        },
        "tokenized": {
            "length": _quantiles(tokenized_lengths),
            "unique_vocab_size": len(overall_token_counter),
            "total_tokens": sum(overall_token_counter.values()),
        },
        "source_distribution": source_dist,
        "top_tokens": top_tokens,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,
        "math_classification": {
            "threshold": MATH_THRESHOLD,
            "math_flagged_count": math_flags,
            "math_flagged_fraction": math_flags / total,
            "math_score_quantiles": _quantiles(math_scores),
        },
        "per_source_token_distribution": per_source_entropy,
    }


def _draw_samples(cfg: ExplorerConfig, stream: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull `effective_samples()` examples off the stream and return
    raw {"text", "_src"} dicts. Skips zero-length texts (they are
    not scored — they just inflate the histogram with zeros and
    misrepresent the pool)."""
    needed = cfg.effective_samples()
    drawn: list[dict[str, Any]] = []
    for example in stream:
        text = example.get("text", "") or ""
        if not text:
            continue
        drawn.append({"text": text, "_src": example.get("_src", "unknown")})
        if len(drawn) >= needed:
            break
    return drawn


def explore(
    cfg: ExplorerConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a single-seed exploration. Builds the stream, draws
    samples, and computes a full report.

    Returns
    -------
    (report, samples)
        ``report`` is the analyzed JSON-serializable dict;
        ``samples`` is the raw ``{"text", "_src"}`` list drawn from
        the stream (kept so callers can dump them to disk without
        re-streaming).
    """
    stream = _build_stream(cfg)
    samples = _draw_samples(cfg, stream)
    tokenizer = _load_tokenizer(cfg.tokenizer_name)
    return _analyze_samples(samples, tokenizer), samples


def _stability_summary(per_seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Across-seeds variability of the most policy-relevant metrics.

    Lower variance here = the per-round pool is statistically stable,
    so the validator's effective eval surface is reproducible from
    one round to the next (modulo a small per-seed jitter). Higher
    variance suggests the per-round draw matters and the team needs
    to either (a) widen the training distribution to cover the
    union, or (b) sample more per round.
    """
    if not per_seed_reports:
        return {}

    def _pick(report: dict[str, Any], path: list[str]) -> float | None:
        cur: Any = report
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        if isinstance(cur, int | float):
            return float(cur)
        return None

    tracked = {
        "tokenized_length_mean": ["tokenized", "length", "mean"],
        "tokenized_length_p95": ["tokenized", "length", "p95"],
        "math_flagged_fraction": ["math_classification", "math_flagged_fraction"],
        "unique_vocab_size": ["tokenized", "unique_vocab_size"],
    }

    out: dict[str, Any] = {}
    for label, path in tracked.items():
        values = [v for v in (_pick(r, path) for r in per_seed_reports) if v is not None]
        if not values:
            continue
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        out[label] = {
            "values": values,
            "mean": mean,
            "stdev": stdev,
            "cv": (stdev / mean) if mean > 0 else 0.0,
        }

    # Source-distribution stability (fraction of samples per label).
    source_fracs: dict[str, list[float]] = collections.defaultdict(list)
    for report in per_seed_reports:
        for label, info in (report.get("source_distribution") or {}).items():
            f = info.get("fraction")
            if isinstance(f, int | float):
                source_fracs[label].append(float(f))
    out["source_distribution_per_seed"] = {
        label: {
            "values": vs,
            "mean": statistics.fmean(vs) if vs else 0.0,
            "stdev": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
        }
        for label, vs in source_fracs.items()
    }
    return out


def _dump_text_samples(samples: list[dict[str, Any]], path: Path, limit: int) -> None:
    """Write the first `limit` raw text samples for human inspection.

    Format: one sample per block, prefixed with `[N] [_src]` header
    line + a separator. Newlines inside samples are preserved so the
    output reflects what the tokenizer sees.
    """
    with path.open("w", encoding="utf-8") as fh:
        for i, sample in enumerate(samples[:limit]):
            label = sample.get("_src", "unknown")
            text = sample.get("text", "")
            fh.write(f"=== sample {i} src={label} ===\n")
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            fh.write("\n")


def _resolve_seed(arg: str | None) -> str:
    """Return a 64-hex-char seed. Accepts hex, decimal int, or
    `None` (random)."""
    if arg is None:
        return secrets.token_hex(32)
    s = arg.strip()
    if s.startswith("0x"):
        s = s[2:]
    # If it's pure decimal, hash it into a stable 256-bit seed so
    # tiny ints (1, 2, 42) still exercise different `int_seed[:8]`.
    try:
        as_int = int(s)
        return hashlib.sha256(str(as_int).encode()).hexdigest()
    except ValueError:
        pass
    # Hex-like string. Pad/truncate to 64 chars for compatibility
    # with the production seed format.
    if all(c in "0123456789abcdefABCDEF" for c in s):
        return (s + "0" * 64)[:64].lower()
    # Fallback: hash arbitrary text into a hex digest.
    return hashlib.sha256(s.encode()).hexdigest()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench.eval_pool_explorer",
        description=(
            "Characterize the validator's eval pool. Samples from the EXACT "
            "loader the validator uses (same shuffle, skip, interleave, "
            "fractional filter, sharding) and emits a JSON report of "
            "token-length / source / vocab / math vs prose statistics."
        ),
    )
    p.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Eval seed. Hex digest (any length, padded), decimal int, or "
             "arbitrary text (hashed). Default: random 256-bit.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to draw per seed. Overridden by --max-batches.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="If set, draw this many BATCHES (size = --per-device-batch-size). "
             "Useful for matching the validator's EVAL_MAX_BATCHES.",
    )
    p.add_argument(
        "--per-device-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_BATCH_SIZE,
        help="Validator default is 1.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path for the report. If omitted, prints to stdout.",
    )
    p.add_argument(
        "--dump-text",
        type=int,
        default=0,
        help="Also dump first N raw text samples to <out>.samples_dump.txt "
             "(or stdout-adjacent if --out missing).",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name. Default: {DEFAULT_TOKENIZER}.",
    )
    p.add_argument(
        "--multi-seed",
        type=int,
        default=1,
        help="Run N seeds and emit per-seed + aggregate stability. "
             "Set to >1 to compare draws across rounds.",
    )
    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER,
        help="Validator default 50000.",
    )
    p.add_argument(
        "--skip-max",
        type=int,
        default=DEFAULT_SKIP_MAX,
        help="Validator default 50000.",
    )
    p.add_argument(
        "--vali-fraction",
        type=float,
        default=DEFAULT_VALI_FRACTION,
        help="Validator default 0.1.",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=DEFAULT_WORLD_SIZE,
        help="Sharding world_size. Validator default 10.",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_RANK,
        help="Sharding rank within world_size. Default 0.",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Tokenizer max_length. Default 4096.",
    )
    return p


def _run_multi_seed(base_cfg: ExplorerConfig, n_seeds: int) -> dict[str, Any]:
    """Run `n_seeds` independent draws and return a combined report.

    The first draw uses the configured seed; subsequent draws derive
    deterministic child seeds by hashing the parent. This keeps the
    multi-seed comparison reproducible: rerun with the same base
    seed and you get the same children. Each child seed is a full
    256-bit hex digest, matching production seed shape.
    """
    seeds = [base_cfg.seed]
    for i in range(1, n_seeds):
        parent = seeds[-1]
        seeds.append(hashlib.sha256(f"{parent}:{i}".encode()).hexdigest())

    per_seed = []
    last_samples: list[dict[str, Any]] = []
    for s in seeds:
        cfg = ExplorerConfig(
            seed=s,
            sources=[dict(src) for src in base_cfg.sources],
            shuffle_buffer=base_cfg.shuffle_buffer,
            skip_max=base_cfg.skip_max,
            vali_fraction=base_cfg.vali_fraction,
            world_size=base_cfg.world_size,
            rank=base_cfg.rank,
            sequence_length=base_cfg.sequence_length,
            samples=base_cfg.samples,
            max_batches=base_cfg.max_batches,
            per_device_batch_size=base_cfg.per_device_batch_size,
            tokenizer_name=base_cfg.tokenizer_name,
        )
        logger.info("explorer: running seed", seed=s[:12], samples=cfg.effective_samples())
        report, samples = explore(cfg)
        report["seed"] = s
        per_seed.append(report)
        last_samples = samples

    return {
        "config": {
            "seeds": seeds,
            "sources": base_cfg.sources,
            "shuffle_buffer": base_cfg.shuffle_buffer,
            "skip_max": base_cfg.skip_max,
            "vali_fraction": base_cfg.vali_fraction,
            "world_size": base_cfg.world_size,
            "rank": base_cfg.rank,
            "sequence_length": base_cfg.sequence_length,
            "samples_per_seed": base_cfg.effective_samples(),
            "tokenizer": base_cfg.tokenizer_name,
        },
        "per_seed_reports": per_seed,
        "stability": _stability_summary(per_seed),
        "_last_samples": last_samples,  # popped before write
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    seed = _resolve_seed(args.seed)
    cfg = ExplorerConfig(
        seed=seed,
        shuffle_buffer=args.shuffle_buffer,
        skip_max=args.skip_max,
        vali_fraction=args.vali_fraction,
        world_size=args.world_size,
        rank=args.rank,
        sequence_length=args.sequence_length,
        samples=args.samples,
        max_batches=args.max_batches,
        per_device_batch_size=args.per_device_batch_size,
        tokenizer_name=args.tokenizer,
    )
    if args.multi_seed < 1:
        parser.error("--multi-seed must be >= 1")

    bundle = _run_multi_seed(cfg, args.multi_seed)
    last_samples = bundle.pop("_last_samples", [])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, default=_json_default)
        logger.info("explorer: wrote report", path=str(args.out))
        if args.dump_text > 0 and last_samples:
            dump_path = args.out.with_suffix(args.out.suffix + ".samples_dump.txt")
            _dump_text_samples(last_samples, dump_path, args.dump_text)
            logger.info("explorer: wrote samples dump", path=str(dump_path), n=args.dump_text)
    else:
        json.dump(bundle, sys.stdout, indent=2, default=_json_default)
        sys.stdout.write("\n")
        if args.dump_text > 0 and last_samples:
            dump_path = Path("samples_dump.txt")
            _dump_text_samples(last_samples, dump_path, args.dump_text)
            logger.info("explorer: wrote samples dump", path=str(dump_path), n=args.dump_text)
    return 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(main())

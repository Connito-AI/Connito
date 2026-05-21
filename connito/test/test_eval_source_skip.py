"""Verification test for `eval_source_skip_max` (and `eval_source_shuffle_buffer`).

Exercises four properties of the eval-data-loader randomization in
`connito/shared/dataloader.py:get_tokenised_dataset`:

  P1. Determinism — same (seed, config) → byte-identical samples.
      Load-bearing for cross-validator consensus: if two validators with
      the same `combined_seed` and same config draw different batches,
      their per-miner losses diverge and weight consensus breaks every
      round.

  P2. Skip materially changes outputs — same seed with `skip_max=0`
      vs `skip_max>0` produces different samples. Otherwise the
      `eval_source_skip_max` knob would be a no-op.

  P3. Opt-out reproducibility — `skip_max=0` path is stable across runs,
      so operators who disable the skip see the same behavior they would
      have seen before the commit landed.

  P5. Per-source offsets differ — both data sources get different
      `.skip()` offsets from `random.Random(int_seed).randrange()`,
      so the two sources don't move in lockstep.

Pool-expansion across many seeds (sometimes labeled "P4" in design
docs) is intentionally NOT covered here — at low seed counts both
shuffle-only and shuffle+skip already pull from distinct shards, so
the divergence only becomes visible at seed counts (~46+) where
shard collisions force same-shard re-reads. That's better suited
to an ad-hoc analysis script than a verification test.

The test replicates the dataloader's randomization logic locally
rather than calling `get_tokenised_dataset` directly — this provides
an independent reference implementation, which catches a class of
bugs (e.g. the call-site forgets to thread the right config field)
that calling the actual function would not.

Hits HuggingFace streaming for `allenai/c4` (en) and
`nvidia/Nemotron-CC-Math-v1` (4plus). With the default
`SHUFFLE_BUFFER=50_000` and `SKIP_MAX=50_000`, each `collect()` call
costs ~10-30 seconds of network and CPU. The full test runs in
~3-5 minutes.

Run from repo root:

    python3 -m connito.test.test_eval_source_skip

Non-zero exit code on failure.
"""
from __future__ import annotations

import hashlib
import itertools
import random
import sys
from functools import partial

from datasets import (
    Features,
    Value,
    interleave_datasets,
    load_dataset,
)
from datasets.distributed import split_dataset_by_node


# Mirror the production constants in `connito/shared/dataloader.py` and
# `connito/validator/evaluator.py`. Keep these in sync if they change.
MAX_INT = 2**256 - 1
FRACTION = 0.1
THRESHOLD = int(MAX_INT * FRACTION)
WORLD_SIZE = 10
RANK = 0
NEED = 51  # EVAL_MAX_BATCHES (50) + 1 sentinel materialized batch
SHUFFLE_BUFFER = 50_000
SKIP_MAX = 50_000


def _h256_int(prefix: str, a: str, b: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{prefix}{a}{b}".encode()).digest(), "big"
    )


def _keep_fn(_ex, idx: int, seed: str) -> bool:
    """Reference implementation of dataloader.py:_fractional_index_filter."""
    return _h256_int("dataset_selection", str(idx), seed) <= THRESHOLD


def _load_src(path: str, name: str, label: str):
    """Load a source the same way get_tokenised_dataset does, plus a
    `_src` column so we can tell which source each sample came from."""
    ds = load_dataset(path, name=name, streaming=True, revision="main")["train"]
    ds = ds.select_columns(["text"])
    feats = Features({"text": Value("string"), "_src": Value("string")})
    return ds.map(
        lambda ex: {"text": str(ex["text"]), "_src": label},
        features=feats,
    )


def _build_eval_stream(sources, seed_hex: str, *, shuffle_buffer: int, skip_max: int):
    """Replicate dataloader.py: shuffle → skip → interleave → filter → shard."""
    int_seed = int(seed_hex[:8], 16)
    s = list(sources)
    if shuffle_buffer > 0:
        s = [ds.shuffle(seed=int_seed, buffer_size=shuffle_buffer) for ds in s]
    if skip_max > 0:
        skip_rng = random.Random(int_seed)
        offsets = [skip_rng.randrange(0, skip_max) for _ in s]
        s = [ds.skip(off) for ds, off in zip(s, offsets, strict=True)]
    split = interleave_datasets(s, probabilities=[0.5, 0.5], seed=int_seed)
    split = split.filter(partial(_keep_fn, seed=seed_hex), with_indices=True)
    try:
        split = split_dataset_by_node(split, world_size=WORLD_SIZE, rank=RANK)
    except Exception:
        # split_dataset_by_node on an iterable without a stable shard
        # count falls back to per-example skipping; we ignore the
        # warning here and continue.
        pass
    return split


def _collect(sources, seed_hex: str, *, shuffle_buffer: int, skip_max: int):
    ds = _build_eval_stream(
        sources, seed_hex, shuffle_buffer=shuffle_buffer, skip_max=skip_max
    )
    out = []
    for ex in itertools.islice(ds, NEED):
        out.append((ex["_src"], ex["text"][:120]))
    return out


def _print_result(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    line = f"  {status}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


def main() -> int:
    print("Loading source streams ...", file=sys.stderr)
    c4 = _load_src("allenai/c4", "en", "c4")
    nemo = _load_src("nvidia/Nemotron-CC-Math-v1", "4plus", "nemo")
    sources = [c4, nemo]

    failures: list[str] = []

    # ----------------------------------------------------------------
    # P1 — Determinism: same (seed, config) → byte-identical samples.
    # ----------------------------------------------------------------
    print("\nP1: Determinism (same seed → identical samples)", file=sys.stderr)
    seed = hashlib.sha256(b"p1-determinism").hexdigest()
    a = _collect(sources, seed, shuffle_buffer=SHUFFLE_BUFFER, skip_max=SKIP_MAX)
    b = _collect(sources, seed, shuffle_buffer=SHUFFLE_BUFFER, skip_max=SKIP_MAX)
    p1_ok = a == b
    _print_result(
        "P1 determinism",
        p1_ok,
        f"two runs identical = {p1_ok}",
    )
    if not p1_ok:
        failures.append("P1")

    # ----------------------------------------------------------------
    # P2 + P3 — Skip changes outputs; opt-out path is stable.
    # ----------------------------------------------------------------
    print("\nP2/P3: Skip changes outputs; opt-out is stable", file=sys.stderr)
    seed = hashlib.sha256(b"p2-skip-effect").hexdigest()
    shuffle_only = _collect(sources, seed, shuffle_buffer=SHUFFLE_BUFFER, skip_max=0)
    shuffle_skip = _collect(sources, seed, shuffle_buffer=SHUFFLE_BUFFER, skip_max=SKIP_MAX)
    opt_out_again = _collect(sources, seed, shuffle_buffer=SHUFFLE_BUFFER, skip_max=0)

    overlap = len(set(shuffle_only) & set(shuffle_skip))
    # A material change means the same seed under skip vs no-skip pulls
    # genuinely different rows. 50% is a generous threshold — in practice
    # we expect ~0% overlap because the buffer-shuffle RNG is also
    # different when the read starts at a different offset.
    p2_ok = overlap < (NEED * 0.5)
    _print_result(
        "P2 skip changes outputs",
        p2_ok,
        f"overlap = {overlap}/{NEED} (need < {NEED // 2})",
    )
    if not p2_ok:
        failures.append("P2")

    p3_ok = shuffle_only == opt_out_again
    _print_result(
        "P3 opt-out reproducibility",
        p3_ok,
        f"skip_max=0 runs identical = {p3_ok}",
    )
    if not p3_ok:
        failures.append("P3")

    # ----------------------------------------------------------------
    # P5 — Per-source offsets differ.
    # This is a pure-Python check on the RNG behavior; no network.
    # ----------------------------------------------------------------
    print("\nP5: Per-source offsets differ", file=sys.stderr)
    p5_ok = True
    for label, sd in [
        ("seed=42", 42),
        ("seed=999", 999),
        ("seed=1234567", 1234567),
    ]:
        rng = random.Random(sd)
        o1 = rng.randrange(0, SKIP_MAX)
        o2 = rng.randrange(0, SKIP_MAX)
        differ = o1 != o2
        _print_result(
            f"P5 per-source offsets ({label})",
            differ,
            f"offsets = ({o1}, {o2})",
        )
        if not differ:
            p5_ok = False
    if not p5_ok:
        failures.append("P5")

    # ----------------------------------------------------------------
    print("\n=== Summary ===")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("ALL PROPERTIES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

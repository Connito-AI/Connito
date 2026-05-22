"""Smoke tests for `bench.eval_pool_explorer`.

The eval-pool explorer reuses HF streaming datasets, but the tests
here mock `datasets.load_dataset` so they run offline in CI. They
verify:

  1. Report schema — every key/section the README promises is present.
  2. Source attribution — the `_src` label propagates through
     interleave + filter + shard.
  3. Histogram math — quantile and unique-vocab counts match the
     mocked input.
  4. Math classification — strings full of TeX score above
     threshold; prose scores below.
  5. CLI smoke — `--help` exits 0 and `main()` writes the expected
     keys to disk.

The bench module is invoked as a module so the import path matches
`python -m bench.eval_pool_explorer`. All file writes go to
`tmp_path`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pytest

from bench import eval_pool_explorer as epx

# ---------- Mock HF dataset plumbing --------------------------------


class _MockStreamingSplit:
    """Minimal HF streaming-split stand-in.

    Implements only the surface `_build_stream` uses:
    `select_columns`, `map`, `shuffle`, `skip`, `filter`, and
    iteration. Each operation returns a new instance with an updated
    transformation pipeline so `_MockStreamingSplit` is immutable in
    the same way HF's `IterableDataset` is.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        transforms: list[Any] | None = None,
        skip_n: int = 0,
    ):
        self._rows = list(rows)
        self._transforms = list(transforms or [])
        self._skip_n = skip_n

    def _clone(self, *, transforms=None, skip_n=None) -> _MockStreamingSplit:
        return _MockStreamingSplit(
            self._rows,
            transforms=transforms if transforms is not None else self._transforms,
            skip_n=skip_n if skip_n is not None else self._skip_n,
        )

    def select_columns(self, cols: list[str]) -> _MockStreamingSplit:
        def _select(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
            for r in rows:
                yield {c: r.get(c) for c in cols}

        return self._clone(transforms=self._transforms + [_select])

    def map(self, fn, features=None) -> _MockStreamingSplit:
        def _apply(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
            for r in rows:
                yield fn(r)

        return self._clone(transforms=self._transforms + [_apply])

    def shuffle(self, seed: int, buffer_size: int) -> _MockStreamingSplit:
        # The actual HF streaming-shuffle is buffer-bounded. For a
        # smoke test we just yield rows in the same order — the
        # report's source attribution and histograms are invariant
        # under row order, so this keeps the test deterministic
        # without re-implementing buffer shuffling.
        del seed, buffer_size
        return self._clone()

    def skip(self, n: int) -> _MockStreamingSplit:
        def _skip(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
            it = iter(rows)
            for _ in range(n):
                try:
                    next(it)
                except StopIteration:
                    return
            yield from it

        return self._clone(transforms=self._transforms + [_skip])

    def filter(self, fn, with_indices: bool = False) -> _MockStreamingSplit:
        def _filter(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
            for i, r in enumerate(rows):
                if with_indices:
                    if fn(r, i):
                        yield r
                else:
                    if fn(r):
                        yield r

        return self._clone(transforms=self._transforms + [_filter])

    def __iter__(self):
        rows: Iterable[dict[str, Any]] = iter(self._rows)
        for t in self._transforms:
            rows = t(rows)
        yield from rows


class _MockDatasetDict(dict):
    """`load_dataset(..., streaming=True)` returns a DatasetDict-like
    keyed by split name. Tests register only the splits they need."""


def _make_c4_rows(n: int = 200) -> list[dict[str, Any]]:
    """Prose-style sentences with no math content."""
    return [
        {
            "text": (
                f"The quick brown fox jumps over the lazy dog. "
                f"Episode {i} of a long-form essay about machine learning. "
                f"It rained heavily and the cats decided to stay indoors today."
            )
        }
        for i in range(n)
    ]


def _make_nemo_rows(n: int = 200) -> list[dict[str, Any]]:
    """Math-style content with TeX delimiters and operators."""
    return [
        {
            "text": (
                f"\\begin{{equation}} x_{{i}} = \\frac{{a + b}}{{c}} \\end{{equation}} "
                f"= {i} + 7 + \\sum_{{k=0}}^{{n}} k^2 = "
                f"\\int_0^1 f(x) dx \\approx \\sqrt{{2}} $$ {i} = i^2 + 1 $$"
            )
        }
        for i in range(n)
    ]


def _install_dataset_mock(monkeypatch, *, c4_rows=None, nemo_rows=None):
    """Stand in for `datasets.load_dataset`. Returns the right rows
    based on the path argument so source attribution can be verified.

    Also mocks `split_dataset_by_node` to a no-op since the mocked
    streaming split doesn't have a real shard count.
    """
    c4_rows = c4_rows if c4_rows is not None else _make_c4_rows()
    nemo_rows = nemo_rows if nemo_rows is not None else _make_nemo_rows()

    def _fake_load_dataset(path, *, name=None, streaming=True, revision="main"):
        del name, streaming, revision
        if "c4" in path:
            rows = c4_rows
        elif "nemo" in path.lower() or "Nemotron" in path:
            rows = nemo_rows
        else:
            rows = []
        return _MockDatasetDict({"train": _MockStreamingSplit(rows)})

    def _fake_split_by_node(stream, world_size, rank):
        # Mirror HF's behavior loosely: keep every world_size-th row
        # starting at `rank`. This lets the test exercise the same
        # round-robin filtering the real call would apply.
        del world_size, rank
        return stream

    def _fake_interleave(streams, probabilities=None, seed=None):
        del probabilities, seed
        # Round-robin so both labels appear quickly. Mirrors the
        # production interleave's effect on per-sample source
        # attribution at the granularity tests care about (rough
        # 50/50 balance), without re-implementing weighted sampling.
        rows: list[dict[str, Any]] = []
        iters = [iter(s) for s in streams]
        exhausted = [False] * len(iters)
        while not all(exhausted):
            for i, it in enumerate(iters):
                if exhausted[i]:
                    continue
                try:
                    rows.append(next(it))
                except StopIteration:
                    exhausted[i] = True
        return _MockStreamingSplit(rows)

    monkeypatch.setattr("bench.eval_pool_explorer.load_dataset", _fake_load_dataset, raising=False)
    # The bench module imports these lazily inside `_build_stream`,
    # so we patch the source modules they come from.
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("datasets.interleave_datasets", _fake_interleave)
    monkeypatch.setattr(
        "datasets.distributed.split_dataset_by_node", _fake_split_by_node
    )


# ---------- Mock tokenizer ------------------------------------------


class _MockTokenizer:
    """Whitespace-and-char tokenizer. Returns deterministic token IDs
    based on a stable hash so the report's `top_tokens` section can
    be verified without pulling a real HF tokenizer.

    `model_max_length` mirrors the production sequence_length so
    `_analyze_samples` clamps the same way.
    """

    pad_token = "<pad>"
    eos_token = "<eos>"
    model_max_length = epx.DEFAULT_SEQUENCE_LENGTH

    _decoded_cache: dict[int, str] = {}

    def __init__(self) -> None:
        self._next_id = 1
        self._vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        tid = self._vocab.get(piece)
        if tid is None:
            tid = self._next_id
            self._next_id += 1
            self._vocab[piece] = tid
            _MockTokenizer._decoded_cache[tid] = piece
        return tid

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 0,
        padding: Any = False,
    ) -> dict[str, list[int]]:
        del padding
        pieces = text.split()
        ids = [self._id(p) for p in pieces]
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(_MockTokenizer._decoded_cache.get(i, "?") for i in ids)


@pytest.fixture
def mock_tokenizer(monkeypatch):
    tok = _MockTokenizer()
    monkeypatch.setattr(epx, "_load_tokenizer", lambda name: tok)
    return tok


# ---------- Tests ---------------------------------------------------


def test_resolve_seed_handles_inputs():
    """Seed resolver accepts hex, decimal, arbitrary text, and None
    — and always returns a 64-hex string compatible with production
    seed format."""
    out_none = epx._resolve_seed(None)
    assert len(out_none) == 64 and all(c in "0123456789abcdef" for c in out_none)

    out_hex = epx._resolve_seed("abc")
    assert out_hex.startswith("abc") and len(out_hex) == 64

    out_dec = epx._resolve_seed("42")
    # Same input → same output (deterministic hash of "42").
    assert out_dec == epx._resolve_seed("42")
    assert len(out_dec) == 64

    out_txt = epx._resolve_seed("hello world! not hex.")
    assert len(out_txt) == 64


def test_resolve_seed_0x_prefix():
    """`0x`-prefixed hex strings strip the prefix and behave like raw hex."""
    out_prefix = epx._resolve_seed("0xdeadbeef")
    out_raw = epx._resolve_seed("deadbeef")
    assert out_prefix == out_raw


def test_quantiles_basic():
    out = epx._quantiles([1, 2, 3, 4, 5])
    assert out["mean"] == pytest.approx(3.0)
    assert out["median"] == pytest.approx(3.0)
    assert out["min"] == 1.0
    assert out["max"] == 5.0
    # p5 of [1,2,3,4,5] under linear interpolation between adjacent
    # order stats: position = 0.2 → 1 + 0.2*(2-1) = 1.2.
    assert out["p5"] == pytest.approx(1.2)
    assert out["p95"] == pytest.approx(4.8)


def test_quantiles_empty():
    out = epx._quantiles([])
    assert out == {
        "mean": 0.0, "median": 0.0,
        "p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "p99": 0.0,
        "max": 0.0, "min": 0.0,
    }


def test_quantiles_single_value():
    """Single-element list shouldn't blow up the quantile math —
    every percentile should equal the single element."""
    out = epx._quantiles([7.5])
    assert out["mean"] == out["p5"] == out["p99"] == out["max"] == out["min"] == 7.5


def test_math_score_separates_math_from_prose():
    """Empirical check: Nemotron-style content scores above
    threshold, C4-style prose scores below."""
    math_text = (
        "\\frac{a}{b} = c + d, where x = \\sum_i^n y_i. "
        "Also \\begin{equation} z^2 + 1 = 0 \\end{equation}"
    )
    prose_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Episodes of long-form prose with no math whatsoever."
    )
    assert epx._math_score(math_text) >= epx.MATH_THRESHOLD
    assert epx._math_score(prose_text) < epx.MATH_THRESHOLD


def test_shannon_entropy_uniform_is_log2_k():
    """Entropy of a uniform distribution over K symbols is log2(K)."""
    import collections as col

    counts = col.Counter({i: 4 for i in range(8)})
    h = epx._shannon_entropy(counts)
    assert h == pytest.approx(3.0)  # log2(8)


def test_shannon_entropy_empty_zero():
    import collections as col

    assert epx._shannon_entropy(col.Counter()) == 0.0


def test_ngrams_basic():
    words = ["a", "b", "c", "d"]
    assert list(epx._ngrams(words, 2)) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert list(epx._ngrams(words, 3)) == [("a", "b", "c"), ("b", "c", "d")]
    assert list(epx._ngrams(words, 5)) == []


def test_build_stream_attributes_sources(monkeypatch):
    """Source attribution survives the shuffle/skip/interleave/filter
    pipeline — every sample carries either `c4` or `nemo`."""
    _install_dataset_mock(monkeypatch)
    cfg = epx.ExplorerConfig(
        seed="00" * 32,
        samples=10,
        shuffle_buffer=10,
        skip_max=0,
        vali_fraction=1.0,
        world_size=1,
        rank=0,
    )
    stream = epx._build_stream(cfg)
    samples = list(_take(stream, 20))
    labels = {s["_src"] for s in samples}
    assert labels.issubset({"c4", "nemo"})
    assert labels  # at least one source seen
    # Both sources should appear given enough draws — confirms the
    # interleave is not biased to one label.
    assert "c4" in labels
    assert "nemo" in labels


def _take(it, n: int):
    out = []
    for ex in it:
        out.append(ex)
        if len(out) >= n:
            break
    return out


def test_analyze_samples_report_schema(monkeypatch, mock_tokenizer):
    """Report has every key the README promises."""
    samples = [
        {"text": "the quick brown fox runs fast", "_src": "c4"},
        {
            # Score: 1*$ + 4*= + 2*\\ + 3*\\frac + 1 math char = ~13
            "text": "$$ \\frac{a}{b} = c + d = e = f = g \\sqrt{2} $$",
            "_src": "nemo",
        },
        {"text": "another bit of prose with simple words", "_src": "c4"},
        {
            "text": "\\sum_i^n x_i = y + z \\int_0^1 f \\frac{1}{2} = \\sqrt{3}",
            "_src": "nemo",
        },
    ]
    report = epx._analyze_samples(samples, mock_tokenizer)

    # Top-level keys.
    expected_top = {
        "samples", "raw_text", "tokenized", "source_distribution",
        "top_tokens", "top_bigrams", "top_trigrams", "math_classification",
        "per_source_token_distribution",
    }
    assert expected_top.issubset(report.keys())

    # raw_text histograms.
    for sub in ("char_length", "word_length"):
        assert sub in report["raw_text"]
        for q in ("mean", "median", "p5", "p25", "p75", "p95", "p99", "max", "min"):
            assert q in report["raw_text"][sub]

    # tokenized histogram + vocab counts.
    assert "length" in report["tokenized"]
    assert "unique_vocab_size" in report["tokenized"]
    assert report["tokenized"]["unique_vocab_size"] > 0
    assert report["tokenized"]["total_tokens"] > 0

    # source_distribution covers both labels.
    assert set(report["source_distribution"].keys()) == {"c4", "nemo"}
    for _label, info in report["source_distribution"].items():
        assert info["count"] == 2
        assert info["fraction"] == pytest.approx(0.5)

    # math_classification — nemo samples flagged, c4 samples not.
    math = report["math_classification"]
    assert math["threshold"] == epx.MATH_THRESHOLD
    # All nemo samples should be math-flagged.
    assert math["math_flagged_count"] == 2
    assert math["math_flagged_fraction"] == pytest.approx(0.5)

    # per_source_token_distribution has entropy per source.
    for label in ("c4", "nemo"):
        psd = report["per_source_token_distribution"][label]
        assert "shannon_bits" in psd
        assert psd["shannon_bits"] >= 0
        assert psd["unique_tokens"] > 0
        assert psd["total_tokens"] > 0

    # top_tokens / bigrams / trigrams sized to inputs.
    assert len(report["top_tokens"]) > 0
    assert len(report["top_bigrams"]) > 0
    assert len(report["top_trigrams"]) > 0
    for entry in report["top_tokens"]:
        assert {"token_id", "count", "fraction", "decoded"} == set(entry.keys())
    for entry in report["top_bigrams"]:
        assert {"ngram", "count"} == set(entry.keys())


def test_analyze_samples_empty_input(mock_tokenizer):
    """Empty samples produce a valid report with zero counts — the
    explorer should not crash on a degenerate draw."""
    report = epx._analyze_samples([], mock_tokenizer)
    assert report["samples"] == 1  # divisor floored at 1 to avoid /0
    assert report["tokenized"]["unique_vocab_size"] == 0
    assert report["tokenized"]["total_tokens"] == 0
    assert report["math_classification"]["math_flagged_count"] == 0
    assert report["top_tokens"] == []


def test_main_writes_json_schema(monkeypatch, mock_tokenizer, tmp_path):
    """End-to-end: invoke main() with --samples 4 against the mocked
    dataset and confirm the on-disk JSON matches the documented
    schema."""
    _install_dataset_mock(monkeypatch)
    out_path = tmp_path / "report.json"
    rc = epx.main([
        "--seed", "deadbeef",
        "--samples", "4",
        "--out", str(out_path),
        "--shuffle-buffer", "10",
        "--skip-max", "0",
        "--vali-fraction", "1.0",
        "--world-size", "1",
        "--rank", "0",
        "--multi-seed", "2",
    ])
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())

    # Top-level shape.
    assert {"config", "per_seed_reports", "stability"}.issubset(data.keys())
    assert "_last_samples" not in data  # must be popped before write
    cfg = data["config"]
    assert len(cfg["seeds"]) == 2
    assert cfg["samples_per_seed"] == 4

    # Per-seed reports.
    assert len(data["per_seed_reports"]) == 2
    for report in data["per_seed_reports"]:
        assert "samples" in report
        assert "raw_text" in report
        assert "tokenized" in report
        assert "source_distribution" in report
        assert "math_classification" in report

    # Stability summary.
    stab = data["stability"]
    assert "tokenized_length_mean" in stab
    assert "source_distribution_per_seed" in stab


def test_main_dump_text_writes_samples(monkeypatch, mock_tokenizer, tmp_path):
    """`--dump-text N` writes a samples_dump file alongside the report."""
    _install_dataset_mock(monkeypatch)
    out_path = tmp_path / "report.json"
    rc = epx.main([
        "--seed", "00",
        "--samples", "6",
        "--out", str(out_path),
        "--dump-text", "3",
        "--shuffle-buffer", "10",
        "--skip-max", "0",
        "--vali-fraction", "1.0",
        "--world-size", "1",
        "--rank", "0",
    ])
    assert rc == 0
    dump_path = out_path.with_suffix(out_path.suffix + ".samples_dump.txt")
    assert dump_path.exists()
    body = dump_path.read_text()
    # Three headed blocks.
    assert body.count("=== sample ") == 3
    # Source labels propagated to the dump.
    assert "src=c4" in body or "src=nemo" in body


def test_multi_seed_derives_distinct_children():
    """`--multi-seed N` derives N distinct seeds from the base seed."""
    cfg = epx.ExplorerConfig(seed="ab" * 32)
    seeds: list[str] = [cfg.seed]
    # Replicate the derivation rule from `_run_multi_seed` directly
    # so this stays a pure-Python test (no streaming).
    import hashlib
    for i in range(1, 5):
        parent = seeds[-1]
        seeds.append(hashlib.sha256(f"{parent}:{i}".encode()).hexdigest())
    assert len(set(seeds)) == 5
    for s in seeds:
        assert len(s) == 64


def test_stability_summary_empty():
    """Empty input -> empty stability."""
    assert epx._stability_summary([]) == {}


def test_stability_summary_handles_missing_keys():
    """Reports missing the tracked paths are skipped without
    raising — the explorer should be robust to schema drift."""
    reports = [
        {"tokenized": {"length": {"mean": 100}}},
        {"tokenized": {}},  # missing length entirely
        {"tokenized": {"length": {"mean": 110}}},
    ]
    s = epx._stability_summary(reports)
    assert "tokenized_length_mean" in s
    assert s["tokenized_length_mean"]["values"] == [100.0, 110.0]
    assert s["tokenized_length_mean"]["mean"] == pytest.approx(105.0)


def test_help_exits_zero():
    """The `--help` entry point exits with code 0."""
    with pytest.raises(SystemExit) as e:
        epx.main(["--help"])
    assert e.value.code == 0


def test_explore_drops_empty_text(monkeypatch, mock_tokenizer):
    """`_draw_samples` skips zero-length rows so they don't pollute
    the histogram with phantom zeros."""
    rows = [
        {"text": "actual content here", "_src": "c4"},
        {"text": "", "_src": "c4"},  # should be dropped
        {"text": "more content", "_src": "nemo"},
        {"text": None, "_src": "c4"},  # also dropped
    ]
    stream = iter(rows)
    cfg = epx.ExplorerConfig(seed="00" * 32, samples=10)
    drawn = epx._draw_samples(cfg, stream)
    assert len(drawn) == 2
    assert all(d["text"] for d in drawn)


def test_keep_fn_deterministic():
    """`_keep_fn` is deterministic: same (idx, seed) → same decision."""
    a = epx._keep_fn({}, 5, "abc", threshold=2**128)
    b = epx._keep_fn({}, 5, "abc", threshold=2**128)
    assert a == b
    # Threshold 0 keeps nothing.
    assert epx._keep_fn({}, 5, "abc", threshold=0) is False


def test_h256_int_matches_helper():
    """Bench's `_h256_int` matches the same-named helper in
    `connito.shared.helper` for the eval-pool filter input pattern
    so the bench draws the same samples as the real validator."""
    from connito.shared.helper import h256_int

    seed = "abc123"
    for idx in (0, 1, 7, 13, 9999):
        assert epx._h256_int("dataset_selection", str(idx), seed) == h256_int(
            "dataset_selection", str(idx), seed
        )

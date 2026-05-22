"""Unit tests for `connito.shared.packed_dataloader.PackedDataset`.

Validates that sequence packing:

  1. Emits exactly-`seq_length` chunks with no pad-token waste.
  2. Inserts the configured separator (EOS / BOS / none) between
     documents.
  3. Improves the useful-tokens-per-batch ratio over the legacy padded
     `tokenize_and_format` path — the entire reason this module exists
     under delta-driven validator scoring.
  4. Honors the opt-in safety contract: when `use_packing=False`, the
     dataloader path is byte-for-byte identical to today's behavior.
  5. Produces `attention_mask = [1] * seq_length` (no padding) for full
     chunks, so the model's forward doesn't waste compute on attention
     to pad tokens.

The tests use a hand-rolled mock tokenizer (no HF network or model
downloads) so they run in milliseconds inside the suite.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from connito.shared.packed_dataloader import (
    IGNORE_LABEL,
    PackedDataset,
    pack_iterable,
)


# ----------------------------------------------------------------------
# Mock tokenizer — minimal stand-in for PreTrainedTokenizerBase.
#
# Behavior:
#   * tokenize: whitespace split, then map each "word" to a token id by
#     `hash(word) % 30_000 + RESERVED_OFFSET`. Deterministic across runs
#     via PYTHONHASHSEED in the test runner — but for safety we use a
#     local hash so tests don't depend on interpreter flags.
#   * truncation / padding flags are honored to the extent
#     `pack_iterable` needs them (truncation=False, padding=False,
#     add_special_tokens=False).
#   * `padding="max_length"` is supported by the legacy callsite test.
# ----------------------------------------------------------------------
class MockTokenizer:
    """Deterministic tokenizer: token id = stable hash of the word."""

    def __init__(
        self,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

    @staticmethod
    def _stable_hash(word: str) -> int:
        # Local FNV-1a so we don't depend on Python's randomized hash().
        h = 2166136261
        for ch in word:
            h = (h ^ ord(ch)) & 0xFFFFFFFF
            h = (h * 16777619) & 0xFFFFFFFF
        # Reserve [0..9] for special tokens; map words into [10, 30009].
        return (h % 30_000) + 10

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        padding: bool | str = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        words = text.split()
        ids = [self._stable_hash(w) for w in words]
        if add_special_tokens:
            # Mirror real tokenizers that may prepend BOS by default.
            ids = [self.bos_token_id] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
        return {"input_ids": ids, "attention_mask": attention_mask}


def _make_samples(token_lengths: list[int]) -> list[dict[str, Any]]:
    """Build synthetic samples where text decodes to `n` real tokens.

    With `add_special_tokens=False` (which `pack_iterable` uses
    internally), the MockTokenizer returns exactly one token per
    whitespace-separated word — so a 50-token document is just 50
    distinct words joined by spaces.
    """
    out = []
    for i, n in enumerate(token_lengths):
        # Use deterministic distinct words per sample so we can verify
        # token identity if needed; `wN_K` keeps both global uniqueness
        # and per-sample distinctness.
        words = [f"w{i}_{k}" for k in range(n)]
        out.append({"text": " ".join(words)})
    return out


# ----------------------------------------------------------------------
# 1. Full chunks have exactly seq_length tokens, no padding.
# ----------------------------------------------------------------------
def test_packed_chunks_are_exact_seq_length():
    seq_length = 256
    tokenizer = MockTokenizer()
    samples = _make_samples([50, 100, 200, 150, 80, 120, 90, 60])

    ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="eos",
        drop_partial=True,
    )

    chunks = list(ds)
    assert len(chunks) >= 1, "Should emit at least one full chunk"
    for chunk in chunks:
        assert len(chunk["input_ids"]) == seq_length
        assert len(chunk["labels"]) == seq_length
        assert len(chunk["attention_mask"]) == seq_length
        # No padding inside a full chunk — pad_token_id (0) should never
        # appear since our mock token range is [10, 30009] ∪ {1 = eos}.
        assert tokenizer.pad_token_id not in chunk["input_ids"], (
            "Full packed chunks must contain no pad tokens"
        )


# ----------------------------------------------------------------------
# 2. EOS separator is inserted between documents.
# ----------------------------------------------------------------------
def test_eos_separator_inserted_between_documents():
    seq_length = 64
    tokenizer = MockTokenizer(eos_token_id=1)

    # Three short docs that all fit in one chunk together. After
    # packing with EOS sep, the buffer is:
    #   tok(doc1) + [eos] + tok(doc2) + [eos] + tok(doc3) + [eos]
    # which should contain >= 2 EOS tokens inside any emitted chunk.
    samples = _make_samples([15, 15, 15, 15, 15])

    chunks = list(
        pack_iterable(
            stream=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            sep_token_id=tokenizer.eos_token_id,
            drop_partial=True,
        )
    )
    assert chunks, "Need at least one chunk to validate sep insertion"
    eos_count = chunks[0]["input_ids"].count(tokenizer.eos_token_id)
    # Five 15-token docs + 5 eos = 80 tokens; 64-token chunk eats the
    # first ~4 docs plus separators = at least 3 EOS tokens.
    assert eos_count >= 2, (
        f"Expected >= 2 EOS separators in a packed chunk, got {eos_count}"
    )


def test_no_separator_when_concat_sep_is_none():
    seq_length = 64
    tokenizer = MockTokenizer()
    samples = _make_samples([20, 20, 20, 20])

    chunks = list(
        pack_iterable(
            stream=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            sep_token_id=None,
            drop_partial=True,
        )
    )
    assert chunks, "Need at least one chunk"
    # Without a separator, the EOS token id never appears (mock words
    # never hash to 1 since the hashing range starts at 10).
    assert tokenizer.eos_token_id not in chunks[0]["input_ids"]


def test_bos_separator_when_configured():
    seq_length = 64
    tokenizer = MockTokenizer(bos_token_id=2)
    samples = _make_samples([20, 20, 20, 20])

    ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="bos",
        drop_partial=True,
    )
    chunks = list(ds)
    assert chunks
    # Inserted between docs — should appear at least once in the chunk.
    assert tokenizer.bos_token_id in chunks[0]["input_ids"]


# ----------------------------------------------------------------------
# 3. Useful-tokens-per-batch ratio beats the padded baseline.
# ----------------------------------------------------------------------
def test_packing_beats_padding_on_useful_tokens_ratio():
    seq_length = 256
    tokenizer = MockTokenizer()

    # Mix of short docs that mimic real C4/Nemotron length distribution.
    lengths = [50, 100, 200, 150, 80, 120, 90, 60, 110, 70, 40, 95, 105]
    samples = _make_samples(lengths)

    # Packed path: count real tokens in emitted chunks. With
    # `pack_concat_sep="eos"`, every token in the chunk is "real"
    # because EOS separators are part of the LM training signal.
    packed_ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="eos",
        drop_partial=True,
    )
    packed_chunks = list(packed_ds)
    if not packed_chunks:
        pytest.skip("Not enough tokens for a single packed chunk")
    packed_useful = sum(
        sum(1 for m in chunk["attention_mask"] if m == 1)
        for chunk in packed_chunks
    )
    packed_total = len(packed_chunks) * seq_length
    packed_ratio = packed_useful / packed_total

    # Padded baseline: tokenize each sample with `padding="max_length"`,
    # mirroring `DefaultStreamingTorchDataset.tokenize_and_format`.
    padded_useful = 0
    padded_total = 0
    for sample in samples:
        out = tokenizer(
            sample["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
        )
        padded_useful += sum(1 for m in out["attention_mask"] if m == 1)
        padded_total += seq_length
    padded_ratio = padded_useful / padded_total

    # The whole point of packing.
    assert packed_ratio > padded_ratio, (
        f"Packed ratio ({packed_ratio:.3f}) must exceed padded ratio "
        f"({padded_ratio:.3f})"
    )
    # Packed is 1.0 (every token is real) for full chunks.
    assert packed_ratio == 1.0


# ----------------------------------------------------------------------
# 4. Attention mask is all 1s for full packed chunks.
# ----------------------------------------------------------------------
def test_attention_mask_all_ones_in_full_chunks():
    seq_length = 128
    tokenizer = MockTokenizer()
    samples = _make_samples([100, 100, 100, 100, 100])

    ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="eos",
        drop_partial=True,
    )
    for chunk in ds:
        assert all(m == 1 for m in chunk["attention_mask"]), (
            "Full chunks must have attention_mask == [1] * seq_length"
        )


# ----------------------------------------------------------------------
# 5. labels mirror input_ids in full chunks (no -100 sentinel anywhere).
# ----------------------------------------------------------------------
def test_labels_equal_input_ids_in_full_chunks():
    seq_length = 128
    tokenizer = MockTokenizer()
    samples = _make_samples([80, 90, 70, 100, 60])

    ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="eos",
        drop_partial=True,
    )
    chunks = list(ds)
    assert chunks
    for chunk in chunks:
        assert chunk["input_ids"] == chunk["labels"]
        assert IGNORE_LABEL not in chunk["labels"]


# ----------------------------------------------------------------------
# 6. Backwards-compat: `use_packing=False` (default) is unchanged.
#
# `get_tokenised_dataset` is the entrypoint; we don't exercise the full
# HF streaming path here — that's covered by `test_eval_source_skip.py`
# and the integration suite. Instead we verify the in-memory shape and
# call site logic, plus that DefaultStreamingTorchDataset alone (when
# `use_packing=False`) is unchanged.
# ----------------------------------------------------------------------
def test_default_padded_path_unchanged_with_packing_off():
    from connito.shared.dataloader import DefaultStreamingTorchDataset

    seq_length = 64
    tokenizer = MockTokenizer()
    samples = _make_samples([10, 12, 8])

    ds = DefaultStreamingTorchDataset(
        hf_iterable=samples, tokenizer=tokenizer, seq_length=seq_length
    )
    out = list(ds)
    assert len(out) == 3, "Padded path emits one chunk per input sample"
    for row in out:
        assert len(row["input_ids"]) == seq_length
        assert len(row["attention_mask"]) == seq_length
        # Padded path has pad tokens at the tail — the very inefficiency
        # we're optimizing away with packing.
        assert tokenizer.pad_token_id in row["input_ids"]


# ----------------------------------------------------------------------
# 7. Long documents are split across chunks, not truncated.
# ----------------------------------------------------------------------
def test_long_documents_split_across_chunks():
    seq_length = 64
    tokenizer = MockTokenizer()
    samples = _make_samples([200])  # single doc, ~3x seq_length

    ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="eos",
        drop_partial=True,
    )
    chunks = list(ds)
    # 200 tokens + 1 eos = 201 tokens → 3 full 64-token chunks, with
    # the remainder dropped.
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk["input_ids"]) == seq_length


# ----------------------------------------------------------------------
# 8. Partial-chunk padding path (`drop_partial=False`).
# ----------------------------------------------------------------------
def test_partial_chunk_padding_when_drop_partial_false():
    seq_length = 100
    tokenizer = MockTokenizer()
    # 50 tokens + 1 eos = 51, well below 100 → forces a partial tail.
    samples = _make_samples([50])

    chunks = list(
        pack_iterable(
            stream=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            sep_token_id=tokenizer.eos_token_id,
            drop_partial=False,
        )
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert len(chunk["input_ids"]) == seq_length
    # Pad tokens at the tail.
    assert chunk["input_ids"][-1] == tokenizer.pad_token_id
    # Labels at padded positions are the ignore sentinel.
    assert chunk["labels"][-1] == IGNORE_LABEL
    # Attention mask is 0 at padded positions, 1 on real tokens.
    assert chunk["attention_mask"][-1] == 0
    assert chunk["attention_mask"][0] == 1


def test_partial_chunk_dropped_when_drop_partial_true():
    seq_length = 100
    tokenizer = MockTokenizer()
    # 30 tokens + 1 eos = 31, far below 100, no full chunk possible.
    samples = _make_samples([30])

    chunks = list(
        pack_iterable(
            stream=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            sep_token_id=tokenizer.eos_token_id,
            drop_partial=True,
        )
    )
    assert chunks == [], "drop_partial=True must yield no partial chunks"


def test_partial_chunk_dropped_when_pad_token_missing():
    """Safety: drop_partial=False + tokenizer.pad_token_id=None must
    drop the partial chunk, NOT pad with eos_token_id as a fallback.

    The collator (`DataCollatorForLanguageModeling(mlm=False)`)
    regenerates labels from input_ids and only converts
    `labels==pad_token_id → -100`. If we padded the tail with eos as a
    fallback, those positions would silently train the model to
    predict EOS at every padding position — loss leakage on the
    partial-chunk tail. The correct behavior is to drop the partial.
    """
    seq_length = 100
    # Pad token is missing — eos still present.
    tokenizer = MockTokenizer(pad_token_id=None, eos_token_id=1)
    samples = _make_samples([30])

    chunks = list(
        pack_iterable(
            stream=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            sep_token_id=tokenizer.eos_token_id,
            drop_partial=False,
        )
    )
    assert chunks == [], (
        "drop_partial=False with no pad_token_id must drop the partial "
        "chunk; padding with eos as a fallback would leak loss on the tail."
    )


# ----------------------------------------------------------------------
# 9. Empty / blank samples are skipped without breaking the buffer.
# ----------------------------------------------------------------------
def test_empty_samples_are_skipped():
    seq_length = 64
    tokenizer = MockTokenizer()
    # Mix of empty, blank, and real samples — must not stall or yield
    # corrupted chunks.
    samples = [
        {"text": ""},
        {"text": " ".join(f"w{i}" for i in range(50))},
        {"text": ""},
        {"text": " ".join(f"x{i}" for i in range(50))},
    ]

    chunks = list(
        pack_iterable(
            stream=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            sep_token_id=tokenizer.eos_token_id,
            drop_partial=True,
        )
    )
    assert chunks, "Should yield at least one chunk from non-empty samples"
    for chunk in chunks:
        assert len(chunk["input_ids"]) == seq_length


# ----------------------------------------------------------------------
# 10. Invalid sep raises a clear error.
# ----------------------------------------------------------------------
def test_invalid_sep_raises():
    tokenizer = MockTokenizer()
    samples = _make_samples([20])

    with pytest.raises(ValueError, match="pack_concat_sep"):
        PackedDataset(
            hf_iterable=samples,
            tokenizer=tokenizer,
            seq_length=64,
            pack_concat_sep="bogus",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# 11. seq_length validation.
# ----------------------------------------------------------------------
def test_invalid_seq_length_raises():
    tokenizer = MockTokenizer()
    with pytest.raises(ValueError, match="seq_length"):
        list(
            pack_iterable(
                stream=_make_samples([20]),
                tokenizer=tokenizer,
                seq_length=0,
                sep_token_id=tokenizer.eos_token_id,
            )
        )


# ----------------------------------------------------------------------
# 12. PackedDataset is iterable multiple times when wrapping a list.
# (Streaming HF datasets are single-pass; for tests with a list we
# expect the dataset to be re-iterable, matching IterableDataset
# semantics on a re-iterable upstream.)
# ----------------------------------------------------------------------
def test_packed_dataset_reiterable_over_list_source():
    seq_length = 64
    tokenizer = MockTokenizer()
    samples = _make_samples([50, 50, 50, 50, 50])

    ds = PackedDataset(
        hf_iterable=samples,
        tokenizer=tokenizer,
        seq_length=seq_length,
        pack_concat_sep="eos",
        drop_partial=True,
    )
    first = list(ds)
    second = list(ds)
    assert first == second


# ----------------------------------------------------------------------
# 13. Smoke test: matches the manifest's first acceptance criterion.
# ----------------------------------------------------------------------
def test_smoke_chunk_shape_matches_manifest():
    """From the unit manifest: '256/sum_input_lengths is closer to 1.0
    than the padded version' — re-stated as a concrete equality check."""
    seq_length = 256
    tokenizer = MockTokenizer()
    samples = _make_samples([50, 100, 200])  # sum = 350 tokens

    chunks = list(itertools.islice(
        PackedDataset(
            hf_iterable=samples,
            tokenizer=tokenizer,
            seq_length=seq_length,
            pack_concat_sep="eos",
            drop_partial=True,
        ),
        10,
    ))
    # 50 + 1 + 100 + 1 + 200 + 1 = 353 tokens → 1 full 256-chunk, with
    # 97 left in the buffer (dropped).
    assert len(chunks) == 1
    chunk = chunks[0]
    assert len(chunk["input_ids"]) == seq_length
    assert all(m == 1 for m in chunk["attention_mask"])

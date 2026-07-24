"""Unit tests for the eval-data-quality gates added for exp_legal.

Covers the pure-function layer only — no HF network access:

  - `_min_text_chars_filter`: drops empty / trivially-short rows
    (Multi_Legal_Pile all_all streams 38% empty-text rows; measured
    2026-07-21 on the first 800 rows).
  - `_PrefixDedupFilter`: keeps only the first row per distinct text
    prefix (75% of non-empty M_L_P rows share an identical 200-char
    prefix with another row), and does so deterministically without
    relying on per-process `hash()`.
  - `tokenize_windowed`: content-hash window sampling for documents
    longer than `sequence_length`; prefix + padding for short ones.
  - `_SourceShardPolicy` `row_count_source="verified_table"`: table
    doubles as allowlist, validation rejects under-headroom shards and
    prefix typos, offset bound honors per-shard counts.

Integration behavior (streaming filter placement, interleave order,
seeded shard-pick equivalence for Multi_Legal_Pile) is exercised by the
notebook-driven checks described in docs/exp-legal-migration-plan.md §2
and the P1 determinism test in test_eval_source_skip.py.
"""
from __future__ import annotations

import pytest

from connito.shared.dataloader import (
    _min_text_chars_filter,
    _PrefixDedupFilter,
    tokenize_windowed,
)
from connito.shared.eval_shard_pick import (
    _SourceShardPolicy,
    _resolve_offset_bound,
    _validate_policy,
)


# ---------------------------------------------------------------------------
# _min_text_chars_filter
# ---------------------------------------------------------------------------

def test_min_chars_drops_empty_and_whitespace_rows():
    assert not _min_text_chars_filter({"text": ""}, min_chars=200)
    assert not _min_text_chars_filter({"text": "   \n\t  "}, min_chars=200)
    assert not _min_text_chars_filter({}, min_chars=200)


def test_min_chars_keeps_substantial_rows():
    assert _min_text_chars_filter({"text": "x" * 200}, min_chars=200)
    assert _min_text_chars_filter({"text": " " + "x" * 200 + " "}, min_chars=200)


def test_min_chars_boundary_is_inclusive():
    assert _min_text_chars_filter({"text": "x" * 200}, min_chars=200)
    assert not _min_text_chars_filter({"text": "x" * 199}, min_chars=200)


# ---------------------------------------------------------------------------
# _PrefixDedupFilter
# ---------------------------------------------------------------------------

def test_prefix_dedup_keeps_first_occurrence_only():
    f = _PrefixDedupFilter(prefix_chars=10)
    boiler = "IN THE SUPREME COURT OF ..."
    assert f({"text": boiler + " case one"})
    assert not f({"text": boiler + " case two"})
    assert f({"text": "completely different text"})


def test_prefix_dedup_distinguishes_beyond_prefix_window():
    f = _PrefixDedupFilter(prefix_chars=100)
    a = "shared start " + "a" * 200
    b = "shared start " + "b" * 200
    # Prefix window (100 chars) reaches into the differing region.
    assert f({"text": a})
    assert f({"text": b})


def test_prefix_dedup_state_is_per_instance():
    text = {"text": "same row"}
    assert _PrefixDedupFilter(50)(text)
    assert _PrefixDedupFilter(50)(text)  # fresh instance, fresh set


# ---------------------------------------------------------------------------
# tokenize_windowed
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """1-token-per-word tokenizer; ids are stable per word."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, truncation=False, add_special_tokens=True, **_kw):
        ids = [7 + (len(w) % 50) for w in text.split()]
        return {"input_ids": ids}


def test_windowed_short_doc_pads_to_length():
    tok = _StubTokenizer()
    out = tokenize_windowed("three word doc", tok, sequence_length=8)
    assert len(out["input_ids"]) == 8
    assert out["attention_mask"] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert out["input_ids"][3:] == [tok.pad_token_id] * 5


def test_windowed_long_doc_returns_full_window():
    tok = _StubTokenizer()
    text = " ".join(f"w{i}" for i in range(500))
    out = tokenize_windowed(text, tok, sequence_length=64)
    assert len(out["input_ids"]) == 64
    assert out["attention_mask"] == [1] * 64


def test_windowed_is_deterministic_per_content():
    tok = _StubTokenizer()
    text = " ".join(f"w{i}" for i in range(500))
    a = tokenize_windowed(text, tok, sequence_length=64)
    b = tokenize_windowed(text, tok, sequence_length=64)
    assert a == b


def test_windowed_start_varies_across_documents():
    class PositionTokenizer(_StubTokenizer):
        # ids encode stream position, so the returned window is
        # `range(start, start+seq)` and directly reveals the chosen start.
        def __call__(self, text, truncation=False, add_special_tokens=True, **_kw):
            return {"input_ids": list(range(len(text.split())))}

    tok = PositionTokenizer()
    full = [" ".join(f"d{k}w{i}" for i in range(500)) for k in range(20)]
    starts = {tokenize_windowed(t, tok, sequence_length=64)["input_ids"][0] for t in full}
    # Content-hash-derived starts over a 437-position span: 20 documents
    # should land on many distinct starts (expected ≈19.6 distinct).
    # A constant-start implementation collapses this to 1.
    assert len(starts) >= 10


def test_windowed_missing_pad_token_falls_back_to_eos():
    class NoPad(_StubTokenizer):
        pad_token_id = None

    out = tokenize_windowed("one two", NoPad(), sequence_length=4)
    assert out["input_ids"][2:] == [NoPad.eos_token_id] * 2


# ---------------------------------------------------------------------------
# verified_table shard policy
# ---------------------------------------------------------------------------

_KEY = ("example/repo", "subset")


def _table_policy(**overrides):
    kwargs = dict(
        path_prefix="data/",
        path_suffix=(".jsonl.xz",),
        revision="deadbeef",
        row_count_source="verified_table",
        min_headroom_rows=5_000,
        verified_shard_rows={
            "data/a/one.jsonl.xz": 50_000,
            "data/b/two.jsonl.xz": 12_000,
        },
    )
    kwargs.update(overrides)
    return _SourceShardPolicy(**kwargs)


def test_verified_table_policy_validates():
    _validate_policy(_KEY, _table_policy())  # should not raise


def test_verified_table_rejects_empty_table():
    with pytest.raises(ValueError, match="non-empty verified_shard_rows"):
        _validate_policy(_KEY, _table_policy(verified_shard_rows={}))


def test_verified_table_rejects_under_headroom_shard():
    with pytest.raises(ValueError, match="no safe offset"):
        _validate_policy(
            _KEY,
            _table_policy(
                verified_shard_rows={"data/a/one.jsonl.xz": 4_000},
            ),
        )


def test_verified_table_rejects_prefix_typo():
    with pytest.raises(ValueError, match="path_prefix/path_suffix"):
        _validate_policy(
            _KEY,
            _table_policy(
                verified_shard_rows={"wrong/one.jsonl.xz": 50_000},
            ),
        )


def test_verified_table_offset_bound_is_per_shard():
    policy = _table_policy()
    bound = _resolve_offset_bound(
        repo_id=_KEY[0],
        name=_KEY[1],
        revision="deadbeef",
        shard_path="data/b/two.jsonl.xz",
        policy=policy,
    )
    assert bound == 12_000 - 5_000

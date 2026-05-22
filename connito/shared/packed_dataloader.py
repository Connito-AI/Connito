"""Sequence-packing dataloader for causal LM training.

The default `tokenize_and_format` in `connito.shared.dataloader` pads every
sample to `sequence_length` with pad tokens. Pad tokens contribute zero
loss but still consume compute, which is wasteful on the C4 / Nemotron
mix where the median sample is far shorter than 4096 tokens.

Sequence packing concatenates tokenized documents back-to-back with an
EOS (or BOS) separator until the buffer reaches `sequence_length`, then
emits a fully-saturated chunk. Empirically this yields roughly 2-3x more
useful tokens per batch on real-world text, which translates directly
into faster `miner_val_loss` reduction under the delta-driven validator
scoring (`delta = max(0, baseline - val_loss)`).

Public API
----------
- `PackedDataset`: a `torch.utils.data.IterableDataset` that wraps an
  upstream iterable of `{"text": str}` samples and yields packed dicts
  `{"input_ids": [seq_length], "labels": [seq_length],
    "attention_mask": [1]*seq_length}` ready for a causal-LM collator.
- `pack_iterable(stream, tokenizer, seq_length, sep_token_id,
  drop_partial=True) -> Iterator[dict]`: lower-level generator used by
  `PackedDataset`; exposed so tests and downstream code can pack any
  iterable of text dicts without constructing a full Dataset.

Design notes
------------
- The packer tokenizes WITHOUT padding and WITHOUT truncation on the
  per-document call; truncation happens implicitly by the buffer slide
  at `sequence_length` boundaries. Documents longer than `sequence_length`
  are split across chunks (no per-document truncation loss).
- Document-boundary attention masking (so attention doesn't bleed
  across packed documents) is deliberately NOT implemented here. It
  requires custom `position_ids` plumbing through the model forward and
  was scoped out as a heavy follow-up; the loss impact of cross-doc
  attention on a pre-trained tokenizer with EOS separators is small in
  practice.
- The last partial chunk is dropped by default (`drop_partial=True`).
  Padding the tail would re-introduce the very pad-token waste that
  packing exists to remove. Set `drop_partial=False` to right-pad the
  final chunk with the tokenizer's pad token (`labels = -100` on the
  pad positions so they get zero loss contribution).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import PreTrainedTokenizerBase

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


# Standard HuggingFace label-ignore sentinel for cross-entropy loss.
# Padding positions in a final partial chunk get this label so the loss
# divisor stays honest.
IGNORE_LABEL = -100


def _resolve_sep_token_id(
    tokenizer: PreTrainedTokenizerBase, sep: str
) -> int | None:
    """Map the config string `"eos" | "bos" | "none"` to a token id.

    Returns None for `"none"` (no separator) or when the requested
    special token is missing from the tokenizer.
    """
    if sep == "none":
        return None
    if sep == "eos":
        return tokenizer.eos_token_id
    if sep == "bos":
        return tokenizer.bos_token_id
    raise ValueError(
        f"pack_concat_sep must be one of 'eos'|'bos'|'none', got {sep!r}"
    )


def pack_iterable(
    stream: Iterable[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    sep_token_id: int | None,
    drop_partial: bool = True,
    text_key: str = "text",
) -> Iterator[dict[str, list[int]]]:
    """Pack an iterable of text samples into fixed-length token chunks.

    Each upstream sample's text is tokenized (no padding, no truncation),
    optionally followed by `sep_token_id`, and appended to a rolling
    buffer. Whenever the buffer reaches `seq_length` tokens, one chunk
    is yielded and the consumed prefix is dropped. A long document that
    exceeds `seq_length` produces multiple consecutive chunks — its
    tokens are not truncated, just split across chunks.

    Parameters
    ----------
    stream :
        Iterable of dicts with a `text_key` entry.
    tokenizer :
        HuggingFace tokenizer used for per-document tokenization.
    seq_length :
        Target chunk length, in tokens.
    sep_token_id :
        Token id inserted between documents. None disables separator.
    drop_partial :
        If True (default), the final partial chunk is silently dropped.
        If False, it is right-padded with `tokenizer.pad_token_id` and
        the padded positions get `label = IGNORE_LABEL` plus
        `attention_mask = 0`.
    text_key :
        Key under which the document text lives on each sample.
    """
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")

    buffer: list[int] = []

    def _emit_chunk(tokens: list[int]) -> dict[str, list[int]]:
        # Full chunk path: every position is real, every label is real,
        # attention is unmasked everywhere. Mirrors the dict shape that
        # `DataCollatorForLanguageModeling(mlm=False)` consumes from the
        # padded path.
        return {
            "input_ids": list(tokens),
            "labels": list(tokens),
            "attention_mask": [1] * seq_length,
        }

    for sample in stream:
        text = sample.get(text_key, "")
        if not text:
            continue
        # `add_special_tokens=False` keeps us in full control of separator
        # placement; otherwise the tokenizer may inject its own BOS/EOS
        # and double the separator overhead.
        token_ids = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        if not token_ids:
            continue
        buffer.extend(token_ids)
        if sep_token_id is not None:
            buffer.append(sep_token_id)

        while len(buffer) >= seq_length:
            chunk = buffer[:seq_length]
            yield _emit_chunk(chunk)
            buffer = buffer[seq_length:]

    # Tail handling: drop or pad the final partial chunk.
    #
    # IMPORTANT correctness note about drop_partial=False:
    # `DataCollatorForLanguageModeling(mlm=False)` regenerates `labels`
    # from `input_ids` and then maps `labels[labels == pad_token_id] =
    # -100`. The IGNORE_LABEL values we put in `labels` here are
    # discarded by the collator; the only thing that keeps the padded
    # positions out of the loss is the `pad_id == tokenizer.pad_token_id`
    # equality at collator time. So we MUST use `tokenizer.pad_token_id`
    # for pad positions — not an EOS fallback, which would silently
    # train the model to predict EOS at every padding position.
    if buffer and not drop_partial:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            logger.warning(
                "pack_iterable: drop_partial=False but tokenizer has no "
                "pad_token_id; dropping the final partial chunk to avoid "
                "emitting one whose padded tail would be scored by the "
                "loss (DataCollatorForLanguageModeling can only mask "
                "pad_token_id, not an EOS fallback).",
                buffer_len=len(buffer),
            )
            return
        pad_len = seq_length - len(buffer)
        input_ids = list(buffer) + [pad_id] * pad_len
        # Labels at pad positions get IGNORE_LABEL for clarity. The
        # current collator overrides this field, but the `pad_id ==
        # tokenizer.pad_token_id` invariant above is what actually
        # keeps the tail out of the loss.
        labels = list(buffer) + [IGNORE_LABEL] * pad_len
        attention_mask = [1] * len(buffer) + [0] * pad_len
        yield {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class PackedDataset(TorchIterableDataset):
    """`IterableDataset` wrapper around `pack_iterable`.

    Designed as a drop-in replacement for
    `connito.shared.dataloader.DefaultStreamingTorchDataset` whenever
    `data.use_packing=True`. Both classes yield collator-ready dicts;
    PackedDataset trades padding tokens for tighter token utilization.

    Parameters
    ----------
    hf_iterable :
        Upstream iterable yielding `{"text": str}` samples — typically
        an HF streaming dataset split shared with the default loader.
    tokenizer :
        HuggingFace tokenizer used for per-document tokenization.
    seq_length :
        Length of each emitted chunk, in tokens.
    pack_concat_sep :
        `"eos" | "bos" | "none"`. The token inserted between documents.
        Defaults to `"eos"` because it matches the convention learned
        during pretraining for most causal LMs.
    drop_partial :
        If True (default), the trailing partial chunk is dropped.
        If False, it is right-padded (see `pack_iterable`).
    """

    def __init__(
        self,
        hf_iterable: Iterable[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        pack_concat_sep: str = "eos",
        drop_partial: bool = True,
    ):
        super().__init__()
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pack_concat_sep = pack_concat_sep
        self.drop_partial = drop_partial
        self._sep_token_id = _resolve_sep_token_id(tokenizer, pack_concat_sep)
        if pack_concat_sep != "none" and self._sep_token_id is None:
            logger.warning(
                "PackedDataset: requested separator is missing from "
                "tokenizer; packing without a separator.",
                pack_concat_sep=pack_concat_sep,
            )

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        yield from pack_iterable(
            stream=self.hf_iterable,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            sep_token_id=self._sep_token_id,
            drop_partial=self.drop_partial,
        )

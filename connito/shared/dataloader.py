from __future__ import annotations

import itertools
import os
import random
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

from datasets import load_dataset, interleave_datasets, Features, Value
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import DataCollator, DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from connito.shared.app_logging import structlog
from connito.shared.helper import h256_int, import_from_string

# Default per-request timeout for HuggingFace Hub network reads. Bg-eval's
# dataloader streams from HF, and a hung connection inside the streaming
# iterator can park a worker thread inside an uncancellable network read
# — observed as the trigger for the bg-eval lock-leak wedges in
# `notebooks/data/validator_a100_v0.1.38.log` (uid 82, 01:35:59) and
# `validator_A6000_v0.1.38.log` (uid 50, 23:32:43).
#
# 120 s rationale:
#   * Far above HF.co's healthy response time (<1 s per batch). A single
#     batch taking 120 s already implies HF is degraded, not just slow.
#   * Comfortably below the 300 s `per_miner_eval_timeout_sec` ceiling,
#     so a stalled fetch still trips this and unwinds via the dataloader
#     iterator long before `wait_for` cancels the awaiter.
#   * Aligned with `bg-download`'s 180 s file-fetch timeout — both speak
#     to "HF is unreachable" rather than "HF is slow."
#   * `setdefault` so operators can override with `HF_HUB_DOWNLOAD_TIMEOUT`
#     in the env without code changes (e.g. raise to 300 on a flaky
#     network, or lower for faster failure-detection).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

logger = structlog.get_logger(__name__)


def _fractional_index_filter(_example, idx: int, seed: str | int, threshold: int) -> bool:
    """Deterministically decide whether to keep a sample based on its streaming index."""
    score = h256_int("dataset_selection", str(idx), seed)
    return score <= threshold


def _min_text_chars_filter(example: dict[str, Any], min_chars: int) -> bool:
    """Drop rows whose text is empty or trivially short.

    Eval-path only. Empirically (Multi_Legal_Pile all_all, 2026-07-21)
    38% of streamed rows carry a completely empty `text`; those rows
    tokenize to all-padding sequences whose labels are fully masked, so
    they produce NaN eval loss and silently drop out of the scored-batch
    divisor — shrinking an already tiny eval sample. The distribution is
    bimodal (0 chars or ≥200), so a low threshold removes exactly the
    degenerate rows without biasing content selection.
    """
    return len(str(example.get("text", "")).strip()) >= min_chars


class _PrefixDedupFilter:
    """Keep only the first row for each distinct text prefix.

    Eval-path only, and only sound single-worker: the `seen` set lives in
    this instance, so the eval dataloader must iterate the stream in one
    process (the eval loader runs `num_workers<=1`; see `get_dataloader`).

    Rationale: templated corpora repeat their opening boilerplate across
    documents — measured 75% of non-empty Multi_Legal_Pile rows sharing an
    identical 200-char prefix. Scoring many near-identical rows lets a
    miner fine-tuned on the template reach near-zero loss on "unseen"
    documents. Deduplicating by prefix keeps one representative per
    template instead of a batch full of copies. Deterministic given a
    deterministic input stream (same seed → same order → same survivors).
    """

    def __init__(self, prefix_chars: int):
        self.prefix_chars = int(prefix_chars)
        # Exact prefixes, not `hash()` digests: builtin str hashing is
        # per-process randomized, so two validators could disagree on a
        # collision. The eval stream retains only ~thousands of rows, so
        # exact storage is a few hundred KB at worst.
        self.seen: set[str] = set()

    def __call__(self, example: dict[str, Any]) -> bool:
        prefix = str(example.get("text", ""))[: self.prefix_chars]
        if prefix in self.seen:
            return False
        self.seen.add(prefix)
        return True


def tokenize_windowed(
    text: str, tokenizer: PreTrainedTokenizerBase, sequence_length: int
) -> dict[str, list]:
    """Tokenize `text`, sampling a deterministic window from long documents.

    The previous behavior (`truncation=True`) always scored/trained on a
    document's FIRST `sequence_length` tokens. For templated corpora
    (legal filings, papers) the prefix is the most boilerplate-heavy,
    most predictable region of the document — document bodies never
    entered the pipeline at all. This helper instead:

    - short documents (≤ sequence_length tokens): pad to length, as before;
    - long documents: take a `sequence_length` window whose start is
      derived from the text's own content hash — deterministic for every
      validator (consensus-safe: no RNG, no config), uniform-ish across
      the document, and not influenceable by the validator.

    The raw text is capped at `sequence_length * 40` chars before
    tokenizing to bound tokenizer cost on pathological documents (real
    corpora run 3–8 chars/token, so the cap only bites degenerate input).
    """
    capped = str(text)[: sequence_length * 40]
    ids = tokenizer(capped, truncation=False, add_special_tokens=True)["input_ids"]
    if len(ids) > sequence_length:
        span = len(ids) - sequence_length
        start = h256_int("token_window", capped) % (span + 1)
        window = ids[start : start + sequence_length]
        return {"input_ids": window, "attention_mask": [1] * sequence_length}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    n = len(ids)
    return {
        "input_ids": ids + [pad_id] * (sequence_length - n),
        "attention_mask": [1] * n + [0] * (sequence_length - n),
    }


# -----------------------------
# Dataset
# -----------------------------
class DefaultStreamingTorchDataset(TorchIterableDataset):
    """
    Thin adapter to wrap a Hugging Face streaming (Iterable) dataset so it yields
    tokenized dicts ready for a collator.

    This is useful when you want to keep the tokenization logic explicit and
    avoid relying on `IterableDataset.map(...)` behaviors.
    """

    def __init__(self, hf_iterable, tokenizer: PreTrainedTokenizerBase, seq_length: int):
        """
        Parameters
        ----------
        hf_iterable :
            A split of an HF streaming dataset, e.g. ds["train"] with streaming=True.
        tokenizer : PreTrainedTokenizerBase
            HF tokenizer to use for tokenization.
        seq_length : int
            Max sequence length for truncation/padding.
        """
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __iter__(self):
        format_example = partial(self.tokenize_and_format, tokenizer=self.tokenizer, sequence_length=self.seq_length)

        # Explicit per-example iteration avoids surprises with HF's streaming `map` api (which
        # can leave original string columns attached when `column_names` is missing), ensuring
        # we only yield the tokenized dict expected by the collator.
        for example in self.hf_iterable:
            yield format_example(example)

    @staticmethod
    def tokenize_and_format(
        example: dict[str, str], tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> dict[str, list]:
        text = example.get("text", "")
        return tokenize_windowed(text, tokenizer, sequence_length)

    @classmethod
    def get_tokenised_dataset(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        rank: int | None = None,
        world_size: int | None = None,
        train: bool = True,
        seed: str | int | None = None,
        fraction: float | None = None,
    ):
        split_name = "train" if train else "validation"

        def _load_streaming_split(
            ds_name: str,
            ds_config: str | None = None,
            trust_remote_code: bool = False,
        ):
            """Helper to load a dataset split safely, falling back to 'train' if 'validation' is missing."""
            try:
                load_kwargs: dict[str, Any] = {"streaming": True, "revision": "main"}
                if ds_config is not None:
                    load_kwargs["name"] = ds_config
                if trust_remote_code:
                    # Authorize HF to execute the dataset repo's custom
                    # builder script. Opt-in per source via
                    # DatasetSourceCfg.trust_remote_code; never on by
                    # default. See config.DatasetSourceCfg for the
                    # rationale.
                    load_kwargs["trust_remote_code"] = True

                ds = load_dataset(ds_name, **load_kwargs)
                if split_name in ds:
                    return ds[split_name]
                else:
                    logger.warning(
                        f"Split '{split_name}' not found for {ds_name}. Falling back to 'train' split."
                    )
                    return ds["train"]
            except Exception as e:
                logger.error(f"Failed to load dataset {ds_name}: {e}")
                raise

        configured_sources = getattr(config.task.exp.data, "dataset_sources", None)
        legacy_dataset_name = getattr(config.task.exp.data, "dataset_name", None)
        legacy_data_dir = getattr(config.task.exp.data, "data_dir", None)

        if configured_sources:
            source_specs = configured_sources
            if not source_specs:
                logger.warning("No dataset sources found in config")
            else:
                logger.debug(
                    "Loading dataset sources from config",
                    sources=[
                        {
                            "path": src.path,
                            "name": src.name,
                            "weight": src.weight,
                            "text_column": src.text_column,
                        }
                        for src in source_specs
                    ],
                )
        elif legacy_dataset_name:
            source_specs = [
                {
                    "path": legacy_dataset_name,
                    "name": legacy_data_dir,
                    "weight": 1.0,
                    "text_column": "text",
                }
            ]
            logger.info(
                "No data.dataset_sources configured. Falling back to legacy data.dataset_name/data_dir.",
                dataset_name=legacy_dataset_name,
                data_dir=legacy_data_dir,
            )
        else:
            source_specs = [
                {
                    "path": "allenai/c4",
                    "name": "en",
                    "weight": 0.5,
                    "text_column": "text",
                },
                {
                    "path": "nvidia/Nemotron-CC-Math-v1",
                    "name": "4plus",
                    "weight": 0.5,
                    "text_column": "text",
                },
            ]
            logger.warning(
                "No dataset_sources or dataset_name configured. Using built-in default mix (C4 + Nemotron)."
            )

        def _source_value(source: Any, key: str, default: Any = None) -> Any:
            if isinstance(source, dict):
                return source.get(key, default)
            return getattr(source, key, default)

        # Force all source text columns to be a standard 'string' under the common key 'text'
        common_features = Features({"text": Value("string")})

        def ensure_string(example: dict[str, Any], source_text_column: str):
            return {"text": str(example[source_text_column])}

        # Convert string seed to integer. Reused as the
        # shard-pick / in-shard offset hash input AND as the
        # `interleave_datasets(seed=...)` argument, so the value must
        # be available before the per-source load loop.
        int_seed = int(str(seed)[:8], 16) if seed else 42

        # Switch to seeded shard-pick when the operator has flipped the
        # gate AND the caller passed a seed (i.e. validator eval, not
        # miner training). See `connito/shared/eval_shard_pick.py` for
        # the consensus assumptions; in particular, every configured
        # source must have a registered policy and (ideally) a pinned
        # revision SHA in `eval_source_revision_pin`.
        seeded_pick_enabled = (
            seed is not None
            and bool(getattr(config.task.exp.data, "eval_source_seeded_shard_pick", False))
        )
        revision_pin_map = (
            getattr(config.task.exp.data, "eval_source_revision_pin", None) or {}
        )
        if seeded_pick_enabled:
            # Lazy import — avoids pulling the HF API stack into the
            # legacy code path.
            from connito.shared.eval_shard_pick import (
                load_streaming_shard,
                pick_shard_for_source,
            )
            logger.info(
                "eval dataloader using seeded shard-pick path",
                seed=seed, int_seed=int_seed,
            )

        dataset_splits = []
        dataset_weights = []

        for source in source_specs:
            ds_name = _source_value(source, "path")
            ds_config = _source_value(source, "name")
            text_column = _source_value(source, "text_column", "text")
            weight = float(_source_value(source, "weight", 1.0))
            trust_remote_code = bool(_source_value(source, "trust_remote_code", False))

            if not ds_name:
                raise ValueError("Each dataset source must define a non-empty 'path'.")
            if not text_column:
                raise ValueError(f"Dataset source {ds_name!r} must define a non-empty 'text_column'.")
            if weight <= 0:
                raise ValueError(f"Dataset source {ds_name!r} must have a positive 'weight'.")

            pick = None
            if seeded_pick_enabled:
                pick = pick_shard_for_source(
                    repo_id=ds_name,
                    name=ds_config,
                    int_seed=int_seed,
                    revision_override=revision_pin_map.get(ds_name),
                )
                logger.info(
                    "shard pick",
                    repo_id=ds_name, name=ds_config,
                    shard=pick.shard_path, revision=pick.revision,
                    shard_rows=pick.shard_rows, offset=pick.in_shard_offset,
                )
                source_split = load_streaming_shard(pick, split_name=split_name)
            else:
                source_split = _load_streaming_split(
                    ds_name,
                    ds_config=ds_config,
                    trust_remote_code=trust_remote_code,
                )

            source_split = source_split.select_columns([text_column])
            source_split = source_split.map(
                partial(ensure_string, source_text_column=text_column),
                features=common_features,
            )

            # Eval-path data-quality gate (seed is None on the miner
            # training path, which stays byte-identical). Applied
            # per-source and before interleave so the source weights
            # keep describing *usable* rows — an unfiltered source with
            # 38% empty rows would otherwise contribute 38% NaN batches
            # at its configured weight.
            eval_min_text_chars = int(
                getattr(config.task.exp.data, "eval_min_text_chars", 0) or 0
            )
            if seed is not None and eval_min_text_chars > 0:
                source_split = source_split.filter(
                    partial(_min_text_chars_filter, min_chars=eval_min_text_chars)
                )

            if pick is not None:
                # In-shard offset goes here so the validator's read
                # window lands at a random depth inside the chosen
                # shard rather than at row 0. Bounded by the chosen
                # shard's own row count — no min-across-sources to
                # maintain, no over-skip risk past end-of-stream.
                source_split = source_split.skip(pick.in_shard_offset)

            dataset_splits.append(source_split)
            dataset_weights.append(weight)

        if not dataset_splits:
            raise ValueError("No dataset sources were configured.")

        # Streaming-shuffle each source BEFORE interleave when the caller
        # passed a seed (validator eval path; miners pass seed=None so this
        # is a no-op for training).
        #
        # Without this, the eval pool is bounded by the HEAD of each
        # source's stream: HF reads shards in file order, `interleave`
        # only changes which source supplies each position, and the
        # `_fractional_index_filter` + `split_dataset_by_node` together
        # consume ~max_eval_batches * world_size / vali_fraction
        # positions per round — for the default config (50, 10, 0.1)
        # that's ~5,000 positions, drawn from ~2,500 head rows of each
        # source regardless of seed. A 50-seed probe over
        # (allenai/c4 en, nvidia/Nemotron-CC-Math-v1 4plus) found only
        # ~2,000 distinct samples ever drawn — small enough for a miner
        # to memorize and reach near-zero validation loss without ever
        # generalizing. `.shuffle()` on a streaming dataset both permutes
        # shard order (so different shards lead each round) and
        # buffer-shuffles within the active window — together that turns
        # the candidate pool into the full source for any seed that lands
        # on a different shard permutation.
        shuffle_buffer = int(
            getattr(config.task.exp.data, "eval_source_shuffle_buffer", 0) or 0
        )
        if seed is not None and not seeded_pick_enabled and shuffle_buffer > 0:
            logger.debug(
                "Shuffling each source before interleave",
                seed=seed, int_seed=int_seed, buffer_size=shuffle_buffer,
            )
            dataset_splits = [
                ds.shuffle(seed=int_seed, buffer_size=shuffle_buffer)
                for ds in dataset_splits
            ]

        # Random per-source read offset, applied AFTER the buffer shuffle.
        # `.shuffle(seed, buffer_size=B)` alone leaves the read locked to the
        # first ~B rows of whichever shard ended up at position 0 of the
        # permuted shard list — the validator only consumes ~5K rows per
        # round and the buffer never slides deeper than that. `.skip(N)`
        # advances the read into the body of the lead shard, so the
        # reachable pool spans the full shard rather than just its head.
        # Different `int_seed` → different offset per source (RNG seeded
        # off `int_seed` advances per source) → window lands at a
        # different depth each round.
        skip_max = int(
            getattr(config.task.exp.data, "eval_source_skip_max", 0) or 0
        )
        if seed is not None and not seeded_pick_enabled and skip_max > 0:
            skip_rng = random.Random(int_seed)
            offsets = [skip_rng.randrange(0, skip_max) for _ in dataset_splits]
            logger.debug(
                "Skipping random offset per source",
                seed=seed, int_seed=int_seed, skip_max=skip_max, offsets=offsets,
            )
            dataset_splits = [
                ds.skip(offset) for ds, offset in zip(dataset_splits, offsets, strict=True)
            ]

        if len(dataset_splits) == 1:
            split = dataset_splits[0]
        else:
            total_weight = sum(dataset_weights)
            probabilities = [weight / total_weight for weight in dataset_weights]
            logger.debug("Interleaving dataset sources", probabilities=probabilities)
            split = interleave_datasets(dataset_splits, probabilities=probabilities, seed=int_seed)

        # Eval-path template dedup: drop rows repeating an already-seen
        # text prefix (templated corpora open millions of documents with
        # the same boilerplate — see `_PrefixDedupFilter`). Runs after
        # interleave so the dedup window spans the whole eval stream, and
        # before the fractional filter so surviving indices stay
        # deterministic for every validator.
        eval_dedup_prefix_chars = int(
            getattr(config.task.exp.data, "eval_dedup_prefix_chars", 0) or 0
        )
        if seed is not None and eval_dedup_prefix_chars > 0:
            split = split.filter(_PrefixDedupFilter(eval_dedup_prefix_chars))

        # Optional deterministic subsampling based on (seed, fraction)
        # Applied *before* sharding on the streaming iterable.
        if seed is not None and fraction is not None and fraction < 1.0:
            if not (0.0 < fraction <= 1.0):
                raise ValueError("fraction must be in (0.0, 1.0].")

            max_int = 2**256 - 1
            threshold = int(max_int * fraction)

            logger.debug("Applying fractional subsampling", seed=seed, fraction=fraction, threshold=threshold)

            # `with_indices=True` gives us a stable index per element in the stream.
            # Wrap with partial instead of relying on fn_kwargs to keep worker execution simple.
            filter_fn = partial(_fractional_index_filter, seed=seed, threshold=threshold)
            split = split.filter(filter_fn, with_indices=True)

        # Shard across processes if rank/world_size are provided.
        # split_dataset_by_node works with streaming datasets and avoids overlapping samples.
        if world_size is not None and rank is not None:
            try:
                split = split_dataset_by_node(split, world_size=world_size, rank=rank)
            except Exception as e:
                logger.warning(f"Falling back to unsharded split due to split_dataset_by_node error: {e}")

        # Tokenize on-the-fly via adapter (safer for streaming than heavy .map chains).
        tokenized_stream = cls(
            hf_iterable=split,
            tokenizer=tokenizer,
            seq_length=config.task.exp.data.sequence_length,
        )

        return tokenized_stream


# -----------------------------
# Dataloader
# -----------------------------
def get_dataloader(
    config,
    tokenizer: PreTrainedTokenizerBase,
    seed: int | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    train: bool = True,
    format_fn: Callable | None = None,
    data_collator: DataCollator | None = None,
) -> StatefulDataLoader:
    """
    Build a `StatefulDataLoader` over a streaming HF dataset, tokenized on the fly.

    Parameters
    ----------
    config :
        An object with fields:
                    - sequence_length (int),
                and data source configuration via either:
                    - dataset_sources (list[DatasetSourceCfg]), or
                    - legacy dataset_name/data_dir fallback behavior
        and optionally:
          - eval_world_size / world_size used by your launcher (provided here for clarity)
    tokenizer : PreTrainedTokenizerBase
        HF tokenizer used for tokenization and by the collator.
    rank : Optional[int]
        Zero-based index of the current process in the node/world. Used for sharding.
    world_size : Optional[int]
        Total number of processes. Used for sharding.
    train : bool
        If True, returns a loader over the training split; else returns a loader for validation
        (or None if the dataset has no validation split).

    Returns
    -------
    Optional[StatefulDataLoader]
        A stateful dataloader for the requested split, or None if the eval split is missing.
    """
    logger.debug("Loading dataloader", split="train" if train else "eval", seed=seed)
    # Prefer provided rank/world_size, else fall back to config (if present), else no sharding.
    world_size = world_size if world_size is not None else config.task.exp.data.world_size
    rank = rank if rank is not None else config.task.exp.data.rank

    dataset_class_path = getattr(config.task.exp.data, "dataset_class", None)
    if dataset_class_path is None:
        DatasetCls = DefaultStreamingTorchDataset
    else:
        DatasetCls = import_from_string(dataset_class_path)

    tokenised_dataset = DatasetCls.get_tokenised_dataset(
        config=config,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        train=train,
        seed=seed,  # e.g. combined validator seed
        fraction=config.task.exp.data.vali_fraction,  # use ~20% of the dataset
    )

    # Collator for causal LM (no MLM)
    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build loader
    num_workers = int(getattr(config.task.exp.data, "num_workers", 1))
    if num_workers < 0:
        num_workers = 0

    loader = StatefulDataLoader(
        tokenised_dataset,  # split
        collate_fn=data_collator,
        batch_size=config.task.exp.data.per_device_train_batch_size,
        num_workers=num_workers,
    )
    return loader


def materialize_batches(
    dataloader: Iterable, max_batches: int,
) -> list:
    """Pull up to ``max_batches + 1`` batches from a (possibly streaming)
    dataloader into a Python list, leaving HF off the per-miner critical path.

    Bg-eval re-evaluates every miner in a round against the same combined
    seed, so every miner sees the same batches. Materializing once at
    round start and iterating from RAM for each miner has two wins:

    1. **Eliminates HF network from the per-miner path.** A hung HF read
       inside the streaming iterator cannot stall a per-miner eval and
       trigger the orphan-lock cascade observed in
       `notebooks/data/validator_a100_v0.1.38.log`.
    2. **Removes redundant work.** Each miner currently rebuilds the
       dataloader and re-streams the same batches; collapsing to a
       single materialization pays the network cost once per round.

    The ``+1`` mirrors the loop guard inside ``evaluate_model``: it
    breaks ``if batch_step >= max_eval_batches``, so we keep one extra
    batch around to cover the off-by-one without it ever being scored.
    Tensors are kept on CPU here; ``evaluate_model`` moves them to the
    GPU per-batch as before.
    """
    return list(itertools.islice(dataloader, max_batches + 1))

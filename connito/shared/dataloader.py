from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from datasets import load_dataset, interleave_datasets, Features, Value
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import DataCollator, DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from connito.shared.app_logging import structlog
from connito.shared.helper import h256_int, import_from_string

logger = structlog.get_logger(__name__)


def _fractional_index_filter(_example, idx: int, seed: str | int, threshold: int) -> bool:
    """Deterministically decide whether to keep a sample based on its streaming index."""
    score = h256_int("dataset_selection", str(idx), seed)
    return score <= threshold


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
        return tokenizer(text, truncation=True, max_length=sequence_length, padding="max_length")  # type: ignore

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

        def _load_streaming_split(ds_name: str, ds_config: str | None = None):
            """Helper to load a dataset split safely, falling back to 'train' if 'validation' is missing."""
            try:
                load_kwargs = {"streaming": True, "revision": "main"}
                if ds_config is not None:
                    load_kwargs["name"] = ds_config

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

        dataset_splits = []
        dataset_weights = []

        for source in source_specs:
            ds_name = _source_value(source, "path")
            ds_config = _source_value(source, "name")
            text_column = _source_value(source, "text_column", "text")
            weight = float(_source_value(source, "weight", 1.0))

            if not ds_name:
                raise ValueError("Each dataset source must define a non-empty 'path'.")
            if not text_column:
                raise ValueError(f"Dataset source {ds_name!r} must define a non-empty 'text_column'.")
            if weight <= 0:
                raise ValueError(f"Dataset source {ds_name!r} must have a positive 'weight'.")

            source_split = _load_streaming_split(ds_name, ds_config=ds_config)
            source_split = source_split.select_columns([text_column])
            source_split = source_split.map(
                partial(ensure_string, source_text_column=text_column),
                features=common_features,
            )

            dataset_splits.append(source_split)
            dataset_weights.append(weight)

        if not dataset_splits:
            raise ValueError("No dataset sources were configured.")

        # Convert string seed to integer for interleave_datasets if provided
        int_seed = int(str(seed)[:8], 16) if seed else 42

        if len(dataset_splits) == 1:
            split = dataset_splits[0]
        else:
            total_weight = sum(dataset_weights)
            probabilities = [weight / total_weight for weight in dataset_weights]
            logger.debug("Interleaving dataset sources", probabilities=probabilities)
            split = interleave_datasets(dataset_splits, probabilities=probabilities, seed=int_seed)

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

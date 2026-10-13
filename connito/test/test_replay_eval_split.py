"""The validator's c4 eval split: replay sources picked by flag, sized by real tokens, batched without padding.

CPU only, nothing downloaded: `python -m pytest connito/test/test_replay_eval_split.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from connito.shared import dataloader
from connito.shared import model as shared_model
from connito.shared.config import DatasetSourceCfg, ResidualCfg
from connito.shared.dataloader import materialize_by_tokens
from connito.shared.evaluate import chunked_lm_loss

TEXTS = {"task/pmc": ["pmc one", "pmc two", "pmc three"], "allenai/c4": ["c4 one", "c4 two", "c4 three"]}


@pytest.fixture
def two_sources(monkeypatch) -> SimpleNamespace:
    monkeypatch.setattr(
        dataloader, "load_dataset",
        lambda path, **kw: {"train": Dataset.from_dict({"text": TEXTS[path]}).to_iterable_dataset()},
    )
    data = SimpleNamespace(sequence_length=8, dataset_sources=[
        DatasetSourceCfg(path="task/pmc", weight=0.8),
        DatasetSourceCfg(path="allenai/c4", weight=0.2, replay=True),
    ])
    return SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(data=data)))


@pytest.mark.parametrize("replay, expected", [
    (True, {"allenai/c4"}), (False, {"task/pmc"}), (None, {"task/pmc", "allenai/c4"}),
])
def test_the_replay_flag_picks_the_sources(two_sources, replay, expected):
    ds = dataloader.DefaultStreamingTorchDataset.get_tokenised_dataset(
        config=two_sources, tokenizer=None, replay=replay,
    )
    seen = {path for example in ds.hf_iterable for path, texts in TEXTS.items() if example["text"] in texts}
    assert seen == expected


@pytest.mark.parametrize("replay, kept", [(False, True), (True, False)])
def test_a_legacy_source_without_the_flag_counts_as_task_data(monkeypatch, replay, kept):
    monkeypatch.setattr(
        dataloader, "load_dataset",
        lambda path, **kw: {"train": Dataset.from_dict({"text": TEXTS[path]}).to_iterable_dataset()},
    )
    data = SimpleNamespace(sequence_length=8, dataset_sources=None, dataset_name="task/pmc")
    config = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(data=data)))

    if kept:
        ds = dataloader.DefaultStreamingTorchDataset.get_tokenised_dataset(config=config, tokenizer=None, replay=replay)
        assert [example["text"] for example in ds.hf_iterable] == TEXTS["task/pmc"]
    else:
        with pytest.raises(ValueError, match="No dataset sources"):
            dataloader.DefaultStreamingTorchDataset.get_tokenised_dataset(config=config, tokenizer=None, replay=replay)


def _row(length: int, width: int = 24, seed: int = 0) -> dict[str, torch.Tensor]:
    """One right-padded row, batched as the eval loader yields it."""
    ids = torch.randint(1, 128, (1, width), generator=torch.Generator().manual_seed(seed))
    mask = torch.zeros_like(ids)
    mask[0, :length] = 1
    ids = ids.masked_fill(mask == 0, 0)
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


LENGTHS = (6, 3, 24, 5, 4)


def _loader() -> list[dict[str, torch.Tensor]]:
    return [_row(n, seed=i) for i, n in enumerate(LENGTHS)]


def test_rows_are_taken_until_the_token_target_and_no_further():
    batches = materialize_by_tokens(_loader(), min_tokens=33, max_positions=24)
    # 6 + 3 + 24 = 33 reaches the target at the third row.
    assert sum(int(b["attention_mask"].sum()) for b in batches) == 33


def test_a_short_stream_keeps_every_row():
    batches = materialize_by_tokens(_loader(), min_tokens=10_000, max_positions=24)
    assert sum(int(b["attention_mask"].sum()) for b in batches) == sum(LENGTHS)


def test_batches_are_cut_to_their_longest_row_and_never_exceed_the_positions():
    batches = materialize_by_tokens(_loader(), min_tokens=10_000, max_positions=24)
    for b in batches:
        assert b["input_ids"].numel() <= 24
        assert bool(b["attention_mask"][:, -1].any())  # no all-padding trailing column
    # 3, 4, 5 and 6 share one forward pass at width 6; the full row runs alone.
    assert sorted(tuple(b["input_ids"].shape) for b in batches) == [(1, 24), (4, 6)]


def test_rows_with_no_labelled_token_are_skipped():
    loader = [_row(1, seed=9), _row(0, seed=8), *_loader()]
    batches = materialize_by_tokens(loader, min_tokens=10_000, max_positions=24)

    assert sorted(int(n) for b in batches for n in b["attention_mask"].sum(dim=1)) == sorted(LENGTHS)


def test_the_same_stream_gives_the_same_batches():
    first, second = (materialize_by_tokens(_loader(), 10_000, 24) for _ in range(2))
    assert all(torch.equal(a[k], b[k]) for a, b in zip(first, second, strict=True) for k in a)


def _summed_ce(model, batches) -> float:
    """Cross-entropy summed over every predicted token, so batches of any shape compare."""
    return sum(float(chunked_lm_loss(model, b)) * int(b["labels"][:, 1:].ne(-100).sum()) for b in batches)


def test_batched_trimmed_rows_score_as_one_row_per_pass(family_name):
    config = SimpleNamespace(
        role="validator",
        model=SimpleNamespace(family=family_name, device="cpu", quantization="off", quantization_validator="off"),
        task=SimpleNamespace(exp=SimpleNamespace(residual=ResidualCfg(k=1, threshold=0.5, temperature=0.5))),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )
    model, _ = shared_model.get_model_from_checkpoint(config=config, load_global_checkpoint=False)
    model.float().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".residual_experts." in name and "down_proj" in name:
                param.normal_(std=0.02)  # a trained residual, so its gate and experts are in the loss
        batched = _summed_ce(model, materialize_by_tokens(_loader(), 10_000, 24))
        one_per_pass = _summed_ce(model, _loader())
    assert batched == pytest.approx(one_per_pass, rel=1e-5)

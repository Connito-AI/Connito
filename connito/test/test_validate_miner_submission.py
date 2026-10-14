"""`validate_miner_submission`: a miner's on-disk shard against its chain commit and the residual spec.

The real `ChainCheckpoint.validate` runs on real files; only the signature check is stubbed.
CPU only, nothing downloaded: `python -m pytest connito/test/test_validate_miner_submission.py`.
"""

from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from connito.shared import checkpoint_helper
from connito.shared.checkpoint_helper import TensorSpec
from connito.shared.checkpoints import ChainCheckpoint
from connito.shared.helper import get_model_hash
from connito.validator.evaluator import validate_miner_submission

HIDDEN, WIDTH, LAYERS = 8, 4, (1, 2)


def _spec(k: int = 1) -> dict[str, TensorSpec]:
    spec = {}
    for layer in LAYERS:
        prefix = f"model.layers.{layer}.mlp.residual_"
        for j in range(k):
            spec[f"{prefix}experts.{j}.gate_proj.weight"] = TensorSpec((WIDTH, HIDDEN), torch.bfloat16)
            spec[f"{prefix}experts.{j}.up_proj.weight"] = TensorSpec((WIDTH, HIDDEN), torch.bfloat16)
            spec[f"{prefix}experts.{j}.down_proj.weight"] = TensorSpec((HIDDEN, WIDTH), torch.bfloat16)
        spec[f"{prefix}router.weight"] = TensorSpec((k, HIDDEN), torch.float32)
    return spec


SPEC = _spec()
EXPERT = "model.layers.2.mlp.residual_experts.0"
ROUTER = "model.layers.2.mlp.residual_router.weight"


def _trained(spec: dict[str, TensorSpec]) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {k: torch.randn(s.shape, generator=gen).to(s.dtype) for k, s in spec.items()}


def _zeroed(*projections: str) -> dict[str, torch.Tensor]:
    tensors = _trained(SPEC)
    for proj in projections:
        tensors[f"{EXPERT}.{proj}.weight"].zero_()
    return tensors


def _with(key: str, value: float) -> dict[str, torch.Tensor]:
    tensors = _trained(SPEC)
    tensors[key][0, 0] = value
    return tensors


# Each is off spec or untrained in one way, and hashes to what its commit says.
MALFORMED = {
    "extra key": {**_trained(SPEC), "model.layers.0.mlp.gate.weight": torch.zeros(2, HIDDEN, dtype=torch.bfloat16)},
    "missing key": {k: t for k, t in _trained(SPEC).items() if k != ROUTER},
    "wrong shape": {**_trained(SPEC), ROUTER: torch.randn(1, HIDDEN + 1)},
    "bf16 router": {**_trained(SPEC), ROUTER: torch.randn(1, HIDDEN).to(torch.bfloat16)},
    "another task's K": _trained(_spec(k=2)),
    "NaN": _with(ROUTER, float("nan")),
    "Inf": _with(f"{EXPERT}.up_proj.weight", float("inf")),
    "all zero": {k: torch.zeros_like(t) for k, t in _trained(SPEC).items()},
    "down_proj zero (an untouched init)": _zeroed("down_proj"),
    "gate_proj and up_proj zero": _zeroed("gate_proj", "up_proj"),
}


@pytest.fixture(autouse=True)
def signed(monkeypatch):
    monkeypatch.setattr("connito.shared.checkpoints.verify_message", lambda *a, **kw: True)


def _commit(tensors: dict[str, torch.Tensor], **fields) -> ChainCheckpoint:
    return ChainCheckpoint(
        uid=7, hotkey="miner-hotkey", signed_model_hash="signed", model_hash=get_model_hash(tensors, hex=True),
        expert_group=9, signature_required=True, hash_required=True, **fields,
    )


def _validate(path: Path, ckpt) -> str | None:
    return validate_miner_submission(
        round_obj=SimpleNamespace(uid_to_chain_checkpoint={7: ckpt}), uid=7, model_path=path, expected=SPEC,
    )


def _submit(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> tuple[str | None, ChainCheckpoint]:
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    ckpt = _commit(tensors)
    return _validate(path, ckpt), ckpt


def test_a_trained_shard_on_spec_passes(tmp_path):
    reason, ckpt = _submit(tmp_path, _trained(SPEC))

    assert reason is None
    assert ckpt.path == tmp_path / "model.safetensors"


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_a_shard_off_spec_or_untrained_is_a_bad_residual_shard(tmp_path, case):
    assert _submit(tmp_path, MALFORMED[case])[0] == "bad_residual_shard"


def test_an_expert_with_only_gate_proj_zero_still_trains_and_passes(tmp_path):
    assert _submit(tmp_path, _zeroed("gate_proj"))[0] is None


def _counting_reads(monkeypatch, fail: bool = False) -> list[str]:
    """Record every tensor `verify_residual_shard` reads; with `fail`, each read raises as a bad disk would."""
    reads: list[str] = []
    real = checkpoint_helper.safe_open

    def _read(f, key):
        reads.append(key)
        if fail:
            raise OSError("read error")
        return f.get_tensor(key)

    @contextlib.contextmanager
    def _open(*args, **kwargs):
        with real(*args, **kwargs) as f:
            yield SimpleNamespace(keys=f.keys, get_slice=f.get_slice, get_tensor=lambda k: _read(f, k))

    monkeypatch.setattr(checkpoint_helper, "safe_open", _open)
    return reads


def test_an_off_spec_header_is_refused_before_any_tensor_is_read(tmp_path, monkeypatch):
    reads = _counting_reads(monkeypatch)

    assert _submit(tmp_path, MALFORMED["extra key"])[0] == "bad_residual_shard"
    assert reads == []


def test_the_tensors_checked_are_read_in_the_same_open_as_the_header(tmp_path, monkeypatch):
    reads = _counting_reads(monkeypatch)

    assert _submit(tmp_path, _trained(SPEC))[0] is None
    assert sorted(reads) == sorted(SPEC)


def test_a_tensor_that_cannot_be_read_is_a_bad_residual_shard(tmp_path, monkeypatch):
    _counting_reads(monkeypatch, fail=True)

    assert _submit(tmp_path, _trained(SPEC))[0] == "bad_residual_shard"


def test_a_shard_not_named_safetensors_is_a_bad_residual_shard(tmp_path):
    """The eval loads by suffix, so even safetensors bytes named `.pt` would go to torch.load."""
    save_file(_trained(SPEC), str(tmp_path / "model.pt"))

    assert _validate(tmp_path / "model.pt", _commit(_trained(SPEC))) == "bad_residual_shard"


def test_an_unreadable_file_is_a_bad_residual_shard(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")

    assert _validate(tmp_path / "model.safetensors", _commit(_trained(SPEC))) == "bad_residual_shard"


def test_a_bad_signature_is_named_first(tmp_path, monkeypatch):
    monkeypatch.setattr("connito.shared.checkpoints.verify_message", lambda *a, **kw: False)

    assert _submit(tmp_path, MALFORMED["NaN"])[0] == "signature"


def test_a_shard_on_spec_with_another_hash_is_a_hash_mismatch(tmp_path):
    save_file(_trained(SPEC), str(tmp_path / "model.safetensors"))
    ckpt = _commit(_trained(SPEC))
    ckpt.model_hash = "abcd"

    assert _validate(tmp_path / "model.safetensors", ckpt) == "hash"


def test_no_chain_commit_is_named(tmp_path):
    reason = validate_miner_submission(
        round_obj=SimpleNamespace(uid_to_chain_checkpoint={}), uid=7, model_path=tmp_path / "x", expected=SPEC,
    )

    assert reason == "no_chain_commit"


def test_validate_raising_is_unknown(tmp_path):
    ckpt = SimpleNamespace(path=None, validate=MagicMock(side_effect=RuntimeError("boom")))

    assert _validate(tmp_path / "x", ckpt) == "unknown"


def test_a_downloaded_baseline_folder_is_checked_by_its_groups_shard(tmp_path):
    """The miner's fetch validates the folder it downloaded into."""
    save_file(_trained(SPEC), str(tmp_path / "model_expgroup_9.safetensors"))
    assert _commit(_trained(SPEC), path=tmp_path).validate(SPEC)

    untrained = MALFORMED["down_proj zero (an untouched init)"]
    save_file(untrained, str(tmp_path / "model_expgroup_9.safetensors"))
    assert not _commit(untrained, path=tmp_path).validate(SPEC)


def test_the_helper_takes_the_kwargs_its_callers_pass():
    params = inspect.signature(validate_miner_submission).parameters
    for name in ("round_obj", "uid", "model_path", "expected"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY

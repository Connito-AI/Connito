"""The residual shard: its spec for each family and K, how it is written, and what a check refuses.

CPU only, tiny models, nothing downloaded: `python -m pytest connito/test/test_residual_shard.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from connito.shared import model as shared_model
from connito.shared.checkpoint_helper import save_checkpoint
from connito.shared.config import ResidualCfg
from connito.shared.modeling.residual import freeze_all_but_residual, residual_shapes

# The tiny checkpoints' MoE layers (DeepSeek-V2's layer 0 is dense), hidden size and expert width.
MOE_LAYERS = {"qwen3_5_moe": [0, 1, 2, 3], "deepseek_v2": [1, 2]}
HIDDEN, WIDTH = 64, 32


def _config(name: str, k: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        role="miner",
        model=SimpleNamespace(family=name, device="cpu", quantization="off", quantization_miner="off"),
        task=SimpleNamespace(
            exp=SimpleNamespace(group_id=9, residual=ResidualCfg(k=k, threshold=0.5, temperature=0.5)),
        ),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
    )


def _model(config) -> torch.nn.Module:
    model, _ = shared_model.get_model_from_checkpoint(config=config, load_global_checkpoint=False)
    return model


def _expected(name: str, k: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    spec = {}
    for layer in MOE_LAYERS[name]:
        prefix = f"model.layers.{layer}.mlp.residual_"
        for j in range(k):
            spec[f"{prefix}experts.{j}.gate_proj.weight"] = ((WIDTH, HIDDEN), torch.bfloat16)
            spec[f"{prefix}experts.{j}.up_proj.weight"] = ((WIDTH, HIDDEN), torch.bfloat16)
            spec[f"{prefix}experts.{j}.down_proj.weight"] = ((HIDDEN, WIDTH), torch.bfloat16)
        spec[f"{prefix}router.weight"] = ((k, HIDDEN), torch.float32)
    return spec


@pytest.mark.parametrize("k", [1, 2])
def test_residual_shapes_are_the_familys_moe_layers_at_k(family_name, k):
    assert residual_shapes(_config(family_name, k)) == _expected(family_name, k)


def test_residual_shapes_match_the_installed_residual(family_name):
    config = _config(family_name, k=2)

    live = {n: (tuple(p.shape), p.dtype) for n, p in _model(config).named_parameters() if ".residual_" in n}

    assert residual_shapes(config) == live


def test_a_checkpoint_holds_exactly_the_residual_each_tensor_in_its_own_dtype(family_name, tmp_path):
    config = _config(family_name, k=2)
    model = _model(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".residual_" in name:
                param.normal_()  # fp32 router values that bf16 would round
    residual = {n: p.detach().clone() for n, p in model.named_parameters() if ".residual_" in n}

    save_checkpoint(checkpoint_path=tmp_path / "globalver_1_inneropt_2", model=model, group_id=9, rank=0,
                    save_global_state=False)

    saved = load_file(str(tmp_path / "globalver_1_inneropt_2" / "model_expgroup_9.safetensors"))
    assert {n: (tuple(t.shape), t.dtype) for n, t in saved.items()} == residual_shapes(config)
    assert all(torch.equal(saved[n], t) for n, t in residual.items())


# --- the miner's start: a trained residual on spec, or its own fresh draw ------

def _shard_tensors(config, k: int | None = None) -> dict[str, torch.Tensor]:
    spec = residual_shapes(_config(config.model.family, k or config.task.exp.residual.k))
    return {n: torch.randn(s.shape).to(s.dtype) for n, s in spec.items()}


BASELINE = "uid_3_hotkey_hk_globalver_5"  # a downloaded round baseline, newer than the miner's own
OWN = "globalver_1_inneropt_100"


def _start(config, tmp_path, shards: dict[str, dict[str, torch.Tensor] | None]):
    """The model the miner trains from and the checkpoint returned, given its folders: {name: tensors or None}."""
    config.ckpt = SimpleNamespace(
        resume_from_ckpt=True, use_pretrained_only=False,
        validator_checkpoint_path=tmp_path / "validator", checkpoint_path=tmp_path / "miner",
    )
    for name, tensors in shards.items():
        root = config.ckpt.validator_checkpoint_path if name.startswith("uid_") else config.ckpt.checkpoint_path
        folder = root / name
        folder.mkdir(parents=True)
        if tensors is not None:
            save_file(tensors, str(folder / "model_expgroup_9.safetensors"))
    config.ckpt.validator_checkpoint_path.mkdir(exist_ok=True)
    config.ckpt.checkpoint_path.mkdir(exist_ok=True)
    return shared_model.get_model_from_checkpoint(config=config)


def _residual(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if ".residual_" in n}


def test_a_trained_checkpoint_is_the_start(family_name, tmp_path):
    config = _config(family_name)
    tensors = _shard_tensors(config)

    model, _ = _start(config, tmp_path, {BASELINE: tensors})

    assert all(torch.equal(p, tensors[n]) for n, p in _residual(model).items())


def test_the_newest_trained_checkpoint_is_the_start(family_name, tmp_path):
    config = _config(family_name)
    newer, own = _shard_tensors(config), _shard_tensors(config)

    model, _ = _start(config, tmp_path, {BASELINE: newer, OWN: own})

    assert all(torch.equal(p, newer[n]) for n, p in _residual(model).items())


@pytest.mark.parametrize("baseline", ["an empty download folder", "an untrained shard"])
def test_a_refused_newer_checkpoint_falls_back_to_the_miners_own(family_name, tmp_path, baseline):
    config = _config(family_name)
    own = _shard_tensors(config)
    newer = None if baseline == "an empty download folder" else _untrained(config, "down_proj zero")

    model, latest = _start(config, tmp_path, {BASELINE: newer, OWN: own})

    assert all(torch.equal(p, own[n]) for n, p in _residual(model).items())
    # Still the newest on disk, which the resync check compares the next newest with.
    assert latest.global_ver == 5


def _untrained(config, case: str) -> dict[str, torch.Tensor]:
    if case == "another task's K":
        return _shard_tensors(config, k=2)
    tensors = _shard_tensors(config)
    zeroed = {"all zero": ("gate_proj", "up_proj", "down_proj", "router"),
              "down_proj zero": ("down_proj",), "gate_proj and up_proj zero": ("gate_proj", "up_proj")}[case]
    for name, t in tensors.items():
        if any(part in name for part in zeroed):
            t.zero_()
    return tensors


@pytest.mark.parametrize("case", ["all zero", "down_proj zero", "gate_proj and up_proj zero", "another task's K"])
def test_an_untrained_or_off_spec_checkpoint_leaves_the_fresh_draw(family_name, tmp_path, case):
    """And one step from that draw moves every down_proj, which no step from an all-zero residual would."""
    config = _config(family_name)
    tensors = _untrained(config, case)

    model, _ = _start(config, tmp_path, {BASELINE: tensors})

    residual = _residual(model)
    # The install's own draw, not the file's tensors.
    assert all(p.any() for n, p in residual.items() if "gate_proj" in n or "up_proj" in n)
    assert not any(torch.equal(p, tensors[n]) for n, p in residual.items() if n in tensors and "down_proj" not in n)
    freeze_all_but_residual(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))
    model.train()(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    assert all(p.any() for n, p in residual.items() if "down_proj" in n)

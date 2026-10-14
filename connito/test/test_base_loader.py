"""`get_base_model` loads each family's text-only causal LM from a checkpoint laid out like the published one.

CPU only, tiny models, nothing is downloaded.
Run with `python -m pytest connito/test/test_base_loader.py`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from connito.shared.config import ModelCfg
from connito.shared.modeling import mycelia
from connito.shared.modeling.families import FAMILIES, family

# Where each family's checkpoint keys land in the loaded text model.
EXPECTED_KEYS = {
    "qwen3_5_moe": lambda names: {
        n.replace("model.language_model.", "model.", 1) for n in names if n.startswith("model.language_model.")
    } | {"lm_head.weight"},
    "deepseek_v2": lambda names: {
        re.sub(r"experts\.\d+\.(gate|up)_proj\.weight", "experts.gate_up_proj",
               re.sub(r"experts\.\d+\.down_proj\.weight", "experts.down_proj", n))
        for n in names
    },
}
# How each model was trained, and what production's transformers 4 tokenizer gave DeepSeek-V2.
TRAINED_WITH_BOS = {"qwen3_5_moe": False, "deepseek_v2": True}
FIXTURE = {"qwen3_5_moe": "tiny_qwen_checkpoint", "deepseek_v2": "tiny_deepseek_v2_checkpoint"}


def _config(monkeypatch, name: str, path: Path) -> SimpleNamespace:
    """A config naming `name`, whose family record points at the tiny checkpoint in `path`."""
    monkeypatch.setitem(FAMILIES, name, replace(FAMILIES[name], model_path=str(path)))
    return SimpleNamespace(model=SimpleNamespace(family=name, torch_compile=False))


@pytest.fixture(params=sorted(FAMILIES))
def checkpoint(request, monkeypatch) -> tuple[SimpleNamespace, Path]:
    path = request.getfixturevalue(FIXTURE[request.param])
    return _config(monkeypatch, request.param, path), path


def test_keys_are_the_text_model(checkpoint):
    config, path = checkpoint
    names = load_file(str(path / "model.safetensors"))

    model = mycelia.get_base_model(config)

    assert isinstance(model, family(config).causal_lm)
    assert set(model.state_dict()) == EXPECTED_KEYS[config.model.family](names)
    assert not any("visual" in k or k.startswith("mtp.") for k in model.state_dict())


def test_loads_in_bf16_and_the_loss_is_the_language_model_loss(checkpoint):
    config, path = checkpoint
    # Saved in fp32, so only the loader can make it bf16.
    weights = {name: tensor.float() for name, tensor in load_file(str(path / "model.safetensors")).items()}
    save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text((path / "config.json").read_text().replace("bfloat16", "float32"))

    model = mycelia.get_base_model(config)
    tokens = torch.randint(0, 128, (2, 8), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        out = model(input_ids=tokens, labels=tokens)

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    # No router aux loss folded in.
    expected = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, 128), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(out.loss.float(), expected)


def test_a_missing_tensor_is_refused(checkpoint):
    config, path = checkpoint
    weights = load_file(str(path / "model.safetensors"))
    del weights[next(n for n in weights if n.endswith("model.norm.weight"))]
    save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})

    with pytest.raises(ValueError, match="model.norm.weight"):
        mycelia.get_base_model(config)


def test_the_familys_revision_reaches_model_and_tokenizer(checkpoint, monkeypatch):
    config, _ = checkpoint
    fam = family(config)
    calls: list[tuple] = []
    load_model = fam.causal_lm.from_pretrained

    def _model(path, **kwargs):
        calls.append(("model", path, kwargs["revision"]))
        return load_model(path, **kwargs)

    def _tokenizer(path, **kwargs):
        calls.append(("tokenizer", path, kwargs["revision"]))

    monkeypatch.setattr(fam.causal_lm, "from_pretrained", _model)
    monkeypatch.setattr(mycelia.AutoTokenizer, "from_pretrained", _tokenizer)

    mycelia.get_base_model(config)
    mycelia.get_base_tokenizer(config)

    assert calls == [
        ("model", fam.model_path, fam.revision),
        ("tokenizer", fam.model_path, fam.revision),
    ]


def test_bos_is_prepended_only_for_the_family_trained_with_it(checkpoint):
    config, _ = checkpoint
    tokenizer = mycelia.get_base_tokenizer(config)

    ids = tokenizer("hello world").input_ids

    plain = tokenizer("hello world", add_special_tokens=False).input_ids
    assert plain
    assert ids == ([tokenizer.bos_token_id] if TRAINED_WITH_BOS[config.model.family] else []) + plain


def test_the_config_names_exactly_the_registered_families():
    assert set(get_args(ModelCfg.model_fields["family"].annotation)) == set(FAMILIES)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_each_family_is_one_config_value(name):
    config = SimpleNamespace(model=ModelCfg(family=name))

    assert family(config) is FAMILIES[name]


def test_qwen_logits_match_the_vision_language_model(tiny_qwen_checkpoint, tiny_qwen_vlm, monkeypatch):
    model = mycelia.get_base_model(_config(monkeypatch, "qwen3_5_moe", tiny_qwen_checkpoint)).float()
    reference = tiny_qwen_vlm.float()
    input_ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        # Not bit-exact: the VLM builds its rotary positions through the multimodal path.
        torch.testing.assert_close(model(input_ids=input_ids).logits, reference(input_ids=input_ids).logits)


def test_deepseek_v2_moe_block_computes_from_the_per_expert_tensors(tiny_deepseek_v2_checkpoint, monkeypatch):
    """Layer 1's MoE output equals DeepSeek's formula applied by hand to the checkpoint's own expert tensors."""
    model = mycelia.get_base_model(_config(monkeypatch, "deepseek_v2", tiny_deepseek_v2_checkpoint))
    block = model.float().model.layers[1].mlp
    prefix = "model.layers.1.mlp."
    w = {
        n[len(prefix):]: t.float()
        for n, t in load_file(str(tiny_deepseek_v2_checkpoint / "model.safetensors")).items()
        if n.startswith(prefix)
    }
    h = torch.randn(5, 64, generator=torch.Generator().manual_seed(2))

    def mlp(x, name):
        return (F.silu(x @ w[f"{name}.gate_proj.weight"].T) * (x @ w[f"{name}.up_proj.weight"].T)) @ w[
            f"{name}.down_proj.weight"].T

    # Greedy top-2 over the softmax, weights not renormalised (norm_topk_prob=False), plus the shared experts.
    top_w, top_i = (h @ w["gate.weight"].T).softmax(-1).topk(2, dim=-1)
    routed = torch.stack([
        sum(top_w[t, k] * mlp(h[t], f"experts.{int(top_i[t, k])}") for k in range(2)) for t in range(len(h))
    ])
    expected = routed + mlp(h, "shared_experts")

    with torch.no_grad():
        torch.testing.assert_close(block(h[None])[0], expected)


def test_expert_tensors_are_read_at_the_familys_revision(tiny_deepseek_v2_checkpoint, monkeypatch):
    config = _config(monkeypatch, "deepseek_v2", tiny_deepseek_v2_checkpoint)
    weights = load_file(str(tiny_deepseek_v2_checkpoint / "model.safetensors"))
    index = {"weight_map": dict.fromkeys(weights, "model.safetensors")}
    (tiny_deepseek_v2_checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    revisions: list[str] = []

    def _cached_file(path, filename, revision):
        revisions.append(revision)
        return str(Path(path) / filename)

    monkeypatch.setattr(mycelia, "cached_file", _cached_file)

    tensors = mycelia.load_pretrained_expert_tensors(config, {1: [(0, 3), (1, 5)]})

    assert revisions == [family(config).revision] * 2
    assert set(tensors) == {
        f"model.layers.1.mlp.experts.{i}.{p}_proj.weight" for i in (3, 5) for p in ("gate", "up", "down")
    }


def test_deepseek_v2_attention_keeps_the_yarn_mscale(tiny_deepseek_v2_checkpoint, monkeypatch):
    """YaRN scales V2's attention by mscale(factor, mscale_all_dim)²; staging_v3's transformers 4 build left it out."""
    model = mycelia.get_base_model(_config(monkeypatch, "deepseek_v2", tiny_deepseek_v2_checkpoint))

    # The fixture's qk_nope_head_dim + qk_rope_head_dim, mscale_all_dim and factor.
    expected = (16 + 8) ** -0.5 * (0.1 * 0.707 * math.log(40) + 1) ** 2
    assert [layer.self_attn.scaling for layer in model.model.layers] == pytest.approx([expected] * 3)


def test_the_two_tiny_checkpoints_do_not_share_a_directory(tiny_qwen_checkpoint, tiny_deepseek_v2_checkpoint):
    model_types = {json.loads((p / "config.json").read_text())["model_type"]
                   for p in (tiny_qwen_checkpoint, tiny_deepseek_v2_checkpoint)}

    assert model_types == {"qwen3_5_moe", "deepseek_v2"}

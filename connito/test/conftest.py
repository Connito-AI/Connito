"""Tiny checkpoints of each model family, laid out like the real ones, for CPU tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers.models.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM
from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

TINY_TEXT_CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=8,
    num_experts_per_tok=2,
    max_position_embeddings=128,
    # On, so a loader that leaves it on is caught.
    output_router_logits=True,
)
TINY_VISION_CONFIG = dict(
    depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, num_position_embeddings=16,
)
# DeepSeek-V2-Lite's config.json in miniature: dense layer 0, MLA without a q LoRA, YaRN in the legacy form.
TINY_DEEPSEEK_V2_CONFIG = dict(
    architectures=["DeepseekV2ForCausalLM"],
    model_type="deepseek_v2",
    auto_map={"AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"},
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=3,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    kv_lora_rank=16,
    q_lora_rank=None,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    n_routed_experts=8,
    n_shared_experts=2,
    num_experts_per_tok=2,
    topk_method="greedy",
    norm_topk_prob=False,
    max_position_embeddings=1280,
    rope_scaling=dict(
        type="yarn", factor=40, original_max_position_embeddings=32,
        beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
    ),
    torch_dtype="bfloat16",
)


def _write_tokenizer(path: Path, tokenizer_config: dict) -> None:
    """A byte-level BPE `tokenizer.json` with no BOS template in its post-processor, as both real ones are."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["hello world"], trainers.BpeTrainer(
        special_tokens=["<bos>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))


@pytest.fixture()
def tiny_qwen_vlm() -> Qwen3_5MoeForConditionalGeneration:
    torch.manual_seed(0)
    config = Qwen3_5MoeConfig(text_config=TINY_TEXT_CONFIG, vision_config=TINY_VISION_CONFIG)
    return Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16).eval()


@pytest.fixture()
def tiny_qwen_checkpoint(tmp_path: Path, tiny_qwen_vlm) -> Path:
    """The VLM saved with fused expert tensors and an `mtp.*` tensor, as the published checkpoint is."""
    tiny_qwen_vlm.save_pretrained(tmp_path)
    # `save_pretrained` splits fused experts into per-expert keys; the real checkpoint keeps them fused.
    state = {name: tensor.contiguous() for name, tensor in tiny_qwen_vlm.state_dict().items()}
    state["mtp.fc.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    _write_tokenizer(tmp_path, {"tokenizer_class": "Qwen2Tokenizer", "bos_token": None, "eos_token": "<eos>"})
    return tmp_path


@pytest.fixture()
def tiny_deepseek_v2_checkpoint(tmp_path: Path) -> Path:
    """Per-expert `mlp.experts.{i}.*_proj` tensors and the original config.json, as the published checkpoint is."""
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(DeepseekV2Config(**TINY_DEEPSEEK_V2_CONFIG)).to(torch.bfloat16)
    # transformers 5 splits its fused experts back into per-expert keys on save.
    model.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(TINY_DEEPSEEK_V2_CONFIG))
    # The published tokenizer_config asks for BOS, but its tokenizer.json has no template that adds one.
    _write_tokenizer(tmp_path, {
        "tokenizer_class": "LlamaTokenizerFast", "add_bos_token": True, "bos_token": "<bos>", "eos_token": "<eos>",
    })
    return tmp_path

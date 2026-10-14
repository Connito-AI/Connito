# Model families

> **Audience:** maintainers. Miners and validators run whatever family the
> release pins; nothing here is an operator setting.

Connito's base model is chosen by one locked config value, `model.family`. Each
family is a record of data in `connito/shared/modeling/families.py`; there is
no per-family code path elsewhere.

## The active family

`qwen3_5_moe`: `Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, loaded as a text-only
`Qwen3_5MoeForCausalLM` (the checkpoint's vision tower and MTP layer are
dropped). `deepseek_v2` (`deepseek-ai/DeepSeek-V2-Lite` on stock transformers)
stays in the registry so the generic code keeps a second family to be tested
against.

`ModelCfg.family` is locked: a node whose YAML names another family is reset
to the release default on load. Switching the subnet's model is a release that
changes that default.

## What a family record holds

`ModelFamily` is a frozen dataclass; `FAMILIES` maps `name` to the record, and
`family(config)` returns `FAMILIES[config.model.family]`.

| field | used for |
|---|---|
| `name` | the `model.family` value |
| `model_path`, `revision` | the Hugging Face checkpoint and its pinned commit; every load and tokenizer read uses both |
| `causal_lm` | the transformers class `get_base_model` loads |
| `add_bos_token` | passed to the tokenizer (transformers 5 ignores DeepSeek-V2's own setting) |
| `experts` | the fused routed-experts module stored in fp8 |
| `fp8_deny` | linear layers kept in full precision under the fp8 `all` scope |
| `moe_block` | the block residual experts attach to (dense layers are other types, so they are skipped) |
| `mlp` | the family's dense SwiGLU MLP, used as each residual expert |

Everything else is generic: hidden and expert widths come from the model
config, MoE layers are found with `isinstance(m, family.moe_block)`, and the
residual's init scale is read from the base's experts and router.

## Adding a family

1. **Add one record** to `families.py` and to `FAMILIES`, and add the name to
   the `ModelCfg.family` `Literal` in `connito/shared/config.py`.
2. **Add a tiny checkpoint fixture** in `connito/test/conftest.py`, laid out
   like the real checkpoint (same key names and nesting), and map the family
   to it in the `family_name` fixture. The CPU suites that take
   `family_name` (`test_base_loader.py`, `test_residual_experts.py`,
   `test_residual_shard.py`, `test_residual_training.py`, among others) then
   run on it.
3. **Run the real-model GPU checks** on one GPU before relying on it:
   - loader parity: `get_base_model` logits equal the plain transformers
     class at the pinned revision, with no missing keys;
   - fp8 at load: loss and top-1 against bf16 for the `experts` and `all`
     scopes;
   - step 0 equals the base bit for bit with the residual installed, and one
     training step moves only residual tensors;
   - `residual_shapes` equals the live model's residual, and a trained shard
     saves, verifies and reloads to identical outputs.
4. **Measure memory** for the miner (one training step at the default
   micro-batch) and the validator (task eval plus c4 eval) on the smallest
   supported card, and update the hardware sections of
   `docs/miner-faq/miner-config.md` and `docs/validator-docker-operators.md`.

**Keep the model on one GPU.** `get_base_model` refuses a model spread over
more than one device: Qwen split with `device_map="auto"` over three GPUs
returned wrong logits with no error. Do not add a multi-GPU path for a new
family without first showing it is exact.

Code: `connito/shared/modeling/families.py`,
`connito/shared/modeling/mycelia.py:get_base_model`,
`connito/shared/modeling/quantization.py`,
`connito/shared/modeling/residual.py`, `connito/test/conftest.py`.

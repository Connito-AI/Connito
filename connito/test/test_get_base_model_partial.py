"""Real integration test for `get_base_model(partial=True)`.

Downloads DeepSeek-V2-Lite from HuggingFace, builds a partial model
through the real `get_base_model` code path with a real `ExpertManager`
loaded from `expert_groups/exp_dummy`, and verifies that backbone +
owned-expert tensors match the pretrained checkpoint instead of the
random `_init_weights` default.

CPU-only — every weight load and tensor comparison runs in host RAM.
VRAM is never touched: the script sets `config.model.device = "cpu"`,
so `get_base_model` keeps the partial model on CPU and the pretrained
state dict is also loaded on CPU (production miners with a real GPU
would land the partial model in VRAM instead, leaving the CPU state
dict to drain into VRAM tensor-by-tensor as it streams).

No mocks. Heavy — peak ~40 GB RAM during the streaming load
(partial model + the still-being-consumed pretrained state dict),
and another ~40 GB peak during the verification compare. First run
pulls ~16 GB from HuggingFace; subsequent runs reuse the HF cache.

Run from the repo root:

    python3 -m connito.test.test_get_base_model_partial

Optional flags:

    --group-id INT        group id from expert_groups/exp_dummy
                          (default: 1; this is what the fixture defines)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from connito.shared.config import MinerConfig
from connito.shared.expert_manager import ExpertManager
from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeekSeekMoE
from connito.shared.modeling.mycelia import (
    get_base_model,
    get_base_tokenizer,
    load_pretrained_state_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DUMMY_DIR = REPO_ROOT / "expert_groups" / "exp_dummy"
MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite"
GROUP_ID = 1


def _build_config(group_id: int) -> MinerConfig:
    """Construct a real `MinerConfig` pointing at the on-disk exp_dummy
    fixture. No subclassing or field bypass — only public knobs.

    `model.device = "cpu"` keeps any downstream `.to(device=...)` calls
    off CUDA. `get_base_model` itself never moves the model anywhere,
    but setting this explicitly makes the CPU-only intent clear."""
    cfg = MinerConfig()
    cfg.role = "miner"
    cfg.model.model_path = MODEL_PATH
    cfg.model.base_arch_model = MODEL_PATH
    cfg.model.device = "cpu"
    cfg.model.precision = "bf16-mixed"
    # `use_quantization` / `use_unsloth` were set here but never existed as
    # ModelCfg fields, so under pydantic v2 with extra="ignore" these lines
    # raised on __setattr__. The knob is now `model.quantization`.
    cfg.model.quantization = "off"
    cfg.model.torch_compile = False
    cfg.task.expert_group_name = "exp_dummy"
    cfg.task.base_path = EXP_DUMMY_DIR.parent
    cfg.task.path = EXP_DUMMY_DIR
    cfg.task.load_all_expert_groups = False
    cfg.task.exp.group_id = group_id
    cfg.task.exp.data.sequence_length = 128
    return cfg


def _assert(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise AssertionError(message)


def _compare_backbone(partial_state: dict, full_state: dict) -> None:
    """Walk every key shared between the partial model and the
    pretrained HF state dict; for shape-matched keys, assert the
    partial model's tensor equals the pretrained tensor."""
    shared = [k for k in partial_state if k in full_state]
    matched = 0
    mismatched: list[str] = []
    skipped_shape: list[str] = []

    for key in shared:
        p = partial_state[key]
        f = full_state[key]
        if tuple(p.shape) != tuple(f.shape):
            skipped_shape.append(key)
            continue
        if torch.equal(p.detach().cpu().to(f.dtype), f.detach().cpu()):
            matched += 1
        else:
            mismatched.append(key)

    print(f"  shape-matched backbone keys: {matched} / {len(shared) - len(skipped_shape)}")
    if mismatched:
        print(f"  first 5 mismatched keys: {mismatched[:5]}")
    if skipped_shape:
        print(f"  shape-mismatched keys (expected — sliced experts): {len(skipped_shape)}")

    _assert(
        matched > 0,
        "at least one backbone tensor matches the pretrained checkpoint",
    )
    _assert(
        not mismatched,
        f"every shape-matched key equals the pretrained value (got {len(mismatched)} mismatches)",
    )


def _compare_owned_experts(
    partial_model: CustomDeekSeekMoE,
    full_state: dict,
    expert_group_assignment: dict,
    target_group: int,
) -> None:
    """For each owned (my_idx, org_idx) pair, verify the partial
    model's expert tensor equals the pretrained expert at org_idx."""
    partial_state = partial_model.state_dict()
    assignments = expert_group_assignment.get(target_group, {})

    checked = 0
    mismatched: list[str] = []
    missing_full: list[str] = []

    for layer_idx, layer_pairs in assignments.items():
        if not layer_pairs:
            continue
        for my_idx, org_idx in layer_pairs:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer_idx}.mlp.experts.{my_idx}.{proj}.weight"
                if key not in partial_state:
                    continue
                full_key_per_expert = (
                    f"model.layers.{layer_idx}.mlp.experts.{org_idx}.{proj}.weight"
                )
                if full_key_per_expert not in full_state:
                    missing_full.append(full_key_per_expert)
                    continue
                p = partial_state[key].detach().cpu()
                f = full_state[full_key_per_expert].detach().cpu()
                if tuple(p.shape) != tuple(f.shape):
                    continue
                if torch.equal(p.to(f.dtype), f):
                    checked += 1
                else:
                    mismatched.append(key)

    print(f"  owned-expert tensors verified: {checked}")
    if mismatched:
        print(f"  first 5 owned-expert mismatches: {mismatched[:5]}")
    if missing_full:
        print(f"  pretrained-side keys not found (skipped): {len(missing_full)}")

    _assert(
        checked > 0,
        "at least one owned-expert tensor matches the pretrained slice",
    )
    _assert(
        not mismatched,
        f"every checked owned-expert equals the pretrained slice (got {len(mismatched)} mismatches)",
    )


def _run_forward_pass(
    partial_model: CustomDeekSeekMoE,
    config: MinerConfig,
) -> None:
    """Run a single CPU forward pass through the partial model.

    This is a structural smoke test: the partial model only owns the
    experts for one group, and `myceliaSparseMoeBlock` silently skips
    tokens routed to unowned experts (`local_expert_idx == -1`), so
    the resulting logits are only partially informed. We assert the
    model is runnable end-to-end and produces a finite logits tensor
    of the expected `(batch, seq, vocab)` shape — not that the output
    is semantically meaningful."""
    tokenizer = get_base_tokenizer(config)
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"  prompt        = {prompt!r}")
    print(f"  input_ids     = shape {tuple(input_ids.shape)}, dtype {input_ids.dtype}")

    partial_model.eval()
    with torch.no_grad():
        outputs = partial_model(input_ids=input_ids)

    logits = outputs.logits
    print(f"  logits        = shape {tuple(logits.shape)}, dtype {logits.dtype}")

    expected_shape = (
        input_ids.shape[0],
        input_ids.shape[1],
        partial_model.config.vocab_size,
    )
    _assert(
        tuple(logits.shape) == expected_shape,
        f"logits shape is (batch, seq, vocab) = {expected_shape}",
    )
    _assert(
        bool(torch.isfinite(logits).all().item()),
        "all logits are finite (no NaN / Inf)",
    )


def _check_not_random(partial_state: dict) -> None:
    """Sanity: HuggingFace `_init_weights` for DeepSeek uses
    `normal_(std=config.initializer_range)` (~0.02). The pretrained
    embedding table has a noticeably different distribution. If the
    fix regressed and weights came back random, every backbone std
    would sit very close to 0.02."""
    key = "model.embed_tokens.weight"
    _assert(key in partial_state, f"partial state dict contains `{key}`")
    std = float(partial_state[key].detach().float().std().item())
    print(f"  embed_tokens.weight std = {std:.4f}")
    _assert(
        abs(std - 0.02) > 0.005,
        "embed_tokens.weight std differs from the random `_init_weights` default of ~0.02",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", type=int, default=GROUP_ID)
    args = parser.parse_args()

    if not EXP_DUMMY_DIR.is_dir():
        print(f"ERROR: expert_groups/exp_dummy not found at {EXP_DUMMY_DIR}")
        return 1

    print("=" * 72)
    print("Real integration test: get_base_model(partial=True)")
    print(f"  model_path   = {MODEL_PATH}")
    print(f"  group_id     = {args.group_id}")
    print(f"  fixture      = {EXP_DUMMY_DIR}")
    print( "  device       = cpu (VRAM untouched)")
    print("=" * 72)

    print("\n[1/5] Building MinerConfig + ExpertManager (real, from disk fixture)")
    config = _build_config(group_id=args.group_id)
    expert_manager = ExpertManager(config)
    n_groups = expert_manager.num_expert_groups
    print(f"  expert_group_assignment loaded ({n_groups} group(s))")
    _assert(
        args.group_id in expert_manager.expert_group_assignment,
        f"expert_manager.expert_group_assignment contains group_id={args.group_id}",
    )

    print("\n[2/5] Calling get_base_model(partial=True)")
    partial_model = get_base_model(
        config=config,
        expert_manager=expert_manager,
        group_ids_trainable=[args.group_id],
        group_ids_helper=None,
        partial=True,
    )
    _assert(partial_model is not None, "get_base_model returned a model")
    _assert(
        isinstance(partial_model, CustomDeekSeekMoE),
        f"model is CustomDeekSeekMoE (got {type(partial_model).__name__})",
    )

    partial_state = partial_model.state_dict()
    print(f"  partial model state_dict keys: {len(partial_state)}")

    print("\n[3/5] Sanity-check: weights are not at the random `_init_weights` default")
    _check_not_random(partial_state)

    print("\n[4/5] Comparing partial model tensors against the HF pretrained checkpoint")
    print("  Loading pretrained state dict in bf16 on CPU (first run pulls ~16 GB)...")
    full_state = load_pretrained_state_dict(MODEL_PATH, dtype=torch.bfloat16)
    print(f"  pretrained state dict keys: {len(full_state)}")

    print("\n  --- Backbone ---")
    _compare_backbone(partial_state, full_state)

    print("\n  --- Owned experts ---")
    _compare_owned_experts(
        partial_model=partial_model,
        full_state=full_state,
        expert_group_assignment=expert_manager.expert_group_assignment,
        target_group=args.group_id,
    )

    # Free the pretrained state dict before the forward pass so peak RAM
    # stays bounded — we no longer need it once comparisons are done.
    del full_state

    print("\n[5/5] Forward pass smoke test (CPU)")
    _run_forward_pass(partial_model, config)

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)

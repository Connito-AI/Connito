"""Does the FIXED full-model path load real pretrained experts?

`gpu_full_load_check.py` reproduces the *bug*: it calls
`load_state_dict(strict=False)` directly, which is what the full branch used to
do. This one exercises the *fix* — `get_base_model(partial=False)`, the real
entry point, which now streams safetensors through
`assignments_from_expert_modules`.

Ground truth is read tensor-by-tensor with `safe_open` rather than by loading a
second full `state_dict`. The old harness needed ~60 GB host RAM precisely
because it held the model and a complete state_dict at once; there is no reason
to pay that to compare four tensors.

Two independent signals, because neither alone is conclusive:

  1. **Weights.** Three expert projections and one backbone tensor (the
     control) compared against the checkpoint. An exact match on the experts is
     the decisive result — stronger than any loss number, which can be moved by
     dtype, routing width or data.
  2. **Loss.** A forward pass over fixed text. Not comparable to production
     `val_loss` (different data), but it separates "loaded" from "random" by an
     order of magnitude, and that is the claim under test.

    python tools/quantization/gpu_full_fix_verify.py [--device cuda] [--layer 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import banner, build_config, host_ram_hwm_gb  # noqa: E402

PROBE_LAYER = 1
PROBE_EXPERT = 0

# Enough tokens for a stable number, short enough to run in seconds. The text
# is arbitrary — only the loaded-vs-random contrast is being read off it.
PROBE_TEXT = (
    "The theory of general relativity describes gravity as a geometric "
    "property of spacetime. Mass and energy curve spacetime, and that "
    "curvature determines how objects move. In the weak-field limit the "
    "equations reduce to Newtonian gravity, which is why the older theory "
    "remained accurate for centuries of astronomical observation. "
) * 8


def _ground_truth(model_path: str, keys: list[str]) -> dict[str, torch.Tensor]:
    """Pull named tensors straight out of the checkpoint shards.

    Uses the resolved snapshot on disk so this never re-downloads 30 GB to
    check four tensors.
    """
    import json as _json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(model_path, allow_patterns=["*.json"]))
    index = _json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    wanted_shards: dict[str, list[str]] = {}
    for key in keys:
        shard = weight_map.get(key)
        if shard is not None:
            wanted_shards.setdefault(shard, []).append(key)

    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in wanted_shards.items():
        shard_path = Path(snapshot_download(model_path, allow_patterns=[shard])) / shard
        with safe_open(shard_path, framework="pt") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=PROBE_LAYER)
    parser.add_argument("--expert", type=int, default=PROBE_EXPERT)
    parser.add_argument("--device", default="cuda", help="device for the forward pass")
    parser.add_argument("--skip-loss", action="store_true", help="weights check only")
    args = parser.parse_args()

    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.mycelia import get_base_model

    cfg = build_config()
    model_path = cfg.model.model_path

    banner("Building the full model through the REAL entry point")
    print(f"get_base_model(partial=False)  full_topk={cfg.moe.full_topk}  "
          f"precision={cfg.model.precision}", flush=True)
    expert_manager = ExpertManager(cfg)
    model = get_base_model(
        cfg,
        expert_manager=expert_manager,
        group_ids_trainable=None,
        group_ids_helper=None,
        partial=False,
    )
    print(f"built, host hwm {host_ram_hwm_gb():.1f} GB", flush=True)

    layer, expert = args.layer, args.expert
    gate_key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    up_key = f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
    down_key = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
    backbone_key = f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"

    banner("Reading ground truth out of the checkpoint shards")
    truth = _ground_truth(model_path, [gate_key, up_key, down_key, backbone_key])
    print(f"probe keys found: {sorted(truth)}", flush=True)

    banner("Did the pretrained expert weights arrive?")
    experts = model.model.layers[layer].mlp.experts
    inter = experts.intermediate_dim
    stacked = experts.gate_up_proj.data
    local = int(experts.global_to_local_map[expert])
    print(f"global expert {expert} -> local slot {local}; "
          f"gate_up_proj {tuple(stacked.shape)}")

    checks: dict[str, dict] = {}

    def compare(label: str, got: torch.Tensor, want: torch.Tensor) -> None:
        got_f = got.detach().float().cpu()
        want_f = want.detach().float().cpu()
        delta = (got_f - want_f).abs().max().item()
        checks[label] = {
            "max_abs_delta": delta,
            "matches": delta == 0.0,
            "got_absmean": got_f.abs().mean().item(),
            "want_absmean": want_f.abs().mean().item(),
        }
        print(f"{label:<24} {'MATCH' if delta == 0.0 else 'MISMATCH'}  "
              f"max|delta|={delta:.6g}  |got|={checks[label]['got_absmean']:.6g}  "
              f"|want|={checks[label]['want_absmean']:.6g}")

    if gate_key in truth:
        compare("expert gate_proj", stacked[local, :inter, :], truth[gate_key])
    if up_key in truth:
        compare("expert up_proj", stacked[local, inter:, :], truth[up_key])
    if down_key in truth:
        compare("expert down_proj", experts.down_proj.data[local], truth[down_key])
    if backbone_key in truth:
        compare(
            "backbone (control)",
            model.model.layers[layer].self_attn.kv_a_proj_with_mqa.weight.data,
            truth[backbone_key],
        )

    experts_loaded = all(v["matches"] for k, v in checks.items() if k.startswith("expert"))

    loss_val = None
    if not args.skip_loss:
        banner(f"Forward pass on {args.device}")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        ids = tok(PROBE_TEXT, return_tensors="pt").input_ids
        print(f"tokens: {ids.numel()}", flush=True)

        model = model.to(args.device).eval()
        ids = ids.to(args.device)
        with torch.no_grad():
            loss_val = float(model(input_ids=ids, labels=ids).loss)
        print(f"loss = {loss_val:.4f}")
        print(
            "  a loaded 15 B model lands in single digits low; "
            "randomly-initialised experts push this toward ln(vocab) ~ 11.6"
        )

    print("\nRESULT " + json.dumps({
        "experts_loaded": experts_loaded,
        "backbone_loaded": checks.get("backbone (control)", {}).get("matches"),
        "loss": loss_val,
        "checks": checks,
        "host_ram_hwm_gb": round(host_ram_hwm_gb(), 2),
    }))


if __name__ == "__main__":
    main()

"""Does the FULL model path actually load the pretrained routed experts?

`load_pretrained_model_low_mem`'s full branch is one `load_state_dict(sd,
strict=False)` against the HuggingFace checkpoint. But HF stores experts as
`...experts.{i}.gate_proj.weight` / `up_proj.weight` / `down_proj.weight`, while
`CustomDeepseekV2Experts` serialises the *fused* `...experts.{i}.gate_up_proj`
and `...experts.{i}.down_proj` (no `.weight`). If those names don't meet,
`strict=False` swallows it and every routed expert stays at `_init_weights`
random — which would explain a near-random val_loss.

This checks it three ways, so the answer doesn't rest on key names alone:
  1. `load_state_dict` incompatible-key report, split into expert / non-expert.
  2. A pretrained expert tensor from the checkpoint vs the corresponding slice
     of the built model.
  3. A backbone tensor, same comparison — the control that proves the loader
     works at all and that this is specific to the experts.

CPU only; no GPU allocation, no eval. Peak host RAM ~60 GB (model + state_dict).

    python tools/quantization/gpu_full_load_check.py
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


def main() -> None:
    # An argparse pass even though there is little to configure: without one,
    # `--help` falls straight through into a 60 GB host-RAM build.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=PROBE_LAYER, help="MoE layer to probe")
    parser.add_argument("--expert", type=int, default=PROBE_EXPERT, help="global expert id to probe")
    args = parser.parse_args()
    probe_layer, probe_expert = args.layer, args.expert

    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeekSeekMoE
    from connito.shared.modeling.mycelia import (
        get_moe_model_config,
        load_pretrained_state_dict,
    )

    cfg = build_config()
    dtype = torch.bfloat16

    banner("Building the full model (CPU) exactly as the repo's full branch does")
    expert_manager = ExpertManager(cfg)
    moe_config = get_moe_model_config(
        cfg, cfg.moe.full_topk, None, None, expert_manager, full=True,
    )
    model = CustomDeekSeekMoE(moe_config).to(dtype=dtype)
    print(f"model built, host hwm {host_ram_hwm_gb():.1f} GB", flush=True)

    banner("Loading the pretrained state_dict")
    sd = load_pretrained_state_dict(cfg.model.model_path, dtype=dtype)
    print(f"pretrained keys: {len(sd)}, host hwm {host_ram_hwm_gb():.1f} GB", flush=True)

    expert_key_names = sorted(k for k in sd if ".mlp.experts." in k)
    print(f"pretrained expert keys: {len(expert_key_names)}")
    print(f"  sample: {expert_key_names[:3]}")

    # Ground truth, captured BEFORE load_state_dict — the custom
    # `_load_from_state_dict` pops entries out of the dict it is handed.
    gate_key = f"model.layers.{probe_layer}.mlp.experts.{probe_expert}.gate_proj.weight"
    up_key = f"model.layers.{probe_layer}.mlp.experts.{probe_expert}.up_proj.weight"
    down_key = f"model.layers.{probe_layer}.mlp.experts.{probe_expert}.down_proj.weight"
    backbone_key = f"model.layers.{probe_layer}.self_attn.kv_a_proj_with_mqa.weight"
    truth = {
        k: sd[k].clone() for k in (gate_key, up_key, down_key, backbone_key) if k in sd
    }
    print(f"probe keys present in checkpoint: {sorted(truth)}")

    banner("load_state_dict(strict=False)")
    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    missing_expert = [k for k in missing if ".mlp.experts." in k]
    unexpected_expert = [k for k in unexpected if ".mlp.experts." in k]
    print(f"missing    {len(missing)} ({len(missing_expert)} expert)")
    print(f"  sample   {missing[:4]}")
    print(f"unexpected {len(unexpected)} ({len(unexpected_expert)} expert)")
    print(f"  sample   {unexpected[:4]}")
    del sd

    banner("Did the weights actually arrive?")
    experts = model.model.layers[probe_layer].mlp.experts
    inter = experts.intermediate_dim
    stacked = experts.gate_up_proj.data
    local = int(experts.global_to_local_map[probe_expert])
    print(f"local slot for global expert {probe_expert}: {local}")
    print(f"gate_up_proj shape {tuple(stacked.shape)}, intermediate_dim {inter}")

    checks: dict[str, dict] = {}

    def compare(label: str, got: torch.Tensor, want: torch.Tensor) -> None:
        got_f, want_f = got.float(), want.float()
        delta = (got_f - want_f).abs().max().item()
        checks[label] = {
            "max_abs_delta": delta,
            "matches": delta == 0.0,
            "got_absmean": got_f.abs().mean().item(),
            "want_absmean": want_f.abs().mean().item(),
        }
        verdict = "MATCH" if delta == 0.0 else "MISMATCH"
        print(
            f"{label:<28} {verdict}  max|delta|={delta:.6g}  "
            f"|got|={checks[label]['got_absmean']:.6g} |want|={checks[label]['want_absmean']:.6g}"
        )

    if gate_key in truth:
        compare("expert gate_proj", stacked[local, :inter, :], truth[gate_key])
    if up_key in truth:
        compare("expert up_proj", stacked[local, inter:, :], truth[up_key])
    if down_key in truth:
        compare("expert down_proj", experts.down_proj.data[local], truth[down_key])
    if backbone_key in truth:
        got = model.model.layers[probe_layer].self_attn.kv_a_proj_with_mqa.weight.data
        compare("backbone (control)", got, truth[backbone_key])

    result = {
        "missing_total": len(missing),
        "missing_expert": len(missing_expert),
        "unexpected_total": len(unexpected),
        "unexpected_expert": len(unexpected_expert),
        "unexpected_expert_sample": unexpected_expert[:4],
        "checks": checks,
        "experts_loaded": all(
            v["matches"] for k, v in checks.items() if k.startswith("expert")
        ),
        "backbone_loaded": checks.get("backbone (control)", {}).get("matches"),
        "host_ram_hwm_gb": round(host_ram_hwm_gb(), 2),
    }
    print(f"\nRESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()

"""Shared setup for the fp8 GPU verification scripts in this directory.

Self-contained on purpose. These scripts run on a bare GPU box with no wallet
and no chain access, so the config builder here mirrors
`connito/test/test_get_base_model_partial.py::_build_config` — the repo's
offline model-construction precedent — rather than loading a validator config.

Every comparison these scripts make is *paired*: both arms draw the same seeded
batches from the same model, so absolute losses are not comparable with a
production round's `val_loss`, but the fp16-vs-fp8 delta is.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from connito.shared.config import MinerConfig  # noqa: E402

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite"
GROUP_ID = 4
HELPER_GROUP_ID = 2
EXP_DIR = REPO / "expert_groups" / "exp_nemotron_c4"

# Fixed so both arms of every comparison draw identical eval batches. Not a seed
# any production round used, so absolute losses will not line up with a specific
# round's val_loss — irrelevant for a paired comparison.
EVAL_SEED = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00"


def drop_gated_sources(cfg) -> list[str]:
    """Drop `nvidia/Nemotron-CC-Math-v1` from the eval mix.

    It is gated: file reads 403 without an HF token whose account accepted the
    licence. Dropping it makes absolute losses non-comparable with production
    (the real mix is 50/50 C4 + Nemotron-Math), but every comparison here is
    paired — both arms draw identical batches from whatever remains — so the
    fp8-vs-fp16 deltas stay valid. Prefer running with a licensed token: math
    text is a different distribution and could carry a different quantization
    error.
    """
    sources = cfg.task.exp.data.dataset_sources or []
    dropped = [s.path for s in sources if "Nemotron" in s.path]
    kept = [s for s in sources if "Nemotron" not in s.path]
    for source in kept:
        source.weight = 1.0
    cfg.task.exp.data.dataset_sources = kept
    return dropped


def build_config(seq_len: int = 1024, quantization: str = "off") -> MinerConfig:
    # ss58 fields prepopulated so `model_post_init` skips `_fill_wallet_data`,
    # which would otherwise open a subtensor connection on a box with no wallet.
    cfg = MinerConfig(chain={"hotkey_ss58": "offline", "coldkey_ss58": "offline"})
    cfg.model.model_path = MODEL_PATH
    cfg.model.base_arch_model = MODEL_PATH
    cfg.model.device = "cuda"
    cfg.model.precision = "bf16-mixed"
    cfg.model.torch_compile = False
    cfg.model.quantization = quantization
    cfg.task.expert_group_name = "exp_nemotron_c4"
    cfg.task.base_path = EXP_DIR.parent
    cfg.task.path = EXP_DIR
    cfg.task.load_all_expert_groups = False
    cfg.task.exp.group_id = GROUP_ID
    cfg.task.helper_group_id = HELPER_GROUP_ID
    cfg.task.exp.data.sequence_length = seq_len
    cfg.task.exp.data.__dict__["num_workers"] = 0
    return cfg


def build_partial_model(cfg):
    """The model a miner and (today) a validator build: one trainable expert
    group plus one frozen helper group, `partial_topk` routing."""
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.mycelia import get_base_model

    expert_manager = ExpertManager(cfg)
    started = time.time()
    model = get_base_model(
        cfg,
        expert_manager,
        group_ids_trainable=[GROUP_ID],
        group_ids_helper=[HELPER_GROUP_ID],
        partial=True,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model, expert_manager, time.time() - started


def vram_gb() -> float:
    """Resident VRAM, GiB (torch's own accounting — excludes the CUDA context)."""
    return torch.cuda.memory_allocated() / 1024**3


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_peak() -> None:
    torch.cuda.reset_peak_memory_stats()


def host_ram_hwm_gb() -> float:
    """Peak resident host RAM for this process, GiB, from VmHWM.

    Load-bearing for the full-model path, which materialises a CPU model *and* a
    complete pretrained state_dict before copying between them.
    """
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024**2
    return float("nan")


def host_ram_now_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024**2
    return float("nan")


def eval_batches(cfg, tokenizer, n_batches: int):
    """Materialise a fixed set of eval batches, identical across both arms."""
    from connito.shared.dataloader import get_dataloader, materialize_batches

    loader = get_dataloader(cfg, rank=0, world_size=1, tokenizer=tokenizer, seed=EVAL_SEED)
    return materialize_batches(loader, n_batches)


def score(model, batches, device) -> float:
    """val_loss through the validator's own eval path."""
    from connito.shared.evaluate import evaluate_model

    metrics = evaluate_model(
        step=0, model=model, eval_dataloader=batches, device=device, max_eval_batches=len(batches)
    )
    return float(metrics["val_loss"])


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


os.environ.setdefault("ENABLE_TELEMETRY", "0")

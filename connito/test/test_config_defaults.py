"""Tests for the inner-optimizer / schedule default values.

These defaults drive the per-miner AdamW + cosine LR schedule used by
`connito/miner/train.py`. The goal of the bundle is a more aggressive
but still safe baseline tuned for the delta-driven validator score
defined in `connito/validator/evaluator.py` — small wins in
`miner_val_loss` translate non-linearly into score, so the defaults
matter.

If anybody intentionally re-tunes these defaults, update the asserts —
the test exists so an *accidental* revert is caught loudly.
"""

from connito.shared.config import MinerConfig, OptimizerCfg, ScheduleCfg


def test_optimizer_defaults() -> None:
    """Bundle of inner-optimizer defaults (Unit 16/18)."""
    cfg = MinerConfig()

    # Higher LR baseline for MoE fine-tuning on streaming data.
    assert cfg.opt.lr == 3e-5

    # Standard AdamW betas — replaces unusual (0.9, 0.95).
    assert cfg.opt.betas == (0.9, 0.999)

    # bf16/fp16-friendly epsilon.
    assert cfg.opt.eps == 1e-6

    # Conservative grad clipping pairs with the higher LR.
    assert cfg.opt.grad_clip_max_norm == 0.5

    # Weight decay lifted from train.py into config.
    assert cfg.opt.weight_decay == 0.1

    # 500-step warmup pairs with the higher LR to avoid early-step spikes.
    assert cfg.sched.warmup_steps == 500


def test_optimizer_cfg_standalone_defaults() -> None:
    """The defaults must hold even when OptimizerCfg is built directly,
    independent of MinerConfig wiring."""
    opt = OptimizerCfg()
    assert opt.lr == 3e-5
    assert opt.betas == (0.9, 0.999)
    assert opt.eps == 1e-6
    assert opt.grad_clip_max_norm == 0.5
    assert opt.weight_decay == 0.1
    # Outer-optimizer defaults should be unchanged by this bundle.
    assert opt.outer_lr == 0.7
    assert opt.outer_momentum == 0.9


def test_schedule_cfg_standalone_defaults() -> None:
    """Schedule defaults independent of MinerConfig wiring."""
    sched = ScheduleCfg()
    assert sched.warmup_steps == 500
    assert sched.total_steps == 88_000


def test_betas_is_tuple_of_floats() -> None:
    """Pydantic should coerce list-style YAML input to a tuple of floats."""
    opt = OptimizerCfg(betas=[0.9, 0.999])
    assert isinstance(opt.betas, tuple)
    assert opt.betas == (0.9, 0.999)
    assert all(isinstance(x, float) for x in opt.betas)


def test_operator_overrides_take_precedence() -> None:
    """Defaults are advisory — explicit operator overrides must win."""
    opt = OptimizerCfg(lr=5e-5, eps=1e-8, grad_clip_max_norm=1.0)
    assert opt.lr == 5e-5
    assert opt.eps == 1e-8
    assert opt.grad_clip_max_norm == 1.0
    # Untouched defaults remain.
    assert opt.betas == (0.9, 0.999)
    assert opt.weight_decay == 0.1

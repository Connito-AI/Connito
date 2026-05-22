"""Unit tests for connito.miner.ema_tracker.ModelEma.

Covers:
  - shadow init mirrors the model params
  - single-step decay math: shadow = decay * θ_init + (1 - decay) * θ_new
  - multi-step EMA decay math accumulates correctly
  - only_trainable filter respects requires_grad
  - apply_to / restore round-trip preserves weights exactly
  - state_dict round-trip via load_state_dict
  - default config disables EMA so existing behavior is unchanged
"""

from __future__ import annotations

import torch
import torch.nn as nn

from connito.miner.ema_tracker import ModelEma
from connito.shared.config import EmaCfg, MinerConfig


def _make_tiny_model(seed: int = 0) -> nn.Module:
    """A deterministic, tiny linear stack — fast to iterate in tests."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(8, 8, bias=False),
        nn.Linear(8, 4, bias=False),
    )
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached clones of every parameter tensor, keyed by name."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_init_matches_live_params():
    model = _make_tiny_model(seed=1)
    ema = ModelEma(model, decay=0.999)

    live = _snapshot(model)
    assert set(ema.shadow.keys()) == set(live.keys()), "shadow keys must match live"
    for name, live_p in live.items():
        # Initial shadow equals the live params (clone, not view).
        assert torch.equal(ema.shadow[name], live_p), f"shadow[{name}] should equal live param"
        assert ema.shadow[name].data_ptr() != live_p.data_ptr(), "shadow must be a clone"


def test_single_step_decay_math():
    """After one update: shadow = decay * θ_init + (1 - decay) * θ_new."""
    decay = 0.9
    model = _make_tiny_model(seed=2)
    theta_init = _snapshot(model)

    ema = ModelEma(model, decay=decay)

    # Mutate the live model to simulate one optimizer step.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))  # θ_new = θ_init + 1
    theta_new = _snapshot(model)

    ema.update(model)

    for name, init_p in theta_init.items():
        expected = decay * init_p + (1.0 - decay) * theta_new[name]
        assert torch.allclose(ema.shadow[name], expected, atol=1e-6), (
            f"EMA math wrong for {name}: got {ema.shadow[name]} expected {expected}"
        )
    assert ema.num_updates == 1


def test_multi_step_decay_math():
    """10 updates with decay=0.5 should match the analytical recurrence.

    After N updates starting from θ_0 with live param frozen at θ:
        shadow_N = decay^N * θ_0 + (1 - decay^N) * θ
    when θ_live is held constant. We verify with θ = 0 (so shadow_N = decay^N * θ_0)
    and a non-zero θ_0, which makes the test robust to floating-point drift.
    """
    decay = 0.5
    n_updates = 10
    model = _make_tiny_model(seed=3)

    # Capture initial param tensors as θ_0.
    theta0 = _snapshot(model)

    ema = ModelEma(model, decay=decay)

    # Force the live params to all-zeros, then update N times.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    for _ in range(n_updates):
        ema.update(model)

    expected_factor = decay ** n_updates
    for name, init_p in theta0.items():
        expected = expected_factor * init_p
        assert torch.allclose(ema.shadow[name], expected, atol=1e-6), (
            f"Multi-step EMA math wrong for {name} after {n_updates} updates: "
            f"got {ema.shadow[name]} expected {expected}"
        )
    assert ema.num_updates == n_updates


def test_only_trainable_filter_respects_requires_grad():
    model = _make_tiny_model(seed=4)
    # Freeze the first layer.
    for name, p in model.named_parameters():
        if name.startswith("0."):
            p.requires_grad_(False)

    ema = ModelEma(model, decay=0.999, only_trainable=True)

    expected_tracked = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(ema.shadow.keys()) == expected_tracked, (
        f"only_trainable=True should track only requires_grad params; "
        f"got {set(ema.shadow.keys())} vs expected {expected_tracked}"
    )

    # Mutate ALL params (including frozen ones).
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))

    ema.update(model)
    # Shadow should not have grown to include the frozen params.
    assert set(ema.shadow.keys()) == expected_tracked


def test_only_trainable_false_tracks_everything():
    model = _make_tiny_model(seed=5)
    for name, p in model.named_parameters():
        if name.startswith("0."):
            p.requires_grad_(False)

    ema = ModelEma(model, decay=0.999, only_trainable=False)

    all_names = {name for name, _ in model.named_parameters()}
    assert set(ema.shadow.keys()) == all_names


def test_apply_to_restore_roundtrip():
    """apply_to() must swap in EMA values; restore() must put live params back exactly."""
    model = _make_tiny_model(seed=6)
    theta_init = _snapshot(model)
    ema = ModelEma(model, decay=0.5)

    # Move the model away from init so EMA != live after the update.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 0.25)
    ema.update(model)
    theta_live = _snapshot(model)
    theta_ema = {name: t.clone() for name, t in ema.state_dict().items()}

    # Swap in EMA, verify model now holds EMA values.
    original = ema.apply_to(model)
    for name, p in model.named_parameters():
        assert torch.allclose(p, theta_ema[name]), f"apply_to did not install EMA for {name}"
    # original must equal theta_live (pre-swap).
    for name, saved in original.items():
        assert torch.equal(saved, theta_live[name]), f"original snapshot wrong for {name}"

    # Restore and verify exact match with the live params we captured before swapping.
    ema.restore(model, original)
    for name, p in model.named_parameters():
        assert torch.equal(p, theta_live[name]), (
            f"restore() did not return param {name} to live values exactly"
        )
    # Sanity: model should NOT equal theta_init (we modified it before update).
    diffs = [
        name for name, p in model.named_parameters()
        if not torch.equal(p, theta_init[name])
    ]
    assert diffs, "test premise violated: model never changed from init"


def test_state_dict_roundtrip_via_load():
    model = _make_tiny_model(seed=7)
    ema = ModelEma(model, decay=0.9)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.5)
    ema.update(model)

    state = ema.state_dict()
    # Build a fresh EMA from a fresh model and load the saved state.
    fresh_model = _make_tiny_model(seed=7)
    fresh_ema = ModelEma(fresh_model, decay=0.9)
    fresh_ema.load_state_dict(state)

    for name in state:
        assert torch.equal(fresh_ema.shadow[name], state[name]), (
            f"load_state_dict did not restore {name}"
        )


def test_decay_bounds_validation():
    model = _make_tiny_model(seed=8)
    try:
        ModelEma(model, decay=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("decay=-0.1 should raise ValueError")

    try:
        ModelEma(model, decay=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("decay=1.5 should raise ValueError")


def test_update_skips_unknown_params():
    """Adding a param to the model after init must not break update()."""
    model = _make_tiny_model(seed=9)
    ema = ModelEma(model, decay=0.999)

    # Inject a new module after init; update should silently ignore it.
    model.add_module("extra", nn.Linear(4, 2, bias=False))
    # extra params get random init by default — make them trainable.
    for p in model.extra.parameters():
        p.requires_grad_(True)

    # Should not raise.
    ema.update(model)
    # The extra module's params must NOT have been added to the shadow.
    assert not any(name.startswith("extra.") for name in ema.shadow), (
        "update() should not add new params to the shadow"
    )


def test_emacfg_defaults_disable_ema():
    """Default EmaCfg must leave behavior unchanged (enabled=False)."""
    cfg = EmaCfg()
    assert cfg.enabled is False
    assert cfg.commit_ema_snapshot is False
    # Sanity on the sane defaults.
    assert 0.0 <= cfg.decay <= 1.0
    assert cfg.update_every_n_steps >= 1


def test_minerconfig_includes_ema_section_off_by_default():
    """MinerConfig should expose `ema` with EmaCfg defaults (enabled=False)."""
    # Avoid touching the chain: MinerConfig's model_post_init tries to look
    # up wallet data; we pass pre-filled values so `_fill_wallet_data`
    # short-circuits.
    cfg = MinerConfig(
        chain={
            "hotkey_ss58": "5" + "1" * 47,
            "coldkey_ss58": "5" + "1" * 47,
        },
    )
    assert hasattr(cfg, "ema"), "MinerConfig must expose an `ema` section"
    assert isinstance(cfg.ema, EmaCfg)
    assert cfg.ema.enabled is False, "ema must default to disabled"
    assert cfg.ema.commit_ema_snapshot is False


def test_build_ema_only_on_rank_zero():
    """Non-zero ranks must not allocate an EMA shadow."""
    from types import SimpleNamespace
    from connito.miner.train import _build_ema_if_enabled

    cfg = SimpleNamespace(
        ema=SimpleNamespace(enabled=True, decay=0.999, only_trainable_params=True, shadow_on_cpu=False),
    )
    m = _make_tiny_model(seed=11)
    assert _build_ema_if_enabled(cfg, m, rank=0) is not None
    assert _build_ema_if_enabled(cfg, m, rank=1) is None
    assert _build_ema_if_enabled(cfg, m, rank=7) is None


def test_save_and_load_ema_shadow_roundtrip(tmp_path):
    """Saving + reloading the EMA shadow file must preserve every tensor exactly."""
    from connito.miner.train import _save_ema_shadow, _try_load_ema_shadow
    from connito.miner.ema_tracker import EMA_SHADOW_FILENAME

    model = _make_tiny_model(seed=12)
    ema = ModelEma(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3)
    ema.update(model)
    expected = ema.state_dict()

    _save_ema_shadow(ema, str(tmp_path))
    assert (tmp_path / EMA_SHADOW_FILENAME).exists()

    # Fresh EMA → restore from disk → must match
    fresh_model = _make_tiny_model(seed=12)
    fresh_ema = ModelEma(fresh_model, decay=0.5)
    _try_load_ema_shadow(fresh_ema, str(tmp_path))
    for name, t in expected.items():
        assert torch.equal(fresh_ema.shadow[name], t), f"persisted EMA mismatch on {name}"


def test_try_load_ema_shadow_missing_file_is_noop(tmp_path):
    """When no EMA shadow file exists on disk, _try_load_ema_shadow must leave the shadow untouched."""
    from connito.miner.train import _try_load_ema_shadow

    model = _make_tiny_model(seed=13)
    ema = ModelEma(model, decay=0.5)
    before = ema.state_dict()
    # tmp_path is empty.
    _try_load_ema_shadow(ema, str(tmp_path))
    for name, t in before.items():
        assert torch.equal(ema.shadow[name], t), f"shadow changed despite no file for {name}"


def test_materialize_ema_snapshot_matches_live_shard_dtype(tmp_path):
    """When a live shard exists in the checkpoint dir, the staged EMA file
    is downcast to that shard's dtype so upload size is preserved."""
    from pathlib import Path
    from connito.miner.model_io import _materialize_ema_snapshot_dir
    from connito.miner.ema_tracker import EMA_SHADOW_FILENAME
    from safetensors.torch import load_file, save_file

    # Build a fake EMA shadow (fp32) and a fake live shard (fp16).
    fake_shadow = {
        "model.layers.0.experts.0.w1.weight": torch.randn(8, 8, dtype=torch.float32),
    }
    save_file(fake_shadow, str(tmp_path / EMA_SHADOW_FILENAME))
    fake_live = {k: v.to(torch.float16) for k, v in fake_shadow.items()}
    save_file(fake_live, str(tmp_path / "model_expgroup_2.safetensors"))

    staged = _materialize_ema_snapshot_dir(Path(tmp_path), expert_group_id=2)
    assert staged is not None
    try:
        staged_state = load_file(str(staged / "model_expgroup_2.safetensors"))
        for v in staged_state.values():
            assert v.dtype == torch.float16, (
                f"staged EMA should match live shard dtype (fp16); got {v.dtype}"
            )
    finally:
        import shutil
        shutil.rmtree(staged, ignore_errors=True)


def test_materialize_ema_snapshot_missing_shadow_returns_none(tmp_path):
    """If the EMA shadow file is absent, _materialize_ema_snapshot_dir returns None
    (caller falls back to live checkpoint)."""
    from pathlib import Path
    from connito.miner.model_io import _materialize_ema_snapshot_dir

    staged = _materialize_ema_snapshot_dir(Path(tmp_path), expert_group_id=0)
    assert staged is None


def test_shadow_on_separate_device():
    """When device='cpu' is passed, shadow lives on CPU even if model on CPU.

    Smoke test that the device kwarg works and update() handles a
    device/dtype mismatch (here we just exercise the same-device CPU path
    so the test runs without a GPU).
    """
    model = _make_tiny_model(seed=10)
    ema = ModelEma(model, decay=0.5, device="cpu")
    for shadow_p in ema.shadow.values():
        assert shadow_p.device.type == "cpu"

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    # Mutate live params to fp32 ones; shadow update should not raise.
    ema.update(model)
    for shadow_p in ema.shadow.values():
        # After updating against an all-ones live param, shadow should
        # have moved toward 1.0 (just sanity, not exact math).
        assert torch.isfinite(shadow_p).all()

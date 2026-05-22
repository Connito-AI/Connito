"""Unit tests for the Flash Attention 2/3 auto-detection plumbing.

Covers `connito.shared.modeling.mycelia._resolve_attn_implementation` —
the helper that maps a `model.attn_implementation` config value (e.g.
"auto", "flash_attention_2") onto a concrete HuggingFace attention
backend string, with a safe fallback to "sdpa" when Flash Attention
isn't available.

These tests use mocking so they pass on CPU-only CI hosts without a
flash-attn install or an Ampere+ GPU. A separate forward-pass smoke
test runs only when CUDA is actually available.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from connito.shared.config import ModelCfg
from connito.shared.modeling import mycelia
from connito.shared.modeling.mycelia import _resolve_attn_implementation


# ---------------------------------------------------------------------
# `_resolve_attn_implementation` — mocked-environment table
# ---------------------------------------------------------------------
class _FakeSpec:
    """Stand-in for `importlib.util.find_spec`'s non-None return."""


def _fake_find_spec_present(_name: str) -> _FakeSpec | None:
    return _FakeSpec()


def _fake_find_spec_absent(_name: str) -> _FakeSpec | None:
    return None


def test_auto_ampere_plus_with_flash_attn_resolves_to_flash_attention_2():
    """`auto` + GPU compute cap >= 8.0 + flash_attn importable
    → resolves to `flash_attention_2`."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(8, 0)):
        assert _resolve_attn_implementation("auto") == "flash_attention_2"


def test_auto_ampere_plus_without_flash_attn_resolves_to_sdpa():
    """`auto` + Ampere+ GPU but flash_attn NOT importable
    → resolves to `sdpa` (no crash)."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_absent), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(8, 0)):
        resolved = _resolve_attn_implementation("auto")
    assert resolved == "sdpa"


def test_auto_turing_volta_with_flash_attn_resolves_to_sdpa():
    """`auto` + flash_attn importable but GPU compute cap < 8.0
    (e.g. Turing 7.5 or Volta 7.0) → resolves to `sdpa`."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(7, 5)):
        assert _resolve_attn_implementation("auto") == "sdpa"


def test_auto_no_cuda_resolves_to_sdpa():
    """`auto` + no CUDA → `sdpa`. Covers CPU-only dev/CI hosts."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=False):
        assert _resolve_attn_implementation("auto") == "sdpa"


def test_explicit_flash_attention_2_unavailable_falls_back_to_sdpa():
    """Explicit `flash_attention_2` + flash_attn NOT importable
    → falls back to `sdpa` (does not crash)."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_absent), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(8, 0)):
        assert _resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_explicit_flash_attention_2_on_pre_ampere_gpu_falls_back_to_sdpa():
    """Explicit `flash_attention_2` on Turing/Volta GPU
    → falls back to `sdpa` (does not crash)."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(7, 5)):
        assert _resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_explicit_flash_attention_2_available_resolves_unchanged():
    """Explicit `flash_attention_2` + flash_attn importable + Ampere+ GPU
    → stays `flash_attention_2`."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert _resolve_attn_implementation("flash_attention_2") == "flash_attention_2"


def test_explicit_flash_attention_3_unavailable_falls_back_to_sdpa():
    """Explicit `flash_attention_3` shares the same flash_attn import +
    GPU gates and falls back to `sdpa` when either gate fails."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_absent), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert _resolve_attn_implementation("flash_attention_3") == "sdpa"


def test_explicit_sdpa_stays_sdpa_regardless_of_environment():
    """Explicit `sdpa` is never auto-upgraded, even on a flash-capable
    host. This is the documented escape hatch for users hitting Flash
    Attention bugs."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert _resolve_attn_implementation("sdpa") == "sdpa"


def test_explicit_eager_stays_eager():
    """`eager` is the reference PyTorch path; it must never be silently
    swapped for `sdpa` or Flash."""
    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert _resolve_attn_implementation("eager") == "eager"


def test_find_spec_value_error_treated_as_unavailable():
    """`find_spec` raises `ValueError` for broken meta-path finders.
    Helper must catch that and treat flash_attn as unavailable rather
    than crashing the loader."""
    def _raises(_name: str):
        raise ValueError("broken finder")

    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_raises), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert _resolve_attn_implementation("auto") == "sdpa"


def test_get_device_capability_runtime_error_treated_as_unsupported():
    """A flaky CUDA driver can make `get_device_capability` raise. The
    helper must catch that, treat the GPU as unsupported, and resolve
    `auto` to `sdpa`."""
    def _raises():
        raise RuntimeError("CUDA driver hiccup")

    with patch.object(mycelia.importlib.util, "find_spec", side_effect=_fake_find_spec_present), \
         patch.object(mycelia.torch.cuda, "is_available", return_value=True), \
         patch.object(mycelia.torch.cuda, "get_device_capability", side_effect=_raises):
        assert _resolve_attn_implementation("auto") == "sdpa"


# ---------------------------------------------------------------------
# `ModelCfg.attn_implementation` validation
# ---------------------------------------------------------------------
def test_model_cfg_defaults_to_auto():
    """The miner-facing default must be `auto` so existing operators
    automatically opt into Flash Attention on supported hardware."""
    cfg = ModelCfg()
    assert cfg.attn_implementation == "auto"


@pytest.mark.parametrize(
    "value",
    ["auto", "sdpa", "flash_attention_2", "flash_attention_3", "eager"],
)
def test_model_cfg_accepts_supported_values(value: str):
    cfg = ModelCfg(attn_implementation=value)
    assert cfg.attn_implementation == value


def test_model_cfg_rejects_unknown_value():
    with pytest.raises(ValueError, match="attn_implementation"):
        ModelCfg(attn_implementation="bogus_kernel")


# ---------------------------------------------------------------------
# Live forward-pass smoke test (CUDA-only)
# ---------------------------------------------------------------------
def test_sdpa_forward_pass_smoke():
    """End-to-end sanity: load a tiny model with `sdpa` and run a
    forward pass. Skipped on hosts without CUDA — flash-attn doesn't
    matter for this check; we just want to confirm that the
    `attn_implementation` kwarg reaches HuggingFace without an error
    on the host actually running the test."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — forward pass smoke test skipped")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="sdpa"
    ).to("cuda").eval()

    inputs = tokenizer("hello world", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[-1] == model.config.vocab_size


if __name__ == "__main__":
    # Manual driver — handy when iterating locally without pytest.
    test_auto_ampere_plus_with_flash_attn_resolves_to_flash_attention_2()
    test_auto_ampere_plus_without_flash_attn_resolves_to_sdpa()
    test_auto_turing_volta_with_flash_attn_resolves_to_sdpa()
    test_auto_no_cuda_resolves_to_sdpa()
    test_explicit_flash_attention_2_unavailable_falls_back_to_sdpa()
    test_explicit_flash_attention_2_on_pre_ampere_gpu_falls_back_to_sdpa()
    test_explicit_flash_attention_2_available_resolves_unchanged()
    test_explicit_flash_attention_3_unavailable_falls_back_to_sdpa()
    test_explicit_sdpa_stays_sdpa_regardless_of_environment()
    test_explicit_eager_stays_eager()
    test_find_spec_value_error_treated_as_unavailable()
    test_get_device_capability_runtime_error_treated_as_unsupported()
    test_model_cfg_defaults_to_auto()
    for _v in ("auto", "sdpa", "flash_attention_2", "flash_attention_3", "eager"):
        test_model_cfg_accepts_supported_values(_v)
    test_model_cfg_rejects_unknown_value()
    print("All Flash Attention resolver tests passed.")

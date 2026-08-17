"""Models are constructed at their target dtype, not fp32-then-cast.

`nn.Parameter(torch.empty(...))` takes torch's *default* dtype. Building a model
and casting afterwards keeps the fp32 storage for every parameter live until
that parameter is individually replaced, so the peak is ~2x the final size.

Measured on the full DeepSeek-V2-Lite topology before the fix: 59.6 GB peak
resident for a ~31 GB model, on a 62 GB host. After: 38.0 GB, with the loaded
weights and the eval loss bit-identical.

`set_default_dtype` is process-global, so the restore matters as much as the
set — a leaked bf16 default would silently downgrade every tensor built later
in the process, including ones that are meant to be fp32.

Run with `python -m pytest connito/test/test_build_at_dtype.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from connito.shared.modeling.mycelia import build_at_dtype


def test_parameters_are_built_at_the_requested_dtype():
    """The point of the exercise: no fp32 allocation to reclaim later."""
    with build_at_dtype(torch.bfloat16):
        param = nn.Parameter(torch.empty(4, 4))
        linear = nn.Linear(4, 4)

    assert param.dtype is torch.bfloat16
    assert linear.weight.dtype is torch.bfloat16


def test_default_dtype_is_restored():
    before = torch.get_default_dtype()
    with build_at_dtype(torch.float16):
        assert torch.get_default_dtype() is torch.float16
    assert torch.get_default_dtype() is before


def test_default_dtype_is_restored_after_an_exception():
    """A leaked global default is worse than the bug being fixed.

    Construction can fail part-way — a missing config field, an OOM. Without
    the `finally`, everything built afterwards in the process silently inherits
    the reduced precision.
    """
    before = torch.get_default_dtype()
    with pytest.raises(RuntimeError):
        with build_at_dtype(torch.bfloat16):
            raise RuntimeError("construction failed")
    assert torch.get_default_dtype() is before


def test_nesting_restores_the_outer_dtype():
    before = torch.get_default_dtype()
    with build_at_dtype(torch.bfloat16):
        with build_at_dtype(torch.float16):
            assert torch.get_default_dtype() is torch.float16
        assert torch.get_default_dtype() is torch.bfloat16
    assert torch.get_default_dtype() is before


def test_a_trailing_cast_stays_a_no_op():
    """Why the call sites keep their `.to(dtype=...)`.

    It is the backstop for a submodule that hard-codes fp32, and it has to be
    free in the normal case or it would reintroduce the copy this removes.
    `Tensor.to` returns self when the dtype already matches.
    """
    with build_at_dtype(torch.bfloat16):
        linear = nn.Linear(4, 4)

    # Storage address, not object identity: `Tensor.data` hands back a fresh
    # Python wrapper on every access, so `is` would compare the wrappers and
    # fail even when nothing was copied.
    storage_before = linear.weight.data_ptr()
    linear = linear.to(dtype=torch.bfloat16)
    assert linear.weight.data_ptr() == storage_before

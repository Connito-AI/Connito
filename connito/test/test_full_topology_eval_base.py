"""Graft-in-place scoring on the full-topology eval base.

The base replaces the per-miner `copy.deepcopy(base_model)` — a second
~29 GiB model, the measured foreground OOM — with: validate the shard, back up
the rows it touches, overlay them in place, evaluate, restore. The load-bearing
properties, each pinned here:

  - restore leaves the base *bit-identical* to pre-graft, which is what every
    subsequent miner's baseline comparability rests on;
  - a shard writing anywhere the graft could not restore — another group's
    experts, the backbone — is rejected before any tensor is read, because an
    in-place write outside the backup would silently contaminate every later
    miner of the round (deepcopy isolation used to make that impossible);
  - a stale eval (orphaned by a per-miner timeout, queued behind the lock
    while the round moved on) fails its generation check without touching the
    base;
  - `refresh_from` carries the partial global model's merged rows into the
    full base through the existing per-expert-key overlay serializer.

Run with `python -m pytest connito/test/test_full_topology_eval_base.py`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from connito.shared.modeling.custom_deepseek_v2_lite import CustomDeepseekV2Experts
from connito.validator.full_topology_eval import (
    FullTopologyEvalBase,
    ShardRejected,
)

LAYER_ID = 1
PREFIX = f"model.layers.{LAYER_ID}.mlp.experts"
HIDDEN, INTER = 16, 12
ALL_IDS = [0, 1, 2, 3]
OWN_IDS = {0, 1}  # this validator's group; 2 and 3 belong to someone else


class _Cfg:
    hidden_size = HIDDEN
    moe_intermediate_size = INTER
    hidden_act = "silu"
    initializer_range = 0.02
    first_k_dense_replace = 1
    num_hidden_layers = 4

    def __init__(self, num_experts: int = 4) -> None:
        self.n_routed_experts = num_experts
        self.num_experts = num_experts


class _Mlp(nn.Module):
    def __init__(self, expert_indices, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.experts = CustomDeepseekV2Experts(_Cfg(), expert_indices=list(expert_indices))


class _Layer(nn.Module):
    def __init__(self, expert_indices, seed: int) -> None:
        super().__init__()
        self.mlp = _Mlp(expert_indices, seed)


class _Model(nn.Module):
    """Enough hierarchy for `model.layers.{L}.mlp.experts` names."""

    def __init__(self, expert_indices=ALL_IDS, seed: int = 0) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), _Layer(expert_indices, seed)])


def _base() -> FullTopologyEvalBase:
    return FullTopologyEvalBase(_Model(seed=0), allowed_by_layer={LAYER_ID: set(OWN_IDS)})


def _snapshot(base: FullTopologyEvalBase) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in base.model.state_dict().items()}


def _shard(tmp_path, tensors: dict[str, torch.Tensor]) -> str:
    path = tmp_path / "model_expgroup_4.safetensors"
    save_file(tensors, str(path))
    return str(path)


def _rows(base: FullTopologyEvalBase, gid: int) -> torch.Tensor:
    experts = base.model.get_submodule(PREFIX)
    return experts.gate_up_proj.data[int(experts.global_to_local_map[gid])]


# ── graft and restore ────────────────────────────────────────────────────────
def test_graft_writes_exactly_the_shard_rows(tmp_path):
    base = _base()
    before = _snapshot(base)
    new_rows = torch.full((2 * INTER, HIDDEN), 7.0)
    path = _shard(tmp_path, {f"{PREFIX}.0.gate_up_proj": new_rows})

    model = base.graft_from_path(path)

    assert model is base.model  # in place, not a copy
    assert torch.equal(_rows(base, 0), new_rows)
    # expert 1 (own group, untouched by this shard) is unchanged
    assert torch.equal(_rows(base, 1), before[f"{PREFIX}.1.gate_up_proj"])


def test_restore_is_bit_identical(tmp_path):
    base = _base()
    before = _snapshot(base)
    path = _shard(tmp_path, {
        f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0),
        f"{PREFIX}.1.down_proj": torch.full((HIDDEN, INTER), -3.0),
    })

    base.graft_from_path(path)
    base.restore_grafted()

    after = base.model.state_dict()
    assert set(after) == set(before)
    for key in before:
        assert torch.equal(after[key], before[key]), key


def test_second_graft_after_restore_sees_the_clean_base(tmp_path):
    base = _base()
    before = _snapshot(base)
    base.graft_from_path(_shard(tmp_path, {
        f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0),
    }))
    base.restore_grafted()

    # Miner B touches expert 1 only; expert 0 must be back to the original,
    # not miner A's 7.0s.
    base.graft_from_path(_shard(tmp_path, {
        f"{PREFIX}.1.gate_up_proj": torch.full((2 * INTER, HIDDEN), 9.0),
    }))
    assert torch.equal(_rows(base, 0), before[f"{PREFIX}.0.gate_up_proj"])


def test_graft_without_restore_is_refused(tmp_path):
    base = _base()
    path = _shard(tmp_path, {f"{PREFIX}.0.gate_up_proj": torch.zeros(2 * INTER, HIDDEN)})
    base.graft_from_path(path)
    with pytest.raises(RuntimeError, match="restore_grafted"):
        base.graft_from_path(path)


# ── the rejection surface ────────────────────────────────────────────────────
def test_foreign_group_expert_is_rejected_untouched(tmp_path):
    base = _base()
    before = _snapshot(base)
    path = _shard(tmp_path, {
        f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0),  # legal
        f"{PREFIX}.2.gate_up_proj": torch.full((2 * INTER, HIDDEN), 6.6),  # not ours
    })

    with pytest.raises(ShardRejected, match="outside"):
        base.graft_from_path(path)

    # Validation is all-or-nothing: the legal key must not have landed either.
    after = base.model.state_dict()
    for key in before:
        assert torch.equal(after[key], before[key]), key


def test_non_expert_key_is_rejected(tmp_path):
    base = _base()
    path = _shard(tmp_path, {
        "lm_head.weight": torch.zeros(4, HIDDEN),
        f"{PREFIX}.0.gate_up_proj": torch.zeros(2 * INTER, HIDDEN),
    })
    with pytest.raises(ShardRejected, match="cannot restore"):
        base.graft_from_path(path)
    assert base._backup == []


# ── the stale-orphan guard ───────────────────────────────────────────────────
def test_stale_generation_fails_without_touching_the_base(tmp_path):
    base = _base()
    handle = base.round_handle()
    base.prepare_for_round(_Model(seed=0), device="cpu")  # round moves on

    before = _snapshot(base)
    path = _shard(tmp_path, {f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0)})
    with pytest.raises(ShardRejected, match="stale"):
        handle.graft_from_path(path)

    after = base.model.state_dict()
    for key in before:
        assert torch.equal(after[key], before[key]), key


def test_current_generation_handle_grafts(tmp_path):
    base = _base()
    base.prepare_for_round(_Model(seed=0), device="cpu")
    handle = base.round_handle()
    new_rows = torch.full((2 * INTER, HIDDEN), 7.0)
    handle.graft_from_path(_shard(tmp_path, {f"{PREFIX}.0.gate_up_proj": new_rows}))
    assert torch.equal(_rows(base, 0), new_rows)
    handle.restore_grafted()


# ── refresh from the partial global model ────────────────────────────────────
def test_refresh_carries_partial_rows_into_the_full_base():
    base = _base()
    before = _snapshot(base)

    # The partial global model holds only this validator's experts (0, 1) —
    # differently-seeded weights standing in for post-merge state.
    partial = _Model(expert_indices=sorted(OWN_IDS), seed=1)
    partial_sd = partial.state_dict()

    base.refresh_from(partial)

    after = base.model.state_dict()
    for gid in sorted(OWN_IDS):  # adopted from the partial model
        key = f"{PREFIX}.{gid}.gate_up_proj"
        assert torch.equal(after[key], partial_sd[key])
        assert not torch.equal(after[key], before[key])
    for gid in (2, 3):  # not in the partial model: untouched
        key = f"{PREFIX}.{gid}.gate_up_proj"
        assert torch.equal(after[key], before[key])


def test_refresh_discards_a_stale_graft_backup(tmp_path):
    """A crashed eval leaves a graft applied; the next round's refresh must
    both rewrite the rows and drop the stale backup, or restore_grafted
    would later roll freshly-merged rows back to the dead round's values."""
    base = _base()
    base.graft_from_path(_shard(tmp_path, {
        f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0),
    }))
    partial = _Model(expert_indices=sorted(OWN_IDS), seed=1)

    base.refresh_from(partial)

    assert base._backup == []
    assert torch.equal(
        _rows(base, 0), partial.state_dict()[f"{PREFIX}.0.gate_up_proj"]
    )


# ── the backup lives on the host, and is pooled ──────────────────────────────
def test_backup_rows_are_staged_on_the_host(tmp_path):
    """The fix for a round that scored zero miners.

    `rows[local].clone()` clones onto the *source* device. With the base on
    the GPU for eval that put ~3.4 GiB of backup in VRAM: baseline fitted at
    39.5 GiB of 44.4, every miner then OOM'd at 42.9 GiB with 213 MB free.
    Nothing reads these rows on the GPU — they are written once and read once,
    at restore — so they belong on the host whatever device the base is on.
    """
    base = _base()
    base.graft_from_path(_shard(tmp_path, {
        f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0),
        f"{PREFIX}.1.down_proj": torch.full((HIDDEN, INTER), -3.0),
    }))

    assert base._backup, "nothing was backed up"
    for _, _, _, saved in base._backup:
        assert saved.device.type == "cpu"


def test_the_backup_pool_is_reused_across_miners(tmp_path):
    """Flat host cost, not 3.4 GB of churn per miner on a swapless box.

    Every miner of a round touches the same expert set — this validator's
    group — so the staging storage is allocated once and written over.
    `data_ptr` is the claim; a fresh tensor each miner would still pass any
    value-based assertion.
    """
    base = _base()
    key = f"{PREFIX}.0.gate_up_proj"

    base.graft_from_path(_shard(tmp_path, {key: torch.full((2 * INTER, HIDDEN), 7.0)}))
    first = [t.data_ptr() for *_, t in base._backup]
    base.restore_grafted()

    base.graft_from_path(_shard(tmp_path, {key: torch.full((2 * INTER, HIDDEN), 9.0)}))
    assert [t.data_ptr() for *_, t in base._backup] == first


def test_restore_from_host_storage_is_still_bit_identical(tmp_path):
    """Staging on the host must not change what restore puts back."""
    base = _base()
    before = _snapshot(base)
    base.graft_from_path(_shard(tmp_path, {
        f"{PREFIX}.0.gate_up_proj": torch.full((2 * INTER, HIDDEN), 7.0),
        f"{PREFIX}.1.down_proj": torch.full((HIDDEN, INTER), -3.0),
    }))
    base.restore_grafted()

    after = base.model.state_dict()
    for key in before:
        assert torch.equal(after[key], before[key]), key


# ── park reuses host storage instead of allocating ───────────────────────────
def test_park_writes_back_into_the_pre_allocated_shadow():
    """The fix for a silent death entering the merge.

    `.to("cpu")` allocates a fresh 29.3 GB — the host tensors were released
    when the model moved to the GPU — right as the merge phase starts wanting
    memory. Parking must land in storage that already exists, so peak host
    never exceeds steady-state host. Asserted by `data_ptr`: identity of the
    Python wrapper says nothing, the storage address is the claim.
    """
    base = _base()
    addresses = {n: p.data.data_ptr() for n, p in base.model.named_parameters()}

    base.prepare_for_round(_Model(seed=0), device="cpu")
    base.park()

    after = {n: p.data.data_ptr() for n, p in base.model.named_parameters()}
    assert after == addresses


def test_park_is_idempotent():
    base = _base()
    base.park()
    base.park()  # must not double-copy or throw when already parked
    assert not base._on_gpu


def test_park_preserves_values():
    base = _base()
    base.prepare_for_round(_Model(seed=1), device="cpu")
    before = _snapshot(base)
    base.park()
    after = base.model.state_dict()
    for key in before:
        assert torch.equal(after[key], before[key]), key


# ── the frozen contract ──────────────────────────────────────────────────────
def test_the_base_is_frozen():
    """No `.grad`, no momentum, no optimizer — the entire memory argument."""
    base = _base()
    assert all(not p.requires_grad for p in base.model.parameters())

from __future__ import annotations

import torch

from connito.shared.checkpoint_helper import save_state_dict_by_expert_group


def _tensor() -> torch.Tensor:
    return torch.zeros(2, 2, dtype=torch.float16)


def test_strict_sharding_rejects_unassigned_expert(tmp_path):
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.experts.0.gate_up_proj": _tensor(),
        "model.layers.1.mlp.experts.7.gate_up_proj": _tensor(),
    }
    expert_groups = {
        0: {
            1: [(0, 0)],
        }
    }

    try:
        save_state_dict_by_expert_group(
            state_dict=state_dict,
            expert_groups=expert_groups,
            save_dir=tmp_path,
            strict_sharding=True,
        )
    except ValueError as exc:
        assert "Strict sharding violation" in str(exc)
    else:
        raise AssertionError("Expected strict sharding to reject unmapped expert params")


def test_strict_sharding_rejects_duplicate_org_assignment(tmp_path):
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.experts.0.gate_up_proj": _tensor(),
    }
    expert_groups = {
        0: {1: [(0, 0)]},
        1: {1: [(0, 0)]},
    }

    try:
        save_state_dict_by_expert_group(
            state_dict=state_dict,
            expert_groups=expert_groups,
            save_dir=tmp_path,
            strict_sharding=True,
        )
    except ValueError as exc:
        assert "duplicate org expert IDs" in str(exc)
    else:
        raise AssertionError("Expected strict sharding to reject duplicate org expert assignments")


def test_non_strict_sharding_writes_shared_and_group_files(tmp_path):
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.experts.0.gate_up_proj": _tensor(),
        "model.layers.1.mlp.experts.7.gate_up_proj": _tensor(),
    }
    expert_groups = {
        0: {
            1: [(0, 0)],
        }
    }

    paths = save_state_dict_by_expert_group(
        state_dict=state_dict,
        expert_groups=expert_groups,
        save_dir=tmp_path,
        strict_sharding=False,
    )

    assert 0 in paths
    assert "shared" in paths
    assert (tmp_path / "model_expgroup_0.pt").exists()
    assert (tmp_path / "model_shared.pt").exists()


def test_strict_sharding_rejects_empty_expert_group_shard(tmp_path):
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": _tensor(),
    }
    expert_groups = {
        0: {
            1: [(0, 0)],
        }
    }

    try:
        save_state_dict_by_expert_group(
            state_dict=state_dict,
            expert_groups=expert_groups,
            save_dir=tmp_path,
            strict_sharding=True,
        )
    except ValueError as exc:
        assert "expert-group shard is empty" in str(exc)
    else:
        raise AssertionError("Expected strict sharding to reject empty expert-group shard")

from __future__ import annotations

import torch

from connito.shared.checkpoint_helper import save_state_dict_by_expert_group


def _tensor() -> torch.Tensor:
    return torch.zeros(2, 2, dtype=torch.float16)


def test_strict_sharding_rejects_unassigned_expert(tmp_path):
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.experts.0.gate_proj.weight": _tensor(),
        "model.layers.1.mlp.experts.7.gate_proj.weight": _tensor(),
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
        "model.layers.1.mlp.experts.0.gate_proj.weight": _tensor(),
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


def test_non_strict_sharding_writes_only_expert_group_files(tmp_path):
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.experts.0.gate_proj.weight": _tensor(),
        "model.layers.1.mlp.experts.7.gate_proj.weight": _tensor(),
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
    # The non-expert param (`model.embed_tokens.weight`) is intentionally
    # dropped — backbone state is reconstructed from from_pretrained at
    # startup, not persisted to disk.
    assert "shared" not in paths
    # Format migrated to safetensors (PR XXX) — `.pt` is no longer written.
    assert (tmp_path / "model_expgroup_0.safetensors").exists()
    assert not (tmp_path / "model_expgroup_0.pt").exists()
    assert not (tmp_path / "model_shared.safetensors").exists()
    assert not (tmp_path / "model_shared.pt").exists()


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


def test_shared_experts_are_treated_as_shared_not_expert(tmp_path):
    """`shared_experts.*` is the *non-routed* MoE block — it's shared across all
    routing decisions, not owned by any expert group. The sharding logic must
    classify it as a non-expert (i.e. shared/backbone) param so it never lands
    in a per-group shard.

    After PR #121 the shared bucket is dropped entirely: shared params are
    skipped on save. This test pins the classification, which is what
    `aggregate_miner_gradient_change` and the freezing logic also rely on.
    """
    from safetensors.torch import load_file

    state_dict = {
        # Backbone — should not be persisted.
        "model.embed_tokens.weight": _tensor(),
        # Shared expert — also backbone-class, must NOT be classified as a
        # routed expert and must NOT end up in the expert-group shard.
        "model.layers.1.mlp.shared_experts.gate_proj.weight": _tensor(),
        "model.layers.1.mlp.shared_experts.down_proj.weight": _tensor(),
        # Routed expert — owned by group 0, layer 1.
        "model.layers.1.mlp.experts.0.gate_proj.weight": _tensor(),
    }
    expert_groups = {
        0: {1: [(0, 0)]},
    }

    paths = save_state_dict_by_expert_group(
        state_dict=state_dict,
        expert_groups=expert_groups,
        save_dir=tmp_path,
        strict_sharding=False,
    )

    # The expert-group shard exists and contains the routed expert only.
    assert 0 in paths
    expert_shard = load_file(str(tmp_path / "model_expgroup_0.safetensors"))
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in expert_shard
    # Crucially: no `shared_experts.*` key leaked into the group shard.
    leaked_shared = [k for k in expert_shard if "shared_expert" in k]
    assert not leaked_shared, (
        f"shared_experts param leaked into expert_group shard: {leaked_shared}"
    )
    # Nor did any other non-routed-expert backbone key.
    leaked_backbone = [k for k in expert_shard if "expert" not in k]
    assert not leaked_backbone, (
        f"non-expert (backbone) keys leaked into expert_group shard: {leaked_backbone}"
    )

    # And nothing was written to a `shared` shard — that bucket no longer exists.
    assert "shared" not in paths
    assert not (tmp_path / "model_shared.safetensors").exists()
    assert not (tmp_path / "model_shared.pt").exists()


def test_strict_sharding_accepts_shared_experts_alongside_routed_experts(tmp_path):
    """Strict mode must NOT flag `shared_experts.*` as a sharding violation —
    that param is intentionally not in any expert group, and (post PR #121)
    it's dropped from save just like any other backbone param. The previous
    "shared bucket would contain expert params" guard is gone with the bucket;
    this test guards against re-introducing a false positive on its descendant.
    """
    state_dict = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": _tensor(),
        "model.layers.1.mlp.experts.0.gate_proj.weight": _tensor(),
    }
    expert_groups = {
        0: {1: [(0, 0)]},
    }

    # Should not raise. strict_sharding=True with all routed experts mapped is fine.
    paths = save_state_dict_by_expert_group(
        state_dict=state_dict,
        expert_groups=expert_groups,
        save_dir=tmp_path,
        strict_sharding=True,
    )
    assert 0 in paths
    assert "shared" not in paths
    assert "shared" not in paths


def test_local_expert_ids_do_not_claim_another_groups_experts(tmp_path):
    """A shard must carry only the experts its group actually owns.

    Assignment pairs are `(my_expert_id, org_expert_id)`: the first is a local
    slot index, the second the global id that appears in parameter names. Group
    0 owns global expert 5 and calls it slot 0; group 1 owns global expert 0.
    Matching on either half — as the old local-id fallback did — put global
    expert 0 in *both* shards, so group 0 shipped a expert it does not own.
    """
    state_dict = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": _tensor(),
        "model.layers.1.mlp.experts.5.gate_proj.weight": _tensor(),
    }
    expert_groups = {
        0: {1: [(0, 5)]},   # slot 0 -> global 5
        1: {1: [(0, 0)]},   # slot 0 -> global 0
    }

    save_state_dict_by_expert_group(
        state_dict=state_dict,
        expert_groups=expert_groups,
        save_dir=tmp_path,
        active_expert_group_id=0,
    )

    from safetensors.torch import load_file

    written = load_file(str(tmp_path / "model_expgroup_0.safetensors"))
    assert set(written) == {"model.layers.1.mlp.experts.5.gate_proj.weight"}

"""Every round is scored against the active group's pretrained experts.

Boot writes them once as a shard with the same writer and name as a miner
submission, so the key set is exactly what a submission must cover. These
pin that the file holds the group's experts and nothing else.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from connito.shared.helper import load_state_dict_from_path
from connito.validator.run import _write_pretrained_shard


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)  # backbone: never in a shard
        self.experts = nn.ModuleDict({"7": nn.Linear(4, 2, bias=False), "9": nn.Linear(4, 2, bias=False)})


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Mlp()


class _Model(nn.Module):
    """Parameter names follow the real ones: `layers.<L>.mlp.experts.<E>.*`."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


# Group 1 owns expert 7 of layer 0; group 2 owns expert 9.
ASSIGNMENT = SimpleNamespace(expert_group_assignment={1: {0: [(0, 7)]}, 2: {0: [(0, 9)]}})


def test_the_shard_holds_exactly_the_groups_experts(tmp_path):
    shard = _write_pretrained_shard(_Model(), ASSIGNMENT, 1, tmp_path / "pretrained")

    assert shard == tmp_path / "pretrained" / "model_expgroup_1.safetensors"
    assert set(load_state_dict_from_path(str(shard))) == {"layers.0.mlp.experts.7.weight"}


def test_the_shard_carries_the_models_weights(tmp_path):
    model = _Model()
    with torch.no_grad():
        model.layers[0].mlp.experts["7"].weight.fill_(0.5)

    sd = load_state_dict_from_path(str(_write_pretrained_shard(model, ASSIGNMENT, 1, tmp_path)))

    torch.testing.assert_close(sd["layers.0.mlp.experts.7.weight"].float(), torch.full((2, 4), 0.5))


def test_a_second_boot_overwrites_in_place(tmp_path):
    first = _write_pretrained_shard(_Model(), ASSIGNMENT, 1, tmp_path)
    model = _Model()
    with torch.no_grad():
        model.layers[0].mlp.experts["7"].weight.fill_(0.25)

    second = _write_pretrained_shard(model, ASSIGNMENT, 1, tmp_path)

    assert second == first
    torch.testing.assert_close(
        load_state_dict_from_path(str(first))["layers.0.mlp.experts.7.weight"].float(),
        torch.full((2, 4), 0.25),
    )

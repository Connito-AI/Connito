"""Every shipped expert group's dataset sources must have a shard-pick policy.

This is the one failure mode in the eval path that nothing else catches.

A policy comes from one of two registries: `_KNOWN_SOURCES` in
`connito.shared.eval_shard_pick`, or a `shard_policy.json` served with the task
(and, for a group that is never served, committed beside its `config.yaml`).
A group whose `data.dataset_sources` names a source in neither therefore fails
only on the seeded shard-pick path — which means:

  - it passes every other test in this repo, none of which enumerate
    `expert_groups/`;
  - it passes `expert_groups/build_expert_assignment.py`, which calls
    `get_dataloader` with `seed=None` and so never reaches the pick;
  - it passes miner training, for the same reason;
  - and it fails for the first time on a VALIDATOR, mid-round, with a
    `KeyError`, three retries apart, dropping that round to an unscored
    baseline of 100.0.

So the cheapest place to catch it is at commit time, by reading the shipped
configs the way a validator would.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from connito.shared.config import ExpertCfg
from connito.shared.eval_shard_pick import (
    _KNOWN_SOURCES,
    SHARD_POLICY_FILE,
    parse_served_policies,
)

EXPERT_GROUPS = Path(__file__).resolve().parents[2] / "expert_groups"


def _group_dirs() -> list[Path]:
    return sorted(p for p in EXPERT_GROUPS.iterdir() if (p / "config.yaml").is_file())


def _committed_policies(group_dir: Path) -> dict:
    """Policies from a `shard_policy.json` committed beside `config.yaml`.

    Only groups that are NEVER served carry one in the tree: `materialize_task`
    replaces a task directory wholesale, so a committed file for a served group
    would be deleted at the next switch. A served group's policy is checked by
    `cycle_api.validate` on the publishing side instead, which is the only
    place it exists before it reaches a validator.
    """
    path = group_dir / SHARD_POLICY_FILE
    if not path.is_file():
        return {}
    return parse_served_policies(json.loads(path.read_text(encoding="utf-8")))


def test_expert_groups_directory_is_discoverable():
    """Guard the guard: a wrong path here would make every test below vacuous."""
    groups = _group_dirs()
    assert groups, f"no expert group configs found under {EXPERT_GROUPS}"


@pytest.mark.parametrize("group_dir", _group_dirs(), ids=lambda p: p.name)
def test_group_sources_have_shard_pick_policies(group_dir: Path):
    """A deployable group using seeded shard pick must name only registered sources.

    Two exemptions, both narrow and both load-bearing:

    A negative `group_id` is a sentinel meaning "not assigned to a functioning
    slot". `ExpertManager` refuses such a group as the active task outright
    (`expert_manager.py`, "assign a real (non-negative) group_id ... before
    using it as a trainable task"), so it cannot reach a validator's eval path.
    Exempting it is what gives this test its teeth: `exp_metamath_p02` names an
    unregistered source today, and this assertion fires the moment somebody
    gives that group a real id — which is exactly the edit that would otherwise
    take scoring down.

    A group that explicitly opts out of the seeded path never reaches
    `pick_shard_for_source`. The opt-out must be explicit, because
    `eval_source_seeded_shard_pick` defaults to True: silence means the group
    IS on the pick path.
    """
    cfg = ExpertCfg.from_path(group_dir / "config.yaml")

    if cfg.group_id < 0:
        pytest.skip(
            f"{group_dir.name} has sentinel group_id={cfg.group_id}; "
            f"ExpertManager refuses it as an active task"
        )
    if not cfg.data.eval_source_seeded_shard_pick:
        pytest.skip(f"{group_dir.name} opts out of seeded shard pick")

    sources = cfg.data.dataset_sources or []
    registered = {**_KNOWN_SOURCES, **_committed_policies(group_dir)}
    missing = [(s.path, s.name) for s in sources if (s.path, s.name) not in registered]
    assert not missing, (
        f"{group_dir.name}/config.yaml names dataset source(s) with no entry in "
        f"any shard-pick registry: {missing}. A validator would raise KeyError at "
        f"the first eval dataloader build and fall back to an unscored baseline of "
        f"100.0. Either serve a policy for the source with the task, register it in "
        f"connito/shared/eval_shard_pick.py (with a row count measured from the "
        f"native files), or set `eval_source_seeded_shard_pick: false` on this "
        f"group and say why."
    )


@pytest.mark.parametrize("group_dir", _group_dirs(), ids=lambda p: p.name)
def test_group_revision_pins_name_configured_sources(group_dir: Path):
    """A revision pin for a source the group doesn't use is a stale edit.

    Cheap to check and it catches a rename, which would otherwise leave the
    real source pinned to nothing and silently reading `main`.
    """
    raw = yaml.safe_load((group_dir / "config.yaml").read_text()) or {}
    pins = ((raw.get("data") or {}).get("eval_source_revision_pin") or {}).keys()
    if not pins:
        pytest.skip(f"{group_dir.name} pins no revisions")

    cfg = ExpertCfg.from_path(group_dir / "config.yaml")
    configured = {s.path for s in (cfg.data.dataset_sources or [])}
    orphaned = sorted(set(pins) - configured)
    assert not orphaned, (
        f"{group_dir.name}/config.yaml pins revisions for sources it does not "
        f"configure: {orphaned}. Configured sources are {sorted(configured)}."
    )


@pytest.mark.parametrize("group_dir", _group_dirs(), ids=lambda p: p.name)
def test_committed_shard_policies_are_valid(group_dir: Path):
    """A committed policy document must parse and validate.

    `parse_served_policies` raises on a bad one, and a validator reading it
    would raise in exactly the same way — mid-round, where the cost is a
    dropped round rather than a red test.
    """
    if not (group_dir / SHARD_POLICY_FILE).is_file():
        pytest.skip(f"{group_dir.name} commits no {SHARD_POLICY_FILE}")
    _committed_policies(group_dir)

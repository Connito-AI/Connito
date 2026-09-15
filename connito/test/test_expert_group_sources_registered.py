"""Every shipped expert group's dataset sources must have a shard-pick policy.

This is the one failure mode in the eval path that nothing else catches.

`_KNOWN_SOURCES` in `connito.shared.eval_shard_pick` is a module-level dict,
deliberately not config-driven, because consensus requires every validator to
use the identical policy. A group whose `data.dataset_sources` names a source
missing from that dict therefore fails only on the seeded shard-pick path —
which means:

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

from pathlib import Path

import pytest
import yaml

from connito.shared.config import ExpertCfg
from connito.shared.eval_shard_pick import _KNOWN_SOURCES

EXPERT_GROUPS = Path(__file__).resolve().parents[2] / "expert_groups"


def _group_dirs() -> list[Path]:
    return sorted(p for p in EXPERT_GROUPS.iterdir() if (p / "config.yaml").is_file())


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
    missing = [
        (s.path, s.name) for s in sources if (s.path, s.name) not in _KNOWN_SOURCES
    ]
    assert not missing, (
        f"{group_dir.name}/config.yaml names dataset source(s) with no entry in "
        f"_KNOWN_SOURCES: {missing}. A validator would raise KeyError at the first "
        f"eval dataloader build and fall back to an unscored baseline of 100.0. "
        f"Either register the source in connito/shared/eval_shard_pick.py (with a "
        f"row count measured from the native files) or set "
        f"`eval_source_seeded_shard_pick: false` on this group and say why."
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

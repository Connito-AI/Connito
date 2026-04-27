from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal

import bittensor
import fsspec
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from connito.shared.app_logging import configure_logging, structlog
from connito.shared.helper import convert_to_str

configure_logging()
logger = structlog.get_logger(__name__)


# ---------------------------
# Utilities
# ---------------------------
def is_running_in_docker() -> bool:
    """Return True if we are inside a Docker container."""
    return Path("/.dockerenv").exists()


def find_project_root(start: Path | None = None) -> Path:
    """Walk up until we see a repo/config marker."""
    start = (start or Path(".")).expanduser().resolve(strict=True)
    markers = (".git", "pyproject.toml", "requirements.txt")
    for p in (start, *start.parents):
        if any((p / m).exists() for m in markers):
            return p
    # Fallback: "up one level from connito/" (kept from original intent)
    return start.parents[1]


def bump_run_name(name: str) -> str:
    """
    Increment a run name in a `-vN` style.

    "foo" -> "foo-v2"
    "foo-v3" -> "foo-v4"
    "test-1B" -> "test-1B-v2"
    """
    m = re.match(r"^(.*?)(?:-v(\d+))?$", name)
    if not m:
        return f"{name}-v2"
    base, ver = m.group(1), m.group(2)
    return f"{base}-v{int(ver) + 1}" if ver else f"{base}-v2"


def norm_for_compare(v: Any) -> Any:
    if isinstance(v, Path):
        return v.as_posix()
    return v


def deep_compare(a: Any, b: Any, path: str = "") -> tuple[bool, list[str]]:
    """
    Deep-compare two nested structures.
    Returns (ok, diffs) where diffs are human readable messages.
    """
    diffs: list[str] = []

    if isinstance(a, dict) and isinstance(b, dict):
        a_keys, b_keys = set(a), set(b)
        missing = a_keys - b_keys
        extra = b_keys - a_keys
        if missing:
            diffs.append(f"Missing keys in other at '{path}': {sorted(missing)}")
        if extra:
            diffs.append(f"Extra keys in other at '{path}': {sorted(extra)}")

        ok = not missing and not extra
        for k in sorted(a_keys & b_keys):
            child_ok, child_diffs = deep_compare(a[k], b[k], f"{path}.{k}" if path else k)
            ok = ok and child_ok
            diffs.extend(child_diffs)
        return ok, diffs

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, [f"Length mismatch at '{path}': other={len(b)} new={len(a)}"]
        ok = True
        for i, (ai, bi) in enumerate(zip(a, b, strict=False)):
            child_ok, child_diffs = deep_compare(ai, bi, f"{path}[{i}]")
            ok = ok and child_ok
            diffs.extend(child_diffs)
        return ok, diffs

    a_n, b_n = norm_for_compare(a), norm_for_compare(b)
    if a_n != b_n:
        return False, [f"Mismatch at '{path}': other={b_n!r} new={a_n!r}"]
    return True, []


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Base Config
# ---------------------------
class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Fields listed here are "locked" — they should stay at their defaults.
    # Subclasses override this to declare which fields fall into category (2).
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __str__(self) -> str:
        return self.to_json()

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs, indent=4)

    def locked_defaults(self) -> dict[str, Any]:
        """Return {field_name: default_value} for all locked fields."""
        result: dict[str, Any] = {}
        for name in self._LOCKED_FIELDS:
            info: FieldInfo = self.model_fields[name]
            if info.default is not PydanticUndefined:
                result[name] = info.default
            elif info.default_factory is not None:
                result[name] = info.default_factory()
        return result

    @classmethod
    def from_path(cls, path: str | Path) -> "BaseConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# ---------------------------
# Sections
# ---------------------------
class ChainCfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({"netuid", "network"})
    netuid: int = 102
    uid: int = 0
    hotkey_ss58: str = ""
    coldkey_ss58: str = ""
    ip: str = "0.0.0.0"
    port: int = 8000
    coldkey_name: str = "template_coldkey_name"
    hotkey_name: str = "template_hotkey_name"
    network: str = "archive"
    lite_network: str = "finney"


class CycleCfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "cycle_length", "distribute_period", "train_period", "commit_period",
        "submission_period", "validate_period", "merge_period", "owner_url",
    })
    cycle_length: int = 448 # 1.5 hr
    distribute_period: int = 20 # 4 mins
    train_period: int = 300 # 1 hr (will adjust to 500 mins when mature to align with Diloco)
    commit_period: int = 10 # 2 mins
    submission_period: int = 60 # 4 mins
    validate_period: int = 10 # 10 mins
    merge_period: int = 50 # 10 mins

    owner_url: str = "https://cycle-api.connito.ai:443"
    version_range_cycles: int = 3  # how many cycles back to accept checkpoints
    # Owner-node API retry policy
    api_timeout_sec: int = 10
    api_retries: int = 5
    api_backoff_sec: int = 2


class RunCfg(BaseConfig):
    run_name: str = "foundation"
    root_path: Path = Field(default_factory=find_project_root)


class ModelCfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({"model_path", "base_arch_model"})
    model_path: str = "deepseek-ai/DeepSeek-V2-Lite"
    base_arch_model: str = "deepseek-ai/DeepSeek-V2-Lite"
    foundation: bool = True
    torch_compile: bool = False
    attn_implementation: str = "sdpa"
    precision: str = "fp16-mixed"
    device: str = "cuda"


class DatasetSourceCfg(BaseConfig):
    path: str
    name: str | None = None
    weight: PositiveFloat = 1.0
    text_column: str = "text"

    @model_validator(mode="after")
    def _validate_non_empty(self):
        if not self.path.strip():
            raise ValueError("data.dataset_sources[].path cannot be empty.")
        if not self.text_column.strip():
            raise ValueError("data.dataset_sources[].text_column cannot be empty.")
        return self


class DataCfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({"dataset_name", "data_dir"})
    dataset_name: str = "allenai/c4"
    data_dir: str | None = 'en'
    dataset_sources: list[DatasetSourceCfg] | None = None
    batch_size: PositiveInt = 4
    sequence_length: PositiveInt = 4096
    per_device_train_batch_size: PositiveInt = 1
    world_size: int = 10
    rank: int = 1
    dataset_class: str | None = None
    vali_fraction: float = 0.1

    @model_validator(mode="after")
    def _validate_dataset_sources(self):
        if self.dataset_sources is not None and len(self.dataset_sources) == 0:
            raise ValueError("data.dataset_sources must contain at least one source when provided.")
        return self


class DataloaderCfg(BaseConfig):
    world_size: int = 10  # number of distinct data shards (typically = number of validators)


class MoECfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "num_experts", "num_worker_groups", "num_experts_per_tok", "partial_topk", "full_topk",
    })
    my_expert_group_id: int = 1
    interleave: bool = True
    num_experts: PositiveInt = 8
    num_experts_per_tok: PositiveInt = 2
    partial_topk: PositiveInt = 1
    full_topk: PositiveInt = 2
    aux_load_balance: bool = True
    router_aux_loss_coef: float = 1.0
    partial_moe: bool = True
    num_worker_groups: PositiveInt = 2


class OptimizerCfg(BaseConfig):
    lr: float = 1e-5
    outer_lr: float = 0.7
    outer_momentum: float = 0.9


class ParallelismCfg(BaseConfig):
    gradient_accumulation_steps: PositiveInt = 4
    global_opt_interval: PositiveInt = 100
    world_size: PositiveInt = 1
    port: PositiveInt = 29500
    ip_address: str = "127.0.0.1"

    @staticmethod
    def cuda_device_count_safe() -> int:
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 0


class ScheduleCfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({"total_steps"})
    warmup_steps: int = 0
    total_steps: PositiveInt = 88_000


class CheckpointCfg(BaseConfig):
    resume_from_ckpt: bool = True
    strict_sharding: bool = False
    base_checkpoint_path: Path = Path("checkpoints/miner")
    checkpoint_path: Path | None = None
    checkpoint_interval: PositiveInt | None = None
    full_validation_interval: PositiveInt | None = None
    checkpoint_topk: PositiveInt = 2
    validator_checkpoint_path: Path = Path("validator_checkpoint")
    # Legacy compatibility knob. Miner checkpoint downloads currently use an
    # ordered HF -> validator HTTP fallback path rather than parallel shard
    # fan-out, but older configs may still include this field.
    download_concurrency: PositiveInt = 4


class HfCfg(BaseConfig):
    # HuggingFace Hub is the checkpoint transport: validators upload the
    # checkpoint directory, commit the revision SHA to the Bittensor chain,
    # and miners download from the returned repo@revision.
    #
    # checkpoint_repo: the HF repo the validator pushes to. Must exist and
    # be writable by the HF_TOKEN. If omitted, runtime will derive
    # {authenticated_hf_user}/{default_repo_name}. The on-chain advertised
    # repo can diverge from this upload target; validator code currently
    # derives the advertised repo as {owner}/cycle from the upload repo.
    checkpoint_repo: str | None = None
    default_repo_name: str = "co"
    # Read from HF_TOKEN env at runtime — not stored in config YAML.
    # Validators need write access; miners need read access (or public repo).
    token_env_var: str = "HF_TOKEN"

    @staticmethod
    def _normalize_checkpoint_repo_value(value: str | None) -> str | None:
        checkpoint_repo = (value or "").strip()
        if checkpoint_repo and "/" not in checkpoint_repo:
            raise ValueError("hf.checkpoint_repo must be '<namespace>/<repo>'")
        return checkpoint_repo or None

    @staticmethod
    def _normalize_default_repo_name(value: str) -> str:
        default_repo_name = value.strip()
        if not default_repo_name:
            raise ValueError("hf.default_repo_name cannot be empty")
        if "/" in default_repo_name:
            raise ValueError("hf.default_repo_name must be a repo name, not '<namespace>/<repo>'")
        return default_repo_name

    def resolve_upload_repo(self, derived_repo: str | None = None) -> str | None:
        return self.checkpoint_repo or self._normalize_checkpoint_repo_value(derived_repo)

    def advertised_repo_id(self, upload_repo: str | None) -> str | None:
        if not upload_repo:
            return None
        return upload_repo

    def uses_explicit_checkpoint_repo(self) -> bool:
        return self.checkpoint_repo is not None

    @model_validator(mode="after")
    def _validate_hf_repo_settings(self):
        self.checkpoint_repo = self._normalize_checkpoint_repo_value(self.checkpoint_repo)
        self.default_repo_name = self._normalize_default_repo_name(self.default_repo_name)
        return self


class LoggingCfg(BaseConfig):
    log_wandb: bool = False
    wandb_project_name: str = "test-moe"
    wandb_resume: bool = False
    wandb_full_id: str = ""
    wandb_partial_id: list[str | None] = Field(default_factory=list)
    base_metric_path: Path = Path("metrics")
    metric_path: Path | None = None
    metric_interval: PositiveInt | None = None


class ValidatorCheckpointCfg(CheckpointCfg):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({"max_submission_bytes", "max_submission_bytes_per_expert"})
    base_checkpoint_path: Path = Path("checkpoints/validator")
    miner_submission_path: Path = Path("miner_submission")
    max_submission_bytes: int = 8 * 1024**3
    max_submission_bytes_per_expert: int | None = None
    miner_submission_archive_path: Path = Path("miner_submission_archive")
    archive_submissions: bool = False
    # Max concurrent /submit-checkpoint streams. Protects the server downlink
    # and disk write throughput from N parallel 3.35 GiB uploads dragging each
    # other under the 11.5 MiB/s floor that would trip SUBMISSION_TIMEOUT_SEC.
    submission_concurrency: int = 2
    cleanup_stale_temporary_checkpoints: bool = True
    miner_submission_max_age_cycles: PositiveFloat = 1.5
    miner_submission_archive_max_files: PositiveInt = 500


class DhtCfg(BaseConfig):
    port: int = 6000


class OwnerCheckpointCfg(CheckpointCfg):
    base_checkpoint_path: Path = Path("checkpoints/owner")


class ExpertCfg(BaseConfig):
    data: DataCfg = Field(default_factory=DataCfg)
    group_id: int = 0


class TaskCfg(BaseConfig):
    expert_group_name: str = "exp_math"
    load_all_expert_groups: bool = False
    base_path: Path = Path("expert_groups")
    path: Path | None = None
    exp: ExpertCfg = Field(default_factory=ExpertCfg)


# ---------------------------
# Top-level config
# ---------------------------
class WorkerConfig(BaseConfig):
    """
    Centralized training/eval configuration for mycelia runs.

    Notes
    -----
    - This class derives paths and can optionally bump run_name if an on-disk config differs.
    - Side-effects (mkdir, wallet lookup, task yaml load) are done in model_post_init.
    """

    run: RunCfg = Field(default_factory=RunCfg)
    chain: ChainCfg = Field(default_factory=ChainCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    moe: MoECfg = Field(default_factory=MoECfg)
    ckpt: CheckpointCfg = Field(default_factory=CheckpointCfg)
    hf: HfCfg = Field(default_factory=HfCfg)
    sched: ScheduleCfg = Field(default_factory=ScheduleCfg)
    log: LoggingCfg = Field(default_factory=LoggingCfg)
    opt: OptimizerCfg = Field(default_factory=OptimizerCfg)
    cycle: CycleCfg = Field(default_factory=CycleCfg)
    task: TaskCfg = Field(default_factory=TaskCfg)
    dataloader: DataloaderCfg = Field(default_factory=DataloaderCfg)

    # -----------------------
    # Lifecycle
    # -----------------------
    def model_post_init(self, __context: Any) -> None:
        # Fill keys (best-effort)
        self._fill_wallet_data()

        # When running inside Docker, override root to container mount points.
        # The compose file mounts: repo root → /data, expert_groups → /app/expert_groups.
        if is_running_in_docker():
            self.run.root_path = Path("/data")
            self.task.base_path = Path("/app/expert_groups")
            logger.info("Docker detected — root_path set to /data")

        # Derive paths
        self._refresh_paths()

        # Load per-task overrides
        self._update_by_task()

        # Create directories
        self._ensure_runtime_dirs()

    # -----------------------
    # Derived paths / IO
    # -----------------------
    def _refresh_paths(self) -> None:
        """Derive all runtime paths from root_path + class defaults.

        Always computes from the original default values so this method
        is idempotent — safe to call multiple times without stacking prefixes.
        """
        root = self.run.root_path
        ckpt_cls = type(self.ckpt)
        log_cls = type(self.log)

        # ckpt paths — always start from the class default (relative)
        base_ckpt = root / Path(ckpt_cls.model_fields["base_checkpoint_path"].default)
        self.ckpt.base_checkpoint_path = base_ckpt
        self.ckpt.checkpoint_path = (
            base_ckpt / self.chain.coldkey_name / self.chain.hotkey_name / self.run.run_name
        )
        self.ckpt.validator_checkpoint_path = (
            base_ckpt / Path(ckpt_cls.model_fields["validator_checkpoint_path"].default)
        )

        # logging paths
        base_metric = root / Path(log_cls.model_fields["base_metric_path"].default)
        self.log.base_metric_path = base_metric
        self.log.metric_path = base_metric / f"{self.run.run_name}.csv"

        # task paths
        self.task.base_path = root / self.task.base_path
        self.task.path = self.task.base_path / self.task.expert_group_name

        # optional ckpt sub-paths — always from class default + base_ckpt
        if hasattr(self.ckpt, "miner_submission_path"):
            self.ckpt.miner_submission_path = (
                base_ckpt / Path(ckpt_cls.model_fields["miner_submission_path"].default)
            )
        if hasattr(self.ckpt, "miner_submission_archive_path"):
            self.ckpt.miner_submission_archive_path = (
                base_ckpt / Path(ckpt_cls.model_fields["miner_submission_archive_path"].default)
            )

    def _ensure_runtime_dirs(self) -> None:
        assert self.ckpt.checkpoint_path is not None
        assert self.task.path is not None
        assert self.log.metric_path is not None

        dirs = [
            self.task.base_path,
            self.task.path,
            self.ckpt.base_checkpoint_path,
            self.ckpt.checkpoint_path,
            self.log.base_metric_path,
            self.ckpt.validator_checkpoint_path,
        ]
        ensure_dirs(dirs)

    def _update_by_task(self, expert_group_name: str | None = None) -> None:
        if expert_group_name:
            self.task.expert_group_name = expert_group_name
            self._refresh_paths()

        assert self.task.path is not None
        cfg_path = self.task.path / "config.yaml"
        self.task.exp = ExpertCfg.from_path(cfg_path)  # type: ignore
        self._refresh_paths()

    def _fill_wallet_data(self) -> None:
        if self.chain.hotkey_ss58 and self.chain.coldkey_ss58:
            logger.info(
                "Wallet data already present in config, skipping chain lookup",
                hotkey_ss58=self.chain.hotkey_ss58,
                coldkey_ss58=self.chain.coldkey_ss58,
                uid=self.chain.uid,
            )
            return

        wallet = bittensor.Wallet(name=self.chain.coldkey_name, hotkey=self.chain.hotkey_name)
        logger.info("Resolving wallet data from chain",
                     coldkey=self.chain.coldkey_name,
                     hotkey=self.chain.hotkey_name,
                     network=self.chain.network)
        subtensor = bittensor.Subtensor(network=self.chain.network)
        try:
            self.chain.hotkey_ss58 = wallet.hotkey.ss58_address
            self.chain.coldkey_ss58 = wallet.coldkeypub.ss58_address
            self.chain.uid = subtensor.metagraph(netuid=self.chain.netuid).hotkeys.index(self.chain.hotkey_ss58)
        except bittensor.KeyFileError as e:
            logger.warning(
                "Cannot find wallet keys; check coldkey/hotkey names or pass --hotkey_name/--coldkey_name",
                coldkey_name=self.chain.coldkey_name,
                hotkey_name=self.chain.hotkey_name,
                error=str(e),
            )

    # -----------------------
    # Locked-field enforcement
    # -----------------------
    # Sub-config sections that participate in locked-field checks.
    _LOCKED_SECTIONS: ClassVar[tuple[str, ...]] = ("chain", "cycle", "model", "moe", "sched", "ckpt", "evaluation")

    @classmethod
    def from_path(cls, path: str | Path, auto_update_config: bool = False) -> "WorkerConfig":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        instance = cls(**data)
        instance._prompt_new_fields(yaml_data=data, config_path=path, auto_update=auto_update_config)
        instance.check_and_prompt_locked(config_path=path, auto_update=auto_update_config)
        return instance

    def _prompt_new_fields(
        self,
        yaml_data: dict,
        config_path: Path | None = None,
        auto_update: bool = False,
    ) -> None:
        """
        Detect fields that exist in the code but are missing from the config yaml.
        Prompt the user to accept the default or enter a custom value.
        """
        if not auto_update and not sys.stdin.isatty():
            return

        changed = False
        for section_name in self._LOCKED_SECTIONS:
            sub_cfg = getattr(self, section_name, None)
            if not isinstance(sub_cfg, BaseConfig):
                continue
            yaml_section = yaml_data.get(section_name, {}) or {}

            for field_name, field_info in sub_cfg.model_fields.items():
                if field_name in yaml_section:
                    continue
                # Field is missing from yaml — it's using the code default
                if field_info.default is PydanticUndefined and field_info.default_factory is None:
                    continue  # required field with no default, pydantic would have errored

                default_val = field_info.default if field_info.default is not PydanticUndefined else field_info.default_factory()

                if auto_update:
                    logger.info(
                        "New config field — using default",
                        field=f"{section_name}.{field_name}",
                        default=default_val,
                    )
                    changed = True
                else:
                    print(f"\n[{section_name}.{field_name}] new field not in config file")
                    answer = input(f"  Enter value (or press Enter for default: {default_val!r}): ").strip()
                    if answer:
                        # Try to cast to the field's type
                        try:
                            ann = field_info.annotation
                            if ann is int or ann is float:
                                answer = ann(answer)
                            elif ann is bool:
                                answer = answer.lower() in ("true", "1", "yes")
                            elif ann is Path:
                                answer = Path(answer)
                        except (ValueError, TypeError):
                            pass
                        setattr(sub_cfg, field_name, answer)
                    changed = True

        if changed and config_path is not None:
            data = self.model_dump(exclude={"task": {"exp"}})
            data = self._strip_root(data, self.run.root_path)
            data = convert_to_str(data)
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, sort_keys=False)
                logger.info("Wrote updated config (new fields added)", path=str(config_path))
            except OSError as e:
                logger.warning(
                    "Could not persist new fields — config path is read-only; defaults applied in memory only",
                    path=str(config_path),
                    error=str(e),
                )

    def check_and_prompt_locked(self, config_path: Path | None = None, auto_update: bool = False) -> None:
        """
        For each locked field in category-(2) sub-configs, compare the loaded
        value against the class default.  When they differ:
        - auto_update=True: reset to default without prompting (for systemd / non-interactive).
        - stdin is a TTY: ask the user whether to reset.
        - otherwise: skip silently.
        """
        if not auto_update and not sys.stdin.isatty():
            return

        # Build flat list of (dotted_path, sub_cfg_object) to check.
        # Top-level sections come from _LOCKED_SECTIONS; task.exp.data is added separately.
        candidates: list[tuple[str, BaseConfig]] = [
            (section, getattr(self, section))
            for section in self._LOCKED_SECTIONS
            if isinstance(getattr(self, section, None), BaseConfig)
        ]
        changed = False
        for label, sub_cfg in candidates:
            for field_name, default_val in sub_cfg.locked_defaults().items():
                current_val = getattr(sub_cfg, field_name)
                if norm_for_compare(current_val) == norm_for_compare(default_val):
                    continue
                if auto_update:
                    logger.info(
                        "auto_update_config: resetting locked field to default",
                        field=f"{label}.{field_name}",
                        old_value=current_val,
                        new_value=default_val,
                    )
                    setattr(sub_cfg, field_name, default_val)
                    changed = True
                else:
                    print(f"\n[{label}.{field_name}] current value from local config file : {current_val!r}")
                    print(f"[{label}.{field_name}] expected default value according to chain: {default_val!r}")
                    answer = input("  Reset to default? [y/N] ").strip().lower()
                    if answer == "y":
                        setattr(sub_cfg, field_name, default_val)
                        changed = True

        if changed and config_path is not None:
            data = self.model_dump(exclude={"task": {"exp"}})
            data = self._strip_root(data, self.run.root_path)
            data = convert_to_str(data)
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, sort_keys=False)
                logger.info("Wrote updated config (locked fields reset)", path=str(config_path))
            except OSError as e:
                logger.warning(
                    "Could not persist locked-field reset — config path is read-only; reset applied in memory only",
                    path=str(config_path),
                    error=str(e),
                )

    # -----------------------
    # Config equivalence / versioning
    # -----------------------
    def same_as(self, other: dict) -> bool:
        ok, diffs = deep_compare(self.to_dict(), other)
        for d in diffs:
            logger.debug("Config mismatch", diff=d)
        return ok

    def resolve_run_name_against_disk(
        self,
        *,
        overwrite: bool = False,
        bump_if_diff: bool = True,
    ) -> "WorkerConfig":
        """
        Compare to on-disk config at <checkpoint_path>/config.yaml.
        - If missing: returns self.
        - If same: returns self.
        - If different:
            * overwrite=True -> return WorkerConfig(**other_dict) (matches original behavior)
            * else bump_if_diff=True -> bump run_name and return self
            * else -> return self unchanged
        """
        assert self.ckpt.checkpoint_path is not None
        config_path = self.ckpt.checkpoint_path / "config.yaml"
        if not config_path.exists():
            return self

        with open(config_path, encoding="utf-8") as f:
            other = yaml.safe_load(f) or {}

        if self.same_as(other):
            return self

        if overwrite:
            logger.info("Overwriting existing run_name with on-disk config.")
            return self.__class__(**other)  # preserve subclass type

        if bump_if_diff:
            self.run.run_name = bump_run_name(self.run.run_name)
            self._refresh_paths()
            logger.info("Bumped run_name due to config differences.", run_name=self.run.run_name)

        return self

    # -----------------------
    # Persistence
    # -----------------------
    @staticmethod
    def _strip_root(data: dict, root: Path) -> dict:
        """Recursively strip *root* prefix from Path-like string values so the
        YAML stays portable across host and Docker environments."""
        root_str = root.as_posix().rstrip("/") + "/"

        def _walk(obj):
            if isinstance(obj, dict):
                return {k: _walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_walk(i) for i in obj]
            if isinstance(obj, Path):
                try:
                    return obj.relative_to(root)
                except ValueError:
                    return obj
            if isinstance(obj, str) and obj.startswith(root_str):
                return obj[len(root_str):]
            return obj

        return _walk(data)

    def write(self) -> None:
        """
        Persist this config to `<checkpoint_path>/config.yaml`.
        Excludes task.exp (loaded from task config file) and task.base_path /
        task.path (derived at runtime, remapped in Docker — should not be
        persisted so the YAML stays portable across host and container).
        All remaining paths are written relative to root_path.
        """
        assert self.ckpt.checkpoint_path is not None
        data = self.model_dump(exclude={
            "run": {"root_path"},
            "task": {"exp", "base_path", "path"},
        })
        data = self._strip_root(data, self.run.root_path)
        data = convert_to_str(data)

        ensure_dirs([self.ckpt.checkpoint_path])
        target = self.ckpt.checkpoint_path / "config.yaml"

        with fsspec.open(target.as_posix(), "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False)

        logger.info("Wrote config", path=str(target))


class MinerConfig(WorkerConfig):
    role: str = "miner"
    local_par: ParallelismCfg = Field(default_factory=ParallelismCfg)

    @model_validator(mode="after")
    def _derive_all(self):
        # derive grad accumulation if user explicitly set 0
        if self.local_par.gradient_accumulation_steps == 0:
            effective_batch = self.task.exp.data.batch_size
            device_batch = self.task.exp.data.per_device_train_batch_size
            g = math.ceil(effective_batch / (device_batch * self.local_par.world_size))
            self.local_par.gradient_accumulation_steps = max(1, int(g))

        goi = self.local_par.global_opt_interval
        frac = 0.2

        if self.ckpt.checkpoint_interval is None:
            self.ckpt.checkpoint_interval = max(1, round(goi * frac))
        if self.ckpt.full_validation_interval is None:
            self.ckpt.full_validation_interval = max(1, round(goi * frac))
        if self.log.metric_interval is None:
            self.log.metric_interval = max(1, round(goi * frac))

        return self


class ValidatorRunCfg(RunCfg):
    averager_step_timeout_sec: int = 60  # seconds to wait for averager group formation (1 min)
    averager_step_max_retries: int = 2  # max retry attempts for averager step
    record_cuda_mem_history: bool = False  # enable torch.cuda.memory._record_memory_history (leaks RAM; profiling only)


class EvalCfg(BaseConfig):
    _LOCKED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "top_k_miners_to_merge", "top_k_miners_to_reward", "score_window", "foreground_top_n"
    })
    top_k_miners_to_merge: int = 1    # top-N miners whose gradients are merged into global model
    top_k_miners_to_reward: int = 3   # top-N miners who receive chain weights (proportional to score after normalization)
    score_window: int = 8            # max number of phases (points) retained per miner in MinerScoreAggregator
    foreground_top_n: PositiveInt = 5
    background_worker_enabled: bool = True
    per_miner_download_timeout_sec: PositiveInt = 180
    per_miner_eval_timeout_sec: PositiveInt = 300


class ValidatorConfig(WorkerConfig):
    role: str = "validator"
    ckpt: ValidatorCheckpointCfg = Field(default_factory=ValidatorCheckpointCfg)
    dht: DhtCfg = Field(default_factory=DhtCfg)
    run: ValidatorRunCfg = Field(default_factory=ValidatorRunCfg)
    evaluation: EvalCfg = Field(default_factory=EvalCfg)

    def write_docker_env(self, env_path: Path | None = None) -> None:
        """Generate a Docker compose .env file from the current config."""
        if env_path is None:
            env_path = self.run.root_path / "connito" / "validator" / "docker" / ".env"

        wallet_path = Path.home() / ".bittensor" / "wallets"
        data_dir = self.run.root_path

        # Use hotkey name + run name as project name so multiple validators
        # on the same host get unique container names automatically.
        # e.g. connito-hk1-mainnet-server-1
        project_name = f"connito-{self.chain.hotkey_name}-{self.run.run_name}"
        expert_groups_path = self.run.root_path / "expert_groups"

        lines = [
            f"COMPOSE_PROJECT_NAME={project_name}",
            f"IMAGE=ghcr.io/connito-ai/connito-validator:stable",
            f"WALLET_NAME={self.chain.coldkey_name}",
            f"HOTKEY_NAME={self.chain.hotkey_name}",
            f"BITTENSOR_WALLET_PATH={wallet_path}",
            f"DATA_DIR={data_dir}",
            f"CONFIG_PATH={self.ckpt.checkpoint_path / 'config.yaml'}",
            f"EXPERT_GROUPS_PATH={expert_groups_path}",
            f"VALIDATOR_GPU_ID=0",
            f"SERVER_GPU_ID=0",
            f"HF_TOKEN=",
            f"WANDB_API_KEY=",
            f"WANDB_MODE=online",
            f"WATCHTOWER_POLL_INTERVAL=300",
            f"WATCHTOWER_NOTIFICATIONS=",
            f"WATCHTOWER_NOTIFICATION_SLACK_HOOK_URL=",
        ]

        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote Docker compose .env", path=str(env_path))


class OwnerConfig(WorkerConfig):
    role: str = "owner"
    ckpt: OwnerCheckpointCfg = Field(default_factory=OwnerCheckpointCfg)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train mycelia with config")
    subparsers = parser.add_subparsers(dest="command")

    # --- create_config ---
    create_cfg = subparsers.add_parser("create_config", help="Generate a template config file")
    create_cfg.add_argument("--role", choices=["miner", "validator"], required=True, help="Role to generate config for")
    create_cfg.add_argument("--hotkey_name", type=str, help="Wallet hotkey name")
    create_cfg.add_argument("--coldkey_name", type=str, help="Wallet coldkey name")
    create_cfg.add_argument("--run_name", type=str, help="Run name")

    # --- create_docker_env ---
    create_env = subparsers.add_parser("create_docker_env", help="Generate Docker compose .env from an existing config")
    create_env.add_argument("--path", type=str, required=True, help="Path to existing validator YAML config file")

    # --- Top-level flags used by connito.validator.run (Docker entrypoint) ---
    parser.add_argument("--path", type=str, help="Path to validator YAML config file")
    parser.add_argument(
        "--auto_update_config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-reset locked config fields to defaults without prompting.",
    )

    parser.add_argument("--hotkey_name", type=str, help="Wallet hotkey name")
    parser.add_argument("--coldkey_name", type=str, help="Wallet coldkey name")
    parser.add_argument("--dht_port", type=int, default=7002, help="DHT port for owner DHT bootstrap service")
    parser.add_argument("--dht_public_ip", type=str, help="Public IP address for DHT peer discovery")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite current config with on-disk one.")
    parser.add_argument("--no_bump", action="store_true", help="Do not bump run_name on config diff.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: wait_till() short-circuits without sleeping or polling the chain.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "create_config":
        config_dict: dict[str, Any] = {}
        if args.run_name:
            config_dict["run"] = {"run_name": args.run_name}
        if args.hotkey_name:
            config_dict.setdefault("chain", {})["hotkey_name"] = args.hotkey_name
        if args.coldkey_name:
            config_dict.setdefault("chain", {})["coldkey_name"] = args.coldkey_name

        if args.role == "validator":
            cfg = ValidatorConfig(**config_dict)
            cfg.write()
            cfg.write_docker_env()
        elif args.role == "miner":
            MinerConfig(**config_dict).write()

    elif args.command == "create_docker_env":
        cfg = ValidatorConfig.from_path(args.path, auto_update_config=args.auto_update_config)
        cfg.write_docker_env()


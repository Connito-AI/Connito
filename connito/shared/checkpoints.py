from __future__ import annotations

import os
import shutil
import time
from collections import Counter
from functools import total_ordering
from pathlib import Path
from typing import Any

import bittensor
import fsspec
import torch
from fsspec.generic import GenericFileSystem
from pydantic import BaseModel, ConfigDict, Field

from connito.shared.app_logging import structlog
from connito.shared.checkpoint_helper import compile_full_state_dict_from_path
from connito.shared.expert_manager import get_layer_expert_id
from connito.shared.helper import get_model_hash, parse_dynamic_filename
from connito.shared.schema import sign_message, verify_message

from connito.shared.cycle import (
    PhaseNames,
    PhaseResponse,
    get_allowed_version_range,
    get_blocks_from_previous_phase_from_api,
    get_phase_from_api,
    wait_till,
)
from connito.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from connito.shared.chain import (
    SignedModelHashChainCommit,
    WorkerChainCommit,
    get_chain_commits,
)
from connito.shared.expert_manager import (
    ExpertManager,
    get_layer_expert_id,
    ExpertAssignments
)

logger = structlog.get_logger(__name__)


def _normalize_hash(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value.lower()
    return str(value).lower()


def _hash_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        return bytes.fromhex(value)
    except ValueError:
        return value.encode()


@total_ordering
class ModelCheckpoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, extra="allow")

    signed_model_hash: str | None = Field(default=None, alias="h")
    model_hash: str | None = None
    global_ver: int | None = None
    expert_group: int | None = None

    inner_opt: int | None = None
    path: Path | None = None  # path to folder or file
    role: str | None = None  # [miner, validator]
    place: str = "local"  # [local / onchain]

    signature_required: bool = False
    signature_verified: bool = False

    hash_required: bool = False
    hash_verified: bool = False

    expert_group_check_required: bool = False
    expert_group_verified: bool = False

    def __eq__(self, other: object) -> bool:
        try:
            other_global_ver = other.global_ver  # type: ignore[attr-defined]
            other_inner_opt = other.inner_opt  # type: ignore[attr-defined]
        except AttributeError:
            return NotImplemented
        return (
            self.global_ver == other_global_ver
            and self.inner_opt == other_inner_opt
        )

    def __lt__(self, other: ModelCheckpoint) -> bool:
        try:
            other_global_ver = other.global_ver  # type: ignore[attr-defined]
            other_inner_opt = other.inner_opt  # type: ignore[attr-defined]
        except AttributeError:
            return NotImplemented

        # Compare by global_ver first
        self_global_ver = self.global_ver if isinstance(self.global_ver, int) else -1
        other_global_ver = other_global_ver if isinstance(other_global_ver, int) else -1
        if self_global_ver != other_global_ver:
            return self_global_ver < other_global_ver

        # Then compare by inner_opt
        self_inner_opt = self.inner_opt if isinstance(self.inner_opt, int) else -1
        other_inner_opt = other_inner_opt if isinstance(other_inner_opt, int) else -1
        return self_inner_opt < other_inner_opt

    def _extra(self, key: str, default: Any | None = None) -> Any | None:
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]
        return default

    def expired(self) -> bool:
        return False

    def hash_model(self) -> str:
        if self.path is None:
            raise ValueError("path is required to hash a model")

        if self.expert_group is None:
            raise ValueError("expert_group is required to hash a model")
        
        state = compile_full_state_dict_from_path(self.path, expert_groups=[self.expert_group])
        self.model_hash = get_model_hash(state, hex=True)
        self.hash_verified = True
        return self.model_hash

    def sign_hash(self, wallet: bittensor.Wallet) -> str:
        if self.model_hash is None:
            self.hash_model()

        self.signed_model_hash = sign_message(
            wallet.hotkey,
            self.model_hash,
        )
        self.signature_verified = True
        return self.signed_model_hash

    def _verify_hash(self, state_dict: dict | None) -> bool:
        
        expected_hash = get_model_hash(state_dict, hex=True)

        if expected_hash is None:
            logger.warning(
                "Hash verification failed: unable to compute expected hash",
                checkpoint_path=self.path,
                expert_group=self.expert_group,
            )
            self.hash_verified = False
            return False

        self.hash_verified = self.model_hash == expected_hash
        if not self.hash_verified:
            logger.warning(
                "Checkpoint hash does not match expected hash",
                checkpoint_path=self.path,
                expert_group=self.expert_group,
                expected_hash=expected_hash,
                provided_hash=self.model_hash,
            )

        return self.hash_verified

    def _verify_signature(self) -> bool:
        if self.signed_model_hash is None or self.model_hash is None or self.hotkey is None:
            logger.warning(
                "Signature verification failed: missing signed hash, model hash, or hotkey",
                signed_model_hash_present=self.signed_model_hash,
                model_hash_present=self.model_hash,
                hotkey_present=self.hotkey,
                checkpoint=self,
            )
            self.signature_verified = False
            return False

        self.signature_verified = verify_message(
            self.hotkey, message=self.model_hash, signature_hex=self.signed_model_hash
        )
        if not self.signature_verified:
            logger.warning(
                "Checkpoint signature invalid — signed hash does not match model hash",
                hotkey=self.hotkey[:6] if self.hotkey else None,
                model_hash=self.model_hash[:6] if self.model_hash else None,
            )

        return self.signature_verified

    def _verify_expert_group(self, state_dict: dict | None, expert_group_assignment: ExpertAssignments) -> bool:
        if not self.expert_group_check_required:
            self.expert_group_verified = True
            return True

        if state_dict is None:
            self.expert_group_verified = False
            return False

        if len(state_dict) == 0:
            logger.warning(
                "expert group verification failed: empty model_state_dict",
                expert_group=self.expert_group,
                hotkey=self.hotkey,
            )
            self.expert_group_verified = False
            return self.expert_group_verified

        allowed_layers = expert_group_assignment.get(self.expert_group, {})
        routed_expert_key_count = 0
        for name, tensor in state_dict.items():
            # Check for non-finite weights (NaN, Inf)
            if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                non_finite_count = (~torch.isfinite(tensor)).sum().item()
                logger.warning("expert group verification failed: non-finite weights detected", key=name, non_finite_count=non_finite_count, total_elements=tensor.numel())
                self.expert_group_verified = False
                return self.expert_group_verified

            layer_id, expert_id = get_layer_expert_id(name)
            if layer_id is None or expert_id is None:
                continue

            routed_expert_key_count += 1

            expert_id_mapping = allowed_layers.get(layer_id, [])
            allowed_expert_ids = {int(my_expert_id) for my_expert_id, _ in expert_id_mapping} | {
                int(org_expert_id) for _, org_expert_id in expert_id_mapping
            }
            if int(expert_id) not in allowed_expert_ids:
                logger.debug(
                    "Expert group verification mismatch",
                    expert_group=self.expert_group,
                    layer_id=layer_id,
                    expert_id=int(expert_id),
                    allowed_expert_id_sample=sorted(list(allowed_expert_ids))[:10],
                    param_name=name,
                )
                self.expert_group_verified = False
                return self.expert_group_verified

        if routed_expert_key_count == 0:
            logger.warning(
                "expert group verification failed: no routed expert params in submitted checkpoint",
                expert_group=self.expert_group,
                hotkey=self.hotkey,
            )
            self.expert_group_verified = False
            return self.expert_group_verified

        self.expert_group_verified = True
        
        return self.expert_group_verified

    def validate(self, expert_group_assignment: ExpertAssignments | None) -> bool:

        # --- verify signature ---
        self._verify_signature()

        # --- get state dict ---
        if self.path is None:
            logger.warning("Hash verification failed: missing checkpoint path", checkpoint=self)
            return False
    
        state_dict = compile_full_state_dict_from_path(self.path, expert_groups=[self.expert_group])
        
        # --- verify hash ---
        self._verify_hash(state_dict = state_dict)

        # --- verify expert group ---
        if expert_group_assignment is not None:
            self._verify_expert_group(state_dict = state_dict, expert_group_assignment = expert_group_assignment)
        
        return self.validated()
    
    def validated(self) -> bool:
        if self.expired():
            hk = self.hotkey[:6] if self.hotkey else None
            logger.debug("Checkpoint rejected — expired", hotkey=hk)
            return False
        if self.signature_required and not self.signature_verified:
            hk = self.hotkey[:6] if self.hotkey else None
            logger.debug("Checkpoint rejected — signature not verified", hotkey=hk)
            return False
        if self.hash_required and not self.hash_verified:
            hk = self.hotkey[:6] if self.hotkey else None
            logger.debug("Checkpoint rejected — hash not verified", hotkey=hk)
            return False
        if self.expert_group_check_required and not self.expert_group_verified:
            hk = self.hotkey[:6] if self.hotkey else None
            logger.debug("Checkpoint rejected — expert group not verified", hotkey=hk)
            return False

        return True

    def active(self) -> bool:
        return True


class ChainCheckpoint(ModelCheckpoint):
    uid: int | None = Field(default=None, alias="h")
    ip: str | None = None
    port: int | None = None
    hotkey: str | None = None
    stake: float = 0.0
    hf_repo_id: str | None = None
    hf_revision: str | None = None

    def __init__(self, **data: Any):
        data.setdefault("place", "onchain")
        super().__init__(**data)

    def get_signed_hash_commit(self) -> dict[str, Any] | None:
        if self.signed_model_hash is None:
            return None
        return self.commit_signature()

    def get_hash_commit(self) -> dict[str, Any] | None:
        if self.model_hash is None:
            return None
        return self.commit_hash()

    def priority(self) -> tuple[int, int, int, int, int]:
        return (
            1 if self.active() else 0,
            1 if self.signature_verified else 0,
            1 if self.hash_verified else 0,
            self.global_ver,
            self.inner_opt,
        )


class ChainCheckpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoints: list[ChainCheckpoint]

    def __len__(self) -> int:
        return len(self.checkpoints)

    def get(self, hotkey: str) -> ChainCheckpoint | None:
        for ckpt in self.checkpoints:
            if ckpt.hotkey == hotkey:
                return ckpt  
            
        return None

    def filter_checkpoints(
        self,
        for_role: str = "validator",
        owner_hotkey: str | None = None,
        min_allowed_version: int | None = None,
        max_allowed_version: int | None = None,
    ) -> ChainCheckpoints:
        logger.debug(
            "filter_checkpoints: start",
            for_role=for_role,
            total_checkpoints=len(self.checkpoints),
            min_allowed_version=min_allowed_version,
            max_allowed_version=max_allowed_version,
        )

        # filter out incomplete checkpoints
        filtered = []
        for ckpt in self.checkpoints:
            missing = []
            if ckpt.signed_model_hash is None:
                missing.append("signed_model_hash")
            if ckpt.model_hash is None:
                missing.append("model_hash")
            if ckpt.global_ver is None:
                missing.append("global_ver")
            if ckpt.expert_group is None:
                missing.append("expert_group")
            if ckpt.uid is None:
                missing.append("uid")
            if ckpt.ip is None:
                missing.append("ip")
            if ckpt.port is None:
                missing.append("port")
            if ckpt.hotkey is None:
                missing.append("hotkey")

            if missing:
                logger.info(
                    "filter_checkpoints: excluded (incomplete)",
                    hotkey=ckpt.hotkey,
                    uid=ckpt.uid,
                    missing_fields=missing,
                )
                continue

            filtered.append(ckpt)

        excluded = len(self.checkpoints) - len(filtered)
        if not filtered and excluded:
            logger.warning(
                "filter_checkpoints: all checkpoints excluded by completeness gate",
                excluded=excluded,
            )
        else:
            logger.debug(
                "filter_checkpoints: after completeness gate",
                passed=len(filtered),
                excluded=excluded,
            )

        if not filtered:
            return ChainCheckpoints(checkpoints=[])

        # reject checkpoints with global_ver outside the allowed range
        if min_allowed_version is not None or max_allowed_version is not None:
            before_count = len(filtered)
            version_ok = []
            for ckpt in filtered:
                ver = ckpt.global_ver
                if ver is not None and max_allowed_version is not None and ver > max_allowed_version:
                    logger.warning(
                        "filter_checkpoints: excluded (version exceeds max allowed)",
                        hotkey=ckpt.hotkey,
                        uid=ckpt.uid,
                        ckpt_ver=ver,
                        max_allowed_version=max_allowed_version,
                    )
                elif ver is not None and min_allowed_version is not None and ver < min_allowed_version:
                    logger.info(
                        "filter_checkpoints: excluded (version too old)",
                        hotkey=ckpt.hotkey,
                        uid=ckpt.uid,
                        ckpt_ver=ver,
                        min_allowed_version=min_allowed_version,
                    )
                else:
                    version_ok.append(ckpt)
            filtered = version_ok
            if not filtered and before_count:
                logger.warning(
                    "filter_checkpoints: all checkpoints excluded by version range gate",
                    excluded=before_count,
                    min_allowed_version=min_allowed_version,
                    max_allowed_version=max_allowed_version,
                )
            else:
                logger.debug(
                    "filter_checkpoints: after version range gate",
                    passed=len(filtered),
                    excluded=before_count - len(filtered),
                    min_allowed_version=min_allowed_version,
                    max_allowed_version=max_allowed_version,
                )
            if not filtered:
                return ChainCheckpoints(checkpoints=[])

        if for_role == "miner":
            return ChainCheckpoints(checkpoints=filtered)


        # select majority model_hash (stake-weighted)
        hash_stake: dict[str, float] = {}
        for ckpt in filtered:
            if ckpt.model_hash:
                hash_stake[ckpt.model_hash] = hash_stake.get(ckpt.model_hash, 0.0) + ckpt.stake
        if not hash_stake:
            logger.warning("filter_checkpoints: no model hashes found after version filter")
            return ChainCheckpoints(checkpoints=[])

        majority_hash = max(hash_stake, key=hash_stake.get)
        logger.info(
            "Majority model hash selected",
            majority_hash=majority_hash[:6],
            majority_stake=round(hash_stake[majority_hash], 4),
            competing_hashes=len(hash_stake),
            stake_distribution={h[:6]: round(s, 4) for h, s in hash_stake.items()},
        )

        majority_filtered = []
        for ckpt in filtered:
            if ckpt.model_hash == majority_hash:
                majority_filtered.append(ckpt)
            else:
                logger.debug(
                    "filter_checkpoints: excluded (non-majority hash)",
                    hotkey=ckpt.hotkey,
                    uid=ckpt.uid,
                    ckpt_hash=ckpt.model_hash,
                    majority_hash=majority_hash,
                    ckpt_stake=ckpt.stake,
                )

        logger.debug(
            "Majority hash filter complete",
            passed=len(majority_filtered),
            excluded=len(filtered) - len(majority_filtered),
        )

        return ChainCheckpoints(checkpoints=majority_filtered)

    def renew(self) -> None:
        before = len(self.checkpoints)
        self.checkpoints = [ckpt for ckpt in self.checkpoints if not ckpt.expired()]
        after = len(self.checkpoints)
        if after != before:
            logger.debug("Expired checkpoints removed", removed=before - after, remaining=after)

    def get_signed_hash_commit(self) -> dict[str, Any] | None:
        ordered = sorted(self.checkpoints, key=lambda ckpt: ckpt.priority(), reverse=True)
        for ckpt in ordered:
            commit = ckpt.get_signed_hash_commit()
            if commit is not None:
                return commit
        return None

    def get_hash_commit(self) -> dict[str, Any] | None:
        ordered = sorted(self.checkpoints, key=lambda ckpt: ckpt.priority(), reverse=True)
        for ckpt in ordered:
            commit = ckpt.get_hash_commit()
            if commit is not None:
                return commit
        return None


class ModelCheckpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoints: list[ModelCheckpoint]

    def ordered(self) -> list[ModelCheckpoint]:
        return sorted(
            self.checkpoints,
            key=lambda ckpt: (1 if ckpt.active() else 0, ckpt.global_ver, ckpt.inner_opt),
            reverse=True,
        )


class Checkpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    local: ModelCheckpoints
    chain: ChainCheckpoints

    def ordered(self) -> list[ModelCheckpoint]:
        return sorted(
            [*self.local.checkpoints, *self.chain.checkpoints],
            key=lambda ckpt: (1 if ckpt.active() else 0, ckpt.global_ver, ckpt.inner_opt),
            reverse=True,
        )


def build_local_checkpoint(path: Path, role: str = "miner") -> ModelCheckpoint | None:
    if path.name.startswith(".tmp_") or "yaml" in path.name.lower():
        return None

    meta = parse_dynamic_filename(str(path))
    if meta is None:
        return None

    return ModelCheckpoint(
        global_ver=int(meta.get("globalver", 0)),
        inner_opt=int(meta.get("inneropt", 0)),
        path=path,
        role=role,
        place="local",
    )


def build_local_checkpoints(ckpt_dir: Path, role: str = "miner") -> ModelCheckpoints:
    fs, root = fsspec.core.url_to_fs(str(ckpt_dir))
    checkpoints: list[ModelCheckpoint] = []

    for entry in fs.ls(root, detail=False):
        path = Path(entry)
        if path.name.startswith(".tmp_"):
            continue
        if "yaml" in path.name.lower():
            continue

        meta = parse_dynamic_filename(str(path))
        if meta is None:
            continue

        checkpoints.append(
            ModelCheckpoint(
                global_ver=int(meta.get("globalver", 0)),
                inner_opt=int(meta.get("inneropt", 0)),
                path=path,
                role=role,
                place="local",
            )
        )

    return ModelCheckpoints(checkpoints=checkpoints)


def build_chain_checkpoints(
    signed_hash_chain_commits: list[tuple[Any, Any]],
    hash_chain_commits: list[tuple[Any, Any]],
    for_role: str = "validator",
    owner_hotkey: str | None = None,
    min_allowed_version: int | None = None,
    max_allowed_version: int | None = None,
) -> ChainCheckpoints:
    """
    Build chain checkpoints by joining signed-hash commits with hash commits.
    """
    signed_by_hotkey: dict[str, str] = {}
    for commit, neuron in signed_hash_chain_commits:
        try:
            hotkey = getattr(neuron, "hotkey", None)
            signed = getattr(commit, "signed_model_hash", None)
            if hotkey and signed:
                signed_by_hotkey[hotkey] = signed
            else:
                logger.debug("Skipping signed hash commit: missing hotkey or signed_model_hash", hotkey=hotkey, signed=signed)
        except Exception as e:
            logger.warning("Cannot read signed hash commit", commit=commit, error=str(e))

    if not signed_by_hotkey:
        logger.warning("No signed hash commits indexed")
    else:
        logger.debug(
            "Signed hash commits indexed",
            signed_hotkey_count=len(signed_by_hotkey),
            signed_hotkeys=list(signed_by_hotkey.keys()),
        )

    checkpoints: list[ChainCheckpoint] = []
    for commit, neuron in hash_chain_commits:
        try:
            hotkey = getattr(neuron, "hotkey", None)
            checkpoints.append(
                ChainCheckpoint(
                    signed_model_hash=signed_by_hotkey.get(hotkey) if hotkey else None,
                    model_hash=getattr(commit, "model_hash", None),
                    global_ver=getattr(commit, "global_ver", None),
                    expert_group=getattr(commit, "expert_group", None),
                    inner_opt=getattr(commit, "inner_opt", None),
                    uid=getattr(neuron, "uid", None),
                    ip=getattr(neuron.axon_info, "ip", None),
                    port=getattr(neuron.axon_info, "port", None),
                    hotkey=hotkey,
                    stake=float(getattr(neuron, "stake", 0.0)),
                    hf_repo_id=getattr(commit, "hf_repo_id", None),
                    hf_revision=getattr(commit, "hf_revision", None),
                    signature_required=True,
                    hash_required=True,
                    expert_group_check_required=True,
                )
            )
        except Exception as e:
            logger.warning("Cannot append commit", commit=commit, error=str(e))

    if not checkpoints:
        logger.warning(
            "No pre-filter checkpoints built",
            for_role=for_role,
        )
    else:
        logger.debug(
            "Pre-filter checkpoints built",
            total_pre_filter=len(checkpoints),
            hotkeys_pre_filter=[ckpt.hotkey for ckpt in checkpoints],
            for_role=for_role,
        )

    filtered_checkpoints = ChainCheckpoints(checkpoints=checkpoints).filter_checkpoints(
        for_role=for_role, owner_hotkey=owner_hotkey,
        min_allowed_version=min_allowed_version, max_allowed_version=max_allowed_version,
    )

    if not filtered_checkpoints:
        logger.warning("No checkpoints remain after filtering", for_role=for_role)
    else:
        logger.debug(
            "Post-filter checkpoints",
            total_post_filter=len(filtered_checkpoints),
            hotkeys_post_filter=[ckpt.hotkey for ckpt in filtered_checkpoints.checkpoints],
            for_role=for_role,
        )

    if len(filtered_checkpoints) == 0:
        return ChainCheckpoints(checkpoints=[])
    return filtered_checkpoints

def build_chain_checkpoints_from_previous_phase(
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    for_role: str = "validator",
    owner_hotkey: str | None = None,
) -> ChainCheckpoints:
    logger.debug("Building chain checkpoints from previous phase", for_role=for_role)
    
    # --- Validate type ---
    if for_role == "miner":
        phase_name_1 = PhaseNames.miner_commit_1
        phase_name_2 = PhaseNames.miner_commit_2
        next_phase = PhaseNames.submission
        
    elif for_role == "validator":
        phase_name_1 = PhaseNames.validator_commit_1
        phase_name_2 = PhaseNames.validator_commit_2
        next_phase = PhaseNames.distribute

    else:
        raise ValueError(f"Invalid type: {for_role}. Must be 'miner' or 'validator'.")

    # --- Make sure we are not inbetween commit 1 and 2---    
    current_phase: PhaseResponse | None = get_phase_from_api(config)
    if current_phase is not None and (current_phase.phase_name == phase_name_1 or current_phase.phase_name == phase_name_2):
        logger.info(f"In between hash commit phase, waiting till {next_phase}")
        wait_till(config, next_phase)
        
    # --- Get block ranges for previous phases ---
    previous_phase_range = get_blocks_from_previous_phase_from_api(config)

    if previous_phase_range is not None:
        commit_1_end_block = previous_phase_range[phase_name_1][1] + 1
        commit_2_end_block = previous_phase_range[phase_name_2][1] + 1
        logger.debug(
            "Previous phase range resolved",
            for_role=for_role,
            phase_name_1=phase_name_1,
            phase_name_2=phase_name_2,
            commit_1_end_block=commit_1_end_block,
            commit_2_end_block=commit_2_end_block,
            previous_phase_range=previous_phase_range,
        )

        # --- Get commits from chain at the right blocks ---
        signed_hash_chain_commits: tuple[SignedModelHashChainCommit, bittensor.Neuron] = get_chain_commits(
            config, subtensor, block=commit_1_end_block, signature_commit=True
        )
        hash_chain_commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(
            config, subtensor, block=commit_2_end_block
        )
        if not signed_hash_chain_commits or not hash_chain_commits:
            logger.warning(
                "Chain commits fetched but some are missing",
                for_role=for_role,
                signed_hash_count=len(signed_hash_chain_commits),
                hash_count=len(hash_chain_commits),
            )
        else:
            logger.debug(
                "Chain commits fetched",
                for_role=for_role,
                signed_hash_count=len(signed_hash_chain_commits),
                hash_count=len(hash_chain_commits),
            )

    else:
        signed_hash_chain_commits = []
        hash_chain_commits = []
        logger.warning(
            "Previous phase range unavailable; no chain commits fetched",
            for_role=for_role,
        )

    # --- Build chain checkpoints ---
    min_ver, max_ver = get_allowed_version_range(config)
    return build_chain_checkpoints(
        signed_hash_chain_commits=signed_hash_chain_commits,
        hash_chain_commits=hash_chain_commits,
        for_role=for_role,
        owner_hotkey=owner_hotkey,
        min_allowed_version=min_ver,
        max_allowed_version=max_ver,
    )

def delete_old_checkpoints(checkpoint_path: str | Path, topk: int) -> list[str]:
    """
    Deletes old checkpoints, keeping only the top 'k' most recent ones.
    """
    fs = GenericFileSystem()
    sorted_ckpt_files = build_local_checkpoints(checkpoint_path).ordered()

    ckpt_deleted = []
    for model_meta in sorted_ckpt_files[topk:]:
        fs.rm(str(model_meta.path), recursive=True)
        ckpt_deleted.append(str(model_meta.path))
    return ckpt_deleted


def prune_miner_submission_files(
    folder_path: Path,
    current_block: int,
    cycle_length: int,
    max_age_cycles: float = 1.5,
) -> list[str]:
    """
    Delete miner submission files older than the allowed history window.

    Files are identified by the embedded `block` in their filename. Any file
    older than `max_age_cycles * cycle_length` relative to `current_block` is
    removed.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    max_age_blocks = max(0, int(cycle_length * max_age_cycles))
    min_allowed_block = current_block - max_age_blocks

    for file_path in folder_path.glob("*.pt"):
        meta = parse_dynamic_filename(file_path.name)
        if "hotkey" not in meta or "block" not in meta:
            logger.warning("Skipping malformed submission filename", file=file_path.name)
            continue

    deleted_files: list[str] = []
    for file_path in folder_path.glob("*.pt"):
        meta = parse_dynamic_filename(file_path.name)
        if "hotkey" not in meta or "block" not in meta:
            continue

        hotkey = meta["hotkey"]
        block = meta["block"]
        if not isinstance(block, int):
            logger.warning("Skipping submission with non-integer block", file=file_path.name, block=block)
            continue
        if block > min_allowed_block:
            continue

        try:
            os.remove(file_path)
            deleted_files.append(file_path.name)
        except Exception as exc:
            logger.warning("Failed to delete aged submission file", file=file_path.name, error=str(exc), hotkey=hotkey)

    if deleted_files:
        logger.info(
            "Deleted aged submissions",
            count=len(deleted_files),
            files=deleted_files,
            current_block=current_block,
            cycle_length=cycle_length,
            max_age_cycles=max_age_cycles,
            min_allowed_block=min_allowed_block,
        )
    else:
        logger.debug(
            "No aged submissions to delete",
            current_block=current_block,
            cycle_length=cycle_length,
            max_age_cycles=max_age_cycles,
            min_allowed_block=min_allowed_block,
        )

    return deleted_files


def delete_old_checkpoints_by_hotkey(
    folder_path: Path,
    current_block: int,
    cycle_length: int,
    max_age_cycles: float = 1.5,
) -> list[str]:
    """
    Backward-compatible wrapper for miner submission pruning.
    """
    return prune_miner_submission_files(
        folder_path,
        current_block=current_block,
        cycle_length=cycle_length,
        max_age_cycles=max_age_cycles,
    )


def archive_top_miner_submissions(
    submission_dir: Path,
    archive_dir: Path,
    score_aggregator,
    top_k: int | None = None,
    max_archive: int = 500,
) -> None:
    """
    Archive the top-k and 25th-percentile miner submissions to a permanent folder.
    Files are renamed with a rank prefix (rank01_best, rank02, ..., p25) so the
    quality is visible from the filename. Delete all other submissions.
    Prune archive to max_archive files.
    """
    submission_dir = Path(submission_dir)
    archive_dir = Path(archive_dir)
    if not submission_dir.exists():
        return

    archive_dir.mkdir(parents=True, exist_ok=True)

    # Collect all submission files with their hotkey
    submissions: dict[str, Path] = {}
    for file_path in submission_dir.glob("*.pt"):
        meta = parse_dynamic_filename(file_path.name)
        hotkey = meta.get("hotkey")
        if hotkey:
            submissions[hotkey] = file_path

    if not submissions:
        return

    # Get scores for ranking
    uid_scores = score_aggregator.uid_score_pairs(how="avg")

    # Map hotkey -> best score (lower val_loss = better)
    hotkey_scores: dict[str, float] = {}
    for hotkey in submissions:
        for uid, score in uid_scores.items():
            state = score_aggregator._miners.get(uid)
            if state and state.hotkey == hotkey:
                hotkey_scores[hotkey] = score
                break

    # Rank by score (lower val_loss is better)
    ranked_hotkeys = sorted(
        hotkey_scores.keys(),
        key=lambda hk: hotkey_scores.get(hk, float("inf")),
    )

    # Determine which hotkeys to archive: best + 25th percentile
    best_hotkey = ranked_hotkeys[0] if ranked_hotkeys else None
    p25_idx = max(0, len(ranked_hotkeys) - 1) * 3 // 4
    p25_hotkey = ranked_hotkeys[p25_idx] if ranked_hotkeys else None

    archive_hotkeys: dict[str, str] = {}
    if best_hotkey:
        archive_hotkeys[best_hotkey] = "best"
    if p25_hotkey and p25_hotkey != best_hotkey:
        archive_hotkeys[p25_hotkey] = "p25"

    archived = []
    deleted = []
    for hotkey, file_path in submissions.items():
        if hotkey in archive_hotkeys:
            label = archive_hotkeys[hotkey]
            score_str = f"{hotkey_scores.get(hotkey, 0):.4f}"
            stem = file_path.stem
            dest_name = f"{stem}_rank_{label}_loss_{score_str}.pt"
            dest = archive_dir / dest_name
            shutil.move(str(file_path), str(dest))
            archived.append(dest_name)
        else:
            file_path.unlink(missing_ok=True)
            deleted.append(file_path.name)

    if archived:
        logger.info("Archived miner submissions", count=len(archived), top_k=top_k, files=archived)
    if deleted:
        logger.debug("Deleted non-archived miner submissions", count=len(deleted))

    # Prune archive to max_archive files (keep newest by modification time)
    archive_files = sorted(archive_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
    if len(archive_files) > max_archive:
        pruned = []
        for old_file in archive_files[max_archive:]:
            old_file.unlink(missing_ok=True)
            pruned.append(old_file.name)
        logger.debug("Pruned old archive files", count=len(pruned), max_archive=max_archive)


def select_best_checkpoint(
    primary_dir: Path, secondary_dir: Path | None = None, resume: bool = True
) -> ModelCheckpoint | None:
    if not resume:
        return None

    primary = build_local_checkpoints(primary_dir, role="miner")

    if secondary_dir is None:
        combined = Checkpoints(
            local=primary,
            chain=ChainCheckpoints(checkpoints=[]),
        )
        ordered = combined.ordered()
        return ordered[0] if ordered else None

    secondary = build_local_checkpoints(secondary_dir, role="validator")
    combined_local = ModelCheckpoints(checkpoints=[*primary.checkpoints, *secondary.checkpoints])

    combined = Checkpoints(
        local=combined_local,
        chain=ChainCheckpoints(checkpoints=[]),
    )

    for ckpt in combined.ordered():
        if ckpt.active():
            return ckpt

    return None

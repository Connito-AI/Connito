import errno
import gc
import os
import shutil
from copy import deepcopy
from pathlib import Path

import fsspec
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from connito.shared.app_logging import structlog
from connito.shared.config import MinerConfig
from connito.shared.expert_manager import (
    ExpertAssignments,
    ExpertManager,
    get_layer_expert_id,
)
from connito.shared.helper import (
    MINER_CHECKPOINT_SUFFIXES,
    get_model_hash,
    load_state_dict_from_path,
)

logger = structlog.getLogger(__name__)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f}{units[unit_idx]}"


def cleanup_temporary_checkpoint_dirs(checkpoint_root: str | Path) -> list[str]:
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        return []

    removed: list[str] = []
    for path in checkpoint_root.iterdir():
        if not path.name.startswith(".tmp_"):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        removed.append(str(path))

    if removed:
        logger.warning("Removed stale temporary checkpoints", count=len(removed), paths=removed)

    return removed


# @total_ordering
# @dataclass
# class ModelCheckpoint:
#     global_ver: int = 0
#     inner_opt: int = 0
#     path: Path | None = None
#     role: str | None = None  # [miner, validator]
#     model_hash: str | None = None

#     def __eq__(self, other: object) -> bool:
#         try:
#             other_global_ver = other.global_ver  # type: ignore[attr-defined]
#             other_inner_opt = other.inner_opt  # type: ignore[attr-defined]
#             other_model_hash = getattr(other, "model_hash", None)
#         except AttributeError:
#             return NotImplemented
#         return (
#             self.global_ver == other_global_ver
#             and self.inner_opt == other_inner_opt
#             and self.model_hash == other_model_hash
#         )

#     def __lt__(self, other: "ModelCheckpoint") -> bool:
#         try:
#             other_global_ver = other.global_ver  # type: ignore[attr-defined]
#             other_inner_opt = other.inner_opt  # type: ignore[attr-defined]
#         except AttributeError:
#             return NotImplemented

#         # Compare by global_ver first
#         if self.global_ver != other_global_ver:
#             return self.global_ver < other_global_ver

#         # Then compare by inner_opt
#         return self.inner_opt < other_inner_opt


# ====================
# Checkpoint saving / loading
# ====================
def save_state_dict_by_expert_group(
    state_dict: dict[str, torch.Tensor],
    expert_groups: ExpertAssignments,
    save_dir: str | Path,
    strict_sharding: bool = False,
    active_expert_group_id: int | None = None,
    save_dtype: torch.dtype = torch.float16,
):
    """
    Split a full model state_dict into checkpoint shards.

    Modes:
        - active_expert_group_id is None:
            write one shard per expert group plus model_shared.pt.
        - active_expert_group_id is set:
            write model_expgroup_{id}.pt with experts owned by that group only,
            plus model_shared.pt containing all shared/non-expert params.
    """

    os.makedirs(save_dir, exist_ok=True)

    logger.debug(
        "save_state_dict_by_expert_group called",
        save_dir=str(save_dir),
        total_params=len(state_dict),
        expert_group_ids=list(expert_groups.keys()),
        active_expert_group_id=active_expert_group_id,
    )

    if active_expert_group_id is not None and active_expert_group_id not in expert_groups:
        raise ValueError(
            f"active_expert_group_id={active_expert_group_id} is not present in expert_groups="
            f"{list(expert_groups.keys())}"
        )

    group_ids = [active_expert_group_id] if active_expert_group_id is not None else list(expert_groups.keys())

    has_shared_bucket = True

    # output buckets
    grouped_state = {gid: {} for gid in group_ids}
    if has_shared_bucket:
        grouped_state["shared"] = {}

    # Build fast-lookup structures:
    # - org_expert_lookup uses original/global expert IDs (preferred, unambiguous)
    # - my_expert_lookup is a fallback for backward compatibility with older
    #   checkpoints that serialized local my_expert_idx IDs.
    org_expert_lookup: dict[tuple[int, int], int] = {}
    my_expert_lookup: dict[tuple[int, int], int] = {}
    ambiguous_my_expert_keys: set[tuple[int, int]] = set()
    selected_expert_groups = {gid: expert_groups[gid] for gid in group_ids}
    duplicate_org_assignments: list[tuple[int, int, int, int]] = []
    for gid, layer_map in selected_expert_groups.items():
        for layer_id, mappings in layer_map.items():
            for my_eid, org_eid in mappings:
                org_key = (int(layer_id), int(org_eid))
                prev_org_gid = org_expert_lookup.get(org_key)
                if prev_org_gid is not None and prev_org_gid != gid:
                    duplicate_org_assignments.append((int(layer_id), int(org_eid), int(prev_org_gid), int(gid)))
                    logger.warning(
                        "Duplicate org expert assignment across groups",
                        layer_id=layer_id,
                        expert_id=int(org_eid),
                        previous_group_id=prev_org_gid,
                        group_id=gid,
                    )
                    # Keep the first assignment stable unless strict mode requests hard failure.
                    continue
                org_expert_lookup[org_key] = gid

                my_key = (int(layer_id), int(my_eid))
                prev_my_gid = my_expert_lookup.get(my_key)
                if prev_my_gid is None:
                    my_expert_lookup[my_key] = gid
                elif prev_my_gid != gid:
                    ambiguous_my_expert_keys.add(my_key)

    for key in ambiguous_my_expert_keys:
        my_expert_lookup.pop(key, None)

    if strict_sharding and duplicate_org_assignments:
        preview = ", ".join(
            f"(layer={layer}, expert={expert}, first_gid={first_gid}, dup_gid={dup_gid})"
            for layer, expert, first_gid, dup_gid in duplicate_org_assignments[:8]
        )
        raise ValueError(
            "Invalid expert group assignment: duplicate org expert IDs assigned across groups. "
            f"Sample duplicates: {preview}"
        )

    logger.debug(
        "Expert lookup built",
        org_lookup_size=len(org_expert_lookup),
        my_lookup_size=len(my_expert_lookup),
        ambiguous_my_keys=len(ambiguous_my_expert_keys),
    )

    active_my_ids_by_layer: dict[int, set[int]] = {}
    active_org_ids_by_layer: dict[int, set[int]] = {}
    if active_expert_group_id is not None:
        for layer_id, mappings in selected_expert_groups[active_expert_group_id].items():
            active_my_ids_by_layer[int(layer_id)] = {int(my_id) for my_id, _ in mappings}
            active_org_ids_by_layer[int(layer_id)] = {int(org_id) for _, org_id in mappings}

    # Track per-group counts and bytes for debugging
    group_param_count: dict[int | str, int] = {gid: 0 for gid in group_ids}
    group_bytes: dict[int | str, int] = {gid: 0 for gid in group_ids}
    if has_shared_bucket:
        group_param_count["shared"] = 0
        group_bytes["shared"] = 0
    unassigned_experts: list[str] = []
    skipped_experts_samples: list[str] = []
    skipped_experts_count = 0
    detected_expert_param_count = 0
    assigned_expert_param_count = 0

    # Iterate model weights
    # Convert only selected tensors to CPU fp16 (post-filter) to keep peak RAM low.
    # This avoids materializing a full CPU copy before sharding.
    for name, tensor in state_dict.items():
        layer_id, expert_id = get_layer_expert_id(name)
        # CASE 1: Not an expert parameter
        if layer_id is None or expert_id is None:
            t = tensor.detach().to(dtype=save_dtype, device="cpu", non_blocking=True).contiguous()
            grouped_state["shared"][name] = t
            group_param_count["shared"] += 1
            group_bytes["shared"] += t.nelement() * t.element_size()
            continue

        detected_expert_param_count += 1

        # CASE 2 (active group mode): include only experts owned by active_expert_group_id.
        if active_expert_group_id is not None:
            layer_key = int(layer_id)
            expert_key = int(expert_id)
            allowed_my = active_my_ids_by_layer.get(layer_key, set())
            allowed_org = active_org_ids_by_layer.get(layer_key, set())
            if expert_key in allowed_my or expert_key in allowed_org:
                t = tensor.detach().to(dtype=save_dtype, device="cpu", non_blocking=True).contiguous()
                grouped_state[active_expert_group_id][name] = t
                group_param_count[active_expert_group_id] += 1
                group_bytes[active_expert_group_id] += t.nelement() * t.element_size()
                assigned_expert_param_count += 1
            else:
                skipped_experts_count += 1
                if len(skipped_experts_samples) < 10:
                    skipped_experts_samples.append(f"{name} (layer={layer_id}, expert={expert_id})")
            continue

        # CASE 3: Check if this expert (layer_id, expert_id) belongs to any group.
        # Prefer org/global expert IDs; use local-id fallback only when unique.
        key = (int(layer_id), int(expert_id))
        gid = org_expert_lookup.get(key, None)
        if gid is None:
            gid = my_expert_lookup.get(key, None)

        if gid is None:
            # expert exists but not assigned to any group.
            # In full-shard mode these go to shared for backward compatibility.
            if has_shared_bucket:
                t = tensor.detach().to(dtype=save_dtype, device="cpu", non_blocking=True).contiguous()
                grouped_state["shared"][name] = t
                group_param_count["shared"] += 1
                group_bytes["shared"] += t.nelement() * t.element_size()
                reason = "ambiguous_local_id" if key in ambiguous_my_expert_keys else "unmapped"
                unassigned_experts.append(f"{name} (layer={layer_id}, expert={expert_id}, reason={reason})")
        else:
            t = tensor.detach().to(dtype=save_dtype, device="cpu", non_blocking=True).contiguous()
            grouped_state[gid][name] = t
            group_param_count[gid] += 1
            group_bytes[gid] += t.nelement() * t.element_size()
            assigned_expert_param_count += 1

    # Validate no overlap of expert keys across groups and no expert keys in shared.
    expert_key_owner: dict[str, int] = {}
    overlap_errors: list[str] = []
    shared_expert_keys: list[str] = []
    for gid, sd in grouped_state.items():
        for param_name in sd:
            layer_id, expert_id = get_layer_expert_id(param_name)
            is_expert = layer_id is not None and expert_id is not None
            if gid == "shared" and is_expert:
                shared_expert_keys.append(param_name)
                continue
            if gid != "shared" and is_expert:
                owner = expert_key_owner.get(param_name)
                if owner is None:
                    expert_key_owner[param_name] = gid
                elif owner != gid:
                    overlap_errors.append(f"{param_name}: groups {owner} and {gid}")

    if overlap_errors:
        raise ValueError(f"Expert param overlap across shards detected: {overlap_errors[:10]}")

    # Log summary per group
    for gid in grouped_state:
        logger.debug(
            "Group split summary",
            gid=gid,
            param_count=group_param_count[gid],
            size_mb=round(group_bytes[gid] / (1024 * 1024), 2),
        )

    if unassigned_experts:
        if strict_sharding:
            preview = "; ".join(unassigned_experts[:10])
            raise ValueError(
                "Strict sharding violation: some expert params are unmapped/ambiguous and would go to shared. "
                f"count={len(unassigned_experts)} samples={preview}"
            )
        logger.warning(
            "Expert params not assigned to any group (went to shared)",
            count=len(unassigned_experts),
            samples=unassigned_experts[:10],
        )

    if detected_expert_param_count == 0:
        logger.warning(
            "No expert params detected while splitting checkpoint; expert-group shards may be empty",
            total_params=len(state_dict),
            group_ids=list(expert_groups.keys()),
            active_expert_group_id=active_expert_group_id,
        )

    if detected_expert_param_count > 0 and assigned_expert_param_count == 0:
        logger.warning(
            "Detected expert params but none assigned to output shard(s); expert-group shard may be empty",
            detected_expert_params=detected_expert_param_count,
            active_expert_group_id=active_expert_group_id,
            group_ids=group_ids,
        )

    # Contract validation: shared must contain no expert params; each expert-group file
    # must contain expert params only.
    shared_expert_keys = [
        name for name in grouped_state["shared"] if get_layer_expert_id(name)[1] is not None
    ]
    if strict_sharding and shared_expert_keys:
        raise ValueError(
            "Strict sharding violation: model_shared.pt would contain expert params. "
            f"samples={shared_expert_keys[:10]}"
        )

    for gid in group_ids:
        non_expert_keys = [
            name
            for name in grouped_state[gid]
            if get_layer_expert_id(name)[0] is None or get_layer_expert_id(name)[1] is None
        ]
        if strict_sharding and non_expert_keys:
            raise ValueError(
                "Strict sharding violation: expert-group shard contains non-expert params. "
                f"gid={gid} samples={non_expert_keys[:10]}"
            )
        if strict_sharding and len(grouped_state[gid]) == 0:
            raise ValueError(
                "Strict sharding violation: expert-group shard is empty. "
                f"gid={gid}"
            )

    if skipped_experts_count > 0:
        logger.info(
            "Skipped expert params not owned by active expert group",
            active_expert_group_id=active_expert_group_id,
            count=skipped_experts_count,
            samples=skipped_experts_samples,
        )

    # Save the groups
    paths = {}
    for gid, sd in grouped_state.items():
        fname = f"model_expgroup_{gid}.pt" if gid != "shared" else "model_shared.pt"
        path = os.path.join(save_dir, fname)

        estimated_bytes = int(group_bytes[gid] * 1.05) + (64 * 1024 * 1024)
        free_bytes = shutil.disk_usage(save_dir).free
        if free_bytes < estimated_bytes:
            raise RuntimeError(
                "Insufficient disk space before checkpoint write: "
                f"gid={gid}, path={path}, estimated_required={_format_bytes(estimated_bytes)}, "
                f"free={_format_bytes(free_bytes)}"
            )

        logger.info(
            "Saving expert group checkpoint",
            group_id=gid,
            param_count=group_param_count[gid],
            size_mb=round(group_bytes[gid] / (1024 * 1024), 2),
            model_hash=get_model_hash(sd, hex=True)[:6],
        )
        try:
            torch.save({"model_state_dict": sd}, path)
        except Exception as e:
            free_bytes_after = shutil.disk_usage(save_dir).free
            raise RuntimeError(
                "Checkpoint shard write failed: "
                f"gid={gid}, path={path}, free_after_failure={_format_bytes(free_bytes_after)}, "
                f"original_error={e}"
            ) from e
        file_size = os.path.getsize(path)
        logger.debug(
            "Expert group checkpoint written",
            group_id=gid,
            file_size_mb=round(file_size / (1024 * 1024), 2),
            path=path,
        )
        paths[gid] = path

    return paths


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    rank: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    inner_optimizer: torch.optim.Optimizer | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    inner_scaler: torch.amp.GradScaler | None = None,
    outer_scaler: torch.amp.GradScaler | None = None,
    loss: float | None = None,
    data_loader: StatefulDataLoader | None = None,
    save_global_state: bool = True,
    save_model_by_expert_group: bool = False,
    expert_manager: ExpertManager | None = None,
    strict_sharding: bool = False,
    active_expert_group_id: int | None = None,
) -> None:
    """
    Saves the current model checkpoint.

    Returns:
        None
    """
    checkpoint_path = Path(checkpoint_path)
    tmp_checkpoint_path = checkpoint_path.with_name(f".tmp_{checkpoint_path.name}")

    cleanup_temporary_checkpoint_dirs(checkpoint_path.parent)

    if tmp_checkpoint_path.exists():
        shutil.rmtree(tmp_checkpoint_path)
    tmp_checkpoint_path.mkdir(parents=True, exist_ok=True)

    write_path = tmp_checkpoint_path

    try:
        # === save model, optimizer ===
        model_dtype = next(model.parameters()).dtype if len(list(model.parameters())) > 0 else torch.float16

        if save_model_by_expert_group and expert_manager is not None:
            state_dict = model.state_dict()
            save_state_dict_by_expert_group(
                state_dict,
                expert_manager.expert_group_assignment,
                write_path,
                strict_sharding=strict_sharding,
                active_expert_group_id=active_expert_group_id,
                save_dtype=model_dtype,
            )
            del state_dict
            gc.collect()

        else:
            checkpoint = {
                "model_state_dict": {k: v.detach().to(dtype=model_dtype, device="cpu", non_blocking=True) for k, v in model.state_dict().items()},
                "loss": loss,
            }
            target = write_path / "model.pt"
            if not target.exists():
                with fsspec.open(str(target), "wb") as f:
                    torch.save(checkpoint, f)

        # === save optimizer ===
        if inner_optimizer is not None:
            opt_checkpoint = {
                "optimizer_state_dict": inner_optimizer.state_dict(),
            }

            target = write_path / "inner_optimizer.pt"
            if not target.exists():
                with fsspec.open(str(target), "wb") as f:
                    torch.save(opt_checkpoint, f)

        if outer_optimizer is not None:
            opt_checkpoint = {
                "optimizer_state_dict": outer_optimizer.state_dict(),
            }
            target = write_path / "outer_optimizer.pt"
            if not target.exists():
                with fsspec.open(str(target), "wb") as f:
                    torch.save(opt_checkpoint, f)

        # === save dataloader ===
        if data_loader is not None:
            rank_state_dict = {}
            rank_state_dict["data_loader"] = data_loader.state_dict()

            target = write_path / f"dataloader_rank{rank}.pt"
            if not target.exists():
                with fsspec.open(str(target), "wb") as f:
                    torch.save(rank_state_dict, f)

            del rank_state_dict

        if save_global_state:
            # === save global state ===
            global_state_dict = {
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "loss": loss if loss is not None else 0,
            }

            if inner_scaler is not None:
                global_state_dict["inner_scaler_state_dict"] = inner_scaler.state_dict()

            if outer_scaler is not None:
                global_state_dict["outer_scaler_state_dict"] = outer_scaler.state_dict()

            target = write_path / "global_state.pt"
            if not target.exists():
                with fsspec.open(str(target), "wb") as f:
                    torch.save(global_state_dict, f)

        try:
            os.replace(write_path, checkpoint_path)
        except OSError as e:
            if checkpoint_path.exists() and e.errno in (errno.EEXIST, errno.ENOTEMPTY):
                shutil.rmtree(checkpoint_path)
                os.replace(write_path, checkpoint_path)
            else:
                raise
    except Exception:
        shutil.rmtree(write_path, ignore_errors=True)
        raise

    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_optimizer(checkpoint_path, optimizer):
    def _get_name_to_id(optimizer_state_dict):
        param_name = [pid for g in optimizer_state_dict["param_groups"] for pid in g["param_names"]]
        param_id = [pid for g in optimizer_state_dict["param_groups"] for pid in g["params"]]
        param_name_to_id = {name: pid for name, pid in zip(param_name, param_id, strict=False)}
        return param_name_to_id

    def _get_name_to_param(optimizer_state_dict):
        """
        Build a mapping: parameter_name -> optimizer_state (for that param).
        Works regardless of the optimizer’s internal param IDs.
        """
        state = optimizer_state_dict["state"]
        name_to_id = _get_name_to_id(optimizer_state_dict)
        name_to_state = {param_name: state[pid] for param_name, pid in name_to_id.items()}
        return name_to_state

    def _update_state_dict(optimizer_state_dict, name_to_param):
        """
        Build a *loadable* optimizer.state_dict() for `target_optimizer` such that:
        - Params that belong to the requested experts get their merged state.
        - All other params are left without state (the optimizer will re-init them).
        """

        optimizer_state_dict["state"] = {}

        target_name_to_id = _get_name_to_id(optimizer_state_dict)

        for pid, name in target_name_to_id.items():
            st = name_to_param.get(name, None)
            if st is not None:
                optimizer_state_dict["state"][pid] = deepcopy(st)  # avoid aliasing

        return optimizer_state_dict

    full_name_to_param = {}
    model_files = fsspec.open_files(checkpoint_path, mode="rb")

    if len(model_files) == 0:
        return

    for f in model_files:
        with f as fh:
            state_dict = torch.load(fh, map_location=torch.device("cpu"))
            full_name_to_param = full_name_to_param | _get_name_to_param(state_dict["optimizer_state_dict"])

    optimizer.load_state_dict(_update_state_dict(optimizer.state_dict(), full_name_to_param))


def compile_full_state_dict_from_path(checkpoint_path, expert_groups: list[int | str] | None = None):
    def _matches_expert_group(file_path: str | Path, groups) -> bool:
        if groups is None:
            return True

        if isinstance(groups, (list, tuple, set)):
            return any(_matches_expert_group(file_path, g) for g in groups)

        filename = Path(file_path).name
        if groups == "shared":
            return filename == "model_shared.pt"

        return filename == f"model_expgroup_{groups}.pt"

    full_state_dict = {}
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file() and checkpoint_path.suffix in MINER_CHECKPOINT_SUFFIXES:
        # Miner submission — could be `.safetensors` (preferred) or `.pt`.
        # `load_state_dict_from_path` dispatches by suffix and gates `.pt`
        # with `weights_only=True` so a malicious miner cannot execute code
        # via a crafted `__reduce__` payload.
        full_state_dict = load_state_dict_from_path(checkpoint_path)
        logger.debug("loaded checkpoint file", path=checkpoint_path)

    else:
        model_files = get_model_files(checkpoint_path)
    
        for f in model_files:
            if expert_groups is not None and not _matches_expert_group(f.path, expert_groups):
                logger.debug("skipping checkpoint file", path=f, expert_groups=expert_groups)
                continue

            with f as fh:
                state_dict = torch.load(fh, map_location=torch.device("cpu"))
                full_state_dict.update(state_dict["model_state_dict"])
                loss = state_dict["loss"] if "loss" in state_dict else -1
                del state_dict
                logger.debug("loaded checkpoint file", path=f, loss=round(loss, 5))

    return full_state_dict


def load_checkpoint(
    checkpoint_path: str,
    config: MinerConfig,
    rank: int | None,
    device: torch.device,
    model: torch.nn.Module | None = None,
    inner_optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    inner_scaler: torch.amp.GradScaler | None = None,
    outer_scaler: torch.amp.GradScaler | None = None,
    data_loader: StatefulDataLoader | None = None,
    expert_groups: list[int | str] | None = None,
) -> float:
    """Load the model and optimizer state from a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
        outer_optimizer: the outer optimizer to load
        data_loader: the data loader to load

    Returns:
        loss: the loss from the checkpoint
    """

    if model is not None:
        precision = getattr(config.model, "precision", "fp16-mixed")
        if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            precision = "fp16-mixed"
        model_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16

        full_state_dict = compile_full_state_dict_from_path(checkpoint_path, expert_groups=expert_groups)
        for key, value in full_state_dict.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and value.dtype != model_dtype:
                full_state_dict[key] = value.to(dtype=model_dtype)
        model.load_state_dict(full_state_dict, strict=False)
        model.to(device=device, dtype=model_dtype)

    if inner_optimizer is not None:
        load_optimizer(os.path.join(checkpoint_path, "inner_optimizer*.pt"), inner_optimizer)

    if outer_optimizer is not None:
        load_optimizer(os.path.join(checkpoint_path, "outer_optimizer*.pt"), outer_optimizer)

    if data_loader is not None:
        with fsspec.open(os.path.join(checkpoint_path, f"dataloader_rank{rank}.pt"), "rb") as f:
            rank_state_dict = torch.load(f, map_location=torch.device("cpu"))
        data_loader.load_state_dict(rank_state_dict["data_loader"])

    if scheduler is not None or inner_scaler is not None or outer_scaler is not None:
        with fsspec.open(os.path.join(checkpoint_path, "global_state.pt"), "rb") as f:
            global_state_dict = torch.load(f, map_location=torch.device("cpu"))
    else:
        return -1

    if scheduler is not None:
        scheduler.load_state_dict(global_state_dict["scheduler"])
        inner_optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]

        # Push optimizer tensors to the same device as the model
        for st in outer_optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    if inner_scaler is not None:
        inner_scaler.load_state_dict(global_state_dict["inner_scaler_state_dict"])

    if outer_scaler is not None:
        outer_scaler.load_state_dict(global_state_dict["outer_scaler_state_dict"])

    return global_state_dict["loss"]


# ====================
# Checkpoint selection
# ====================

# def start_model_from(
#     rank: int, config: MinerConfig, primary_ckpt_path: Path, secondary_ckpt_path: Path | None
# ) -> tuple[bool, ModelCheckpoint, str | Path | None]:
#     # if it is a validator, then just start from its own checkpoint
#     if secondary_ckpt_path is None:
#         logger.info("returning primary checkpoint")
#         return get_resume_info(rank, config, config.ckpt.checkpoint_path)

#     primary_ckpt_found, primary_model_meta, latest_primary_ckpt = get_resume_info(rank, config, primary_ckpt_path)
#     secondary_ckpt_found, secondary_model_meta, latest_secondary_ckpt = get_resume_info(
#         rank, config, secondary_ckpt_path
#     )

#     # --- handling either miner / validator checkpoint not found ---
#     if not secondary_ckpt_found:
#         logger.info(
#             "secondary checkpoint not found, using primary",
#             primary_ckpt_path=primary_ckpt_path,
#             secondary_ckpt_path=secondary_ckpt_path,
#             model_meta=primary_model_meta,
#         )
#         return primary_ckpt_found, primary_model_meta, latest_primary_ckpt

#     if not primary_ckpt_found and latest_secondary_ckpt is not None:
#         logger.info(
#             "primary checkpoint not found, using secondary",
#             primary_ckpt_path=primary_ckpt_path,
#             secondary_ckpt_path=secondary_ckpt_path,
#             model_meta=secondary_model_meta,
#         )
#         return secondary_ckpt_found, secondary_model_meta, latest_secondary_ckpt

#     # --- Return based on more updated version ---
#     if secondary_model_meta >= primary_model_meta and latest_secondary_ckpt is not None:
#         logger.info("Largest local model", secondary_model_meta)
#         return secondary_ckpt_found, secondary_model_meta, latest_secondary_ckpt
#     else:
#         logger.info("Largest local model", primary_model_meta)
#         return primary_ckpt_found, primary_model_meta, latest_primary_ckpt


# def get_resume_info(
#     rank: int, config: MinerConfig | ValidatorConfig, path: Path | None = None, msg: str = ""
# ) -> tuple[bool, ModelCheckpoint, Path | None]:
#     """
#     Retrieves the resume information for a given rank and checkpoint configuration.

#     Args:
#         rank (int): The rank of the process.
#         ckpt_config (Config): The configuration object for the checkpoint.

#     Returns:
#         tuple[bool, int, str | None]: A tuple containing a boolean indicating success,
#         the checkpoint step, and an optional string message.
#     """
#     """
#     Check if we should resume from a checkpoint, if yes return the path to the checkpoint, otherwise return None
#     """
#     if config.ckpt.resume_from_ckpt is None:
#         return False, ModelCheckpoint(), None

#     elif isinstance(config.ckpt.resume_from_ckpt, bool):
#         # Using fsspec to list directory contents
#         try:
#             if path is None:
#                 path = config.ckpt.checkpoint_path

#             ckpt_files = get_sorted_checkpoints(path)

#         except FileNotFoundError:
#             logger.debug(
#                 f"Get resume info from folder {msg}", result="folder not found", path={config.ckpt.checkpoint_path}
#             )
#             return False, ModelCheckpoint(), None

#         if len(ckpt_files) == 0:
#             logger.debug(
#                 f"Get resume info from folder {msg}", result="doesnt exist any file", path={config.ckpt.checkpoint_path}
#             )
#             return False, ModelCheckpoint(), None

#         latest_ckpt = ckpt_files[0].path
#         model_meta = ckpt_files[0]
#         logger.debug(
#             "Get resume info from folder",
#             result="found",
#             path={config.ckpt.checkpoint_path},
#             model_meta=model_meta,
#         )
#         return True, model_meta, latest_ckpt

# def get_sorted_checkpoints(checkpoint_path: str) -> dict[ModelCheckpoint]:
#     fs, root = fsspec.core.url_to_fs(checkpoint_path)

#     ckpt_files = []
#     for f in fs.ls(root, detail=False):
#         if Path(f).name.startswith(".tmp_"):
#             continue

#         if "yaml" in f.lower():  # safer, catches .YAML/.Yaml/.yml too
#             continue

#         meta = parse_dynamic_filename(f)
#         if meta is None:
#             continue

#         # ensure both fields exist and are numeric
#         model_meta = ModelCheckpoint(
#             global_ver=int(meta.get("globalver", 0)), inner_opt=int(meta.get("inneropt", 0)), path=Path(f)
#         )
#         ckpt_files.append(model_meta)

#     # sort descending by globalver, then inneropt
#     return sorted(
#         ckpt_files,
#         key=lambda item: (-item.global_ver, -item.inner_opt),
#     )


def get_model_files(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)  # normalize to Path object

    # Case 1: checkpoint_path IS a .pt file
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        return fsspec.open_files(str(checkpoint_path), mode="rb")

    # Case 2: checkpoint_path is a directory → match model*.pt inside it
    pattern = str(checkpoint_path / "model*.pt")
    files = fsspec.open_files(pattern, mode="rb")

    return files


# def delete_old_checkpoints(checkpoint_path: str, topk: int) -> list[str]:
#     """
#     Deletes old checkpoints, keeping only the top 'k' most recent ones.

#     Args:
#         checkpoint_path (str): The path to the checkpoint directory.
#         topk (int): The number of recent checkpoints to keep.

#     Returns:
#         list[str]: A list of deleted checkpoint filenames.
#     """
#     fs = GenericFileSystem()
#     sorted_ckpt_files = get_sorted_checkpoints(checkpoint_path)

#     ckpt_deleted = []
#     for model_meta in sorted_ckpt_files[topk:]:
#         fs.rm(str(model_meta.path), recursive=True)
#         ckpt_deleted.append(str(model_meta.path))
#     return ckpt_deleted


# def delete_old_checkpoints_by_hotkey(folder_path: Path):
#     """
#     Deletes all non-latest submission files coming from the same hotkey.
#     Keeps only the file with the highest block number per hotkey.

#     Requires: parse_dynamic_filename(filename: str) -> dict
#     """
#     if not folder_path.exists():
#         raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

#     # Step 1: Group files by hotkey
#     submissions_by_hotkey = {}
#     for file_path in folder_path.glob("*.pt"):
#         meta = parse_dynamic_filename(file_path.name)
#         if "hotkey" not in meta or "block" not in meta:
#             print(f"⚠️ Skipping malformed filename: {file_path.name}")
#             continue

#         hotkey = meta["hotkey"]
#         block = meta["block"]

#         # Track the latest submission per hotkey
#         if hotkey not in submissions_by_hotkey:
#             submissions_by_hotkey[hotkey] = []
#         submissions_by_hotkey[hotkey].append((block, file_path))

#     # Step 2: For each hotkey, keep only the highest block file
#     deleted_files = []
#     for _, entries in submissions_by_hotkey.items():
#         # Sort by block number descending (latest first)
#         entries.sort(key=lambda x: x[0], reverse=True)

#         # Keep the first (latest) one, delete the rest
#         for _, file_path in entries[2:]:
#             try:
#                 os.remove(file_path)
#                 deleted_files.append(file_path.name)
#             except Exception as e:
#                 print(f"❌ Failed to delete {file_path.name}: {e}")

#     # Step 3: Log result
#     if deleted_files:
#         logger.info(f"🧹 Deleted {len(deleted_files)} outdated submission(s):", deleted_files)
#         for f in deleted_files:
#             print(f"   - {f}")
#     else:
#         logger.info("✅ No outdated submissions found.")

#     return deleted_files

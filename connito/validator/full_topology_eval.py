"""Score miners on the full expert topology without a second model copy.

The direct route — `moe.partial_moe: false`, so `global_model` itself is full —
was measured off a cliff twice over: the merge needs params + `.grad` + SGD
momentum resident (87.8 GiB at 15.71 B params, against 44.4 GiB of VRAM or a
62 GB host), and per-miner eval `deepcopy`s the base (a second 29.3 GiB).
Freezing the unmerged experts can't rescue it either: `requires_grad` is
per-tensor, and a full layer's 64 experts share one stacked tensor.

So this module keeps `global_model` partial — merge, outer optimizer, chain
hash and peer sync exactly as production — and adds a separate frozen
full-topology base used only for scoring:

  - every routed expert resident, routed at `moe.full_topk` (the base
    checkpoint's native width);
  - `requires_grad=False` throughout: no `.grad`, no momentum, no optimizer;
  - miner shards graft *in place* (the stock `load_state_dict` overlay already
    writes per-expert keys into the stacked rows), with the touched rows backed
    up first and restored after;
  - parked on CPU outside the eval window so the merge phase never shares the
    GPU with it.

This is the reference experiment's own architecture, translated:
`~/experiment/partial_moe.py` keeps trainable `local_experts` apart from a full
frozen expert set that lives on CPU and swaps in for merged-topology eval
(`_cached_experts` / `set_eval_use_merged`). Its paradigm D — "full model,
K assigned experts trainable at original positions" — is exactly
partial-global + frozen-full-eval.

Grafting in place is safe *because this model is disposable*: it is never
hashed, submitted, merged or synced. The failure mode that rules out grafting
into `global_model` — a died restore corrupting what the validator commits —
here costs at worst one round's scores, and `refresh_from` rebuilds the base
from `global_model` at the next round anyway.

Budget on an L40S (44.4 GiB usable), measured pieces: eval window holds the
partial global (9.2 GiB, idle) + this base (29.3 GiB) + activations. Merge
window holds the partial merge peak (27.6 GiB) with this base parked on host
(29.3 GB of 62).

The backup rows are deliberately NOT in that VRAM list. v1 put them there and
the first live round scored nothing: baseline eval fitted at 39.5 GiB, then
every miner OOM'd at 42.9 GiB with 213 MB free. The rows are pooled on the
host now — see `_stage_backup`.
"""

from __future__ import annotations

import re
import threading

import torch
from torch import nn

from connito.shared.app_logging import structlog
from connito.shared.helper import load_state_dict_from_path

logger = structlog.get_logger(__name__)


def _log_memory(message: str, device) -> None:
    """VRAM and host RSS either side of every move.

    The first live run died between `park()` and the merge with no traceback,
    and neither transition logged anything — so the post-mortem had a
    seventeen-second window and no numbers in it. Cheap insurance.
    """
    fields: dict[str, float | str] = {"device": str(device)}
    try:
        import os

        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    fields["host_rss_gb"] = round(int(line.split()[1]) / 1024**2, 2)
                    break
    except OSError:
        pass
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        fields["vram_used_gb"] = round((total_b - free_b) / 1024**3, 2)
        fields["vram_free_gb"] = round(free_b / 1024**3, 2)
    logger.info(message, **fields)

# Miner shards as our serializer writes them: per-expert slices of the fused
# stacked storage, global expert id in the key, no trailing `.weight`.
_SHARD_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts\.)(?P<gid>\d+)\.(?P<name>gate_up_proj|down_proj)$"
)
_LAYER_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.$")


class ShardRejected(ValueError):
    """A shard asked to write somewhere an in-place graft must not.

    Subclasses ValueError so `evaluate_one_miner_sync`'s existing
    `statedict_parse_failed` handler records the failure against the miner
    without new plumbing.

    With `deepcopy` isolation a hostile shard could only poison its own copy.
    An in-place graft restores exactly the rows it backed up, so any write
    outside the validated set — another group's experts, the backbone, the
    router — would survive the restore and contaminate every subsequent
    miner's eval against this base. Hence: validate first, write second.
    """


class FullTopologyEvalBase:
    """A frozen full-expert model that miners are grafted into and scored on."""

    def __init__(self, model: nn.Module, allowed_by_layer: dict[int, set[int]]):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self._allowed_by_layer = allowed_by_layer
        # [(experts_module, param_name, global_id, saved_rows)] for the graft
        # currently applied; empty when the base is clean.
        self._backup: list[tuple[nn.Module, str, int, torch.Tensor]] = []
        # Serialises graft→eval→restore against prepare/park. Needed because a
        # foreground per-miner timeout only detaches the awaiter — the
        # `asyncio.to_thread` thread keeps running (see the long note in
        # `evaluate_one_miner_sync`). With a deepcopy that orphan burned VRAM;
        # with a shared base it would interleave its restore with the next
        # miner's graft and silently corrupt that miner's loss.
        self.lock = threading.Lock()
        # Bumped by prepare_for_round and park. An orphan that was still
        # *queued* on the lock when the round moved on would otherwise graft a
        # stale shard onto the refreshed base — the generation check turns it
        # into a clean per-miner failure instead.
        self._generation = 0
        # Host storage, kept alive for the process lifetime.
        #
        # `.to("cpu")` does not reuse anything: the host tensors were released
        # when the model moved to the GPU, so parking would *allocate* another
        # 29.3 GB — at the exact moment the merge phase begins and wants memory
        # of its own. A validator died there, silently, with no traceback.
        #
        # Holding references to the original CPU tensors instead means the
        # host cost is a flat 29.3 GB from build to shutdown, and `park()`
        # becomes a copy into storage that already exists. Peak host is
        # unchanged from steady-state host; there is no spike to mistime.
        self._shadow_params = {n: p.data for n, p in self.model.named_parameters()}
        self._shadow_buffers = {n: b for n, b in self.model.named_buffers()}
        self._on_gpu = False
        # Reusable HOST storage for graft backups, keyed by the row it holds.
        #
        # v1 cloned the touched rows with `rows[local].clone()`, which clones
        # onto the *source* device — the GPU, during the eval window. Measured
        # cost of that: baseline eval ran at 39.5 GiB of 44.4, and every miner
        # then OOM'd at 42.9 GiB with 213 MB free. The delta is this backup.
        # The rows are write-once and read-once at restore; nothing reads them
        # on the GPU, so nothing justifies their being there.
        #
        # Pooled rather than allocated per miner for the same reason `park()`
        # writes into `_shadow_*`: every miner of a round touches the same
        # expert set (this validator's group), so a pool is allocated once and
        # reused, and host cost is flat instead of a 3.4 GB churn per miner on
        # a box that has no swap.
        self._backup_pool: dict[tuple[str, int, str], torch.Tensor] = {}

    @classmethod
    def build(cls, config, expert_manager) -> "FullTopologyEvalBase":
        """Build the full model on CPU from the pretrained checkpoint.

        Import deferred: mycelia's import chain reaches transformers, and this
        module is imported by run.py before config validation.
        """
        from connito.shared.modeling.mycelia import get_base_model

        model = get_base_model(
            config,
            expert_manager=expert_manager,
            group_ids_trainable=None,
            group_ids_helper=None,
            partial=False,
        )
        # Mirror `build_grad_buff_from_model`'s reading of the assignment
        # pairs: take both halves of each tuple rather than assuming which
        # side is the global id.
        assignment = expert_manager.expert_group_assignment.get(
            config.task.exp.group_id, {}
        )
        allowed = {
            int(layer): {int(a) for a, _ in pairs} | {int(b) for _, b in pairs}
            for layer, pairs in assignment.items()
        }
        base = cls(model, allowed)
        logger.info(
            "Full-topology eval base built",
            layers=len(allowed),
            owned_experts=sum(len(v) for v in allowed.values()),
            device="cpu",
        )
        _log_memory("full-topology base: resident on host", "cpu")
        return base

    # ── round lifecycle ──────────────────────────────────────────────────
    def prepare_for_round(self, global_model: nn.Module, device) -> None:
        """Move to the GPU and adopt the current global state.

        Takes the lock, so it *blocks until any orphaned eval thread from the
        previous round drains* — that is a feature: the refresh must not race
        a restore. Bumping the generation then invalidates orphans that were
        still queued rather than running.

        Order matters: `refresh_from` after `.to(device)` keeps the copies
        GPU-to-GPU views instead of staging through host RAM.
        """
        with self.lock:
            self._generation += 1
            # `.to()` in this direction is safe: it allocates on the GPU and
            # *releases* host storage. The shadow dict keeps the host tensors
            # alive across the move, which is what makes park() free later.
            self.model.to(device)
            self._on_gpu = True
            self.refresh_from(global_model)
            _log_memory("full-topology base: on GPU for eval", device)

    def round_handle(self) -> "_RoundGraftHandle":
        """A graft controller pinned to the current generation.

        Foreground eval holds this for the round; any graft attempted through
        it after the round has moved on (prepare/park bumped the generation)
        fails cleanly instead of touching the base.
        """
        return _RoundGraftHandle(self, self._generation)

    def refresh_from(self, global_model: nn.Module) -> None:
        """Overlay everything the partial global model knows onto this base.

        The custom expert serializer emits per-expert keys under *global* ids,
        and the overlay loader writes them into the stacked rows — so one
        `load_state_dict` carries the backbone and the merged expert rows
        across, and leaves the rows this validator doesn't own at their
        pretrained values (the same view of other groups the partial model
        itself has).
        """
        if self._backup:
            # A crashed eval left a graft applied. The overlay below rewrites
            # every row the graft could legally have touched, so dropping the
            # backup is correct — but say so, because it means an eval died
            # between graft and restore.
            logger.warning(
                "refresh_from: discarding stale graft backup",
                rows=len(self._backup),
            )
            self._backup = []
        with torch.no_grad():
            self.model.load_state_dict(global_model.state_dict(), strict=False)

    def park(self) -> None:
        """Release the GPU for the merge window. Idempotent.

        Copies back into the pre-allocated host tensors rather than calling
        `.to("cpu")`, which would allocate a second 29.3 GB alongside whatever
        the merge is about to want. Nothing is allocated here; the GPU tensors
        are dropped when the last reference goes.

        Blocks on the lock until a running eval drains — pulling storage out
        from under a live forward pass is not an option — and bumps the
        generation so a queued orphan fails instead of grafting onto a parked
        model. Worst case is one per-miner eval's wait; the alternative is the
        merge fighting a resident 29.3 GiB base for a 44.4 GiB card.
        """
        with self.lock:
            self._generation += 1
            if not self._on_gpu:
                return
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    shadow = self._shadow_params[name]
                    shadow.copy_(param.data)
                    param.data = shadow
                for name, buf in self.model.named_buffers():
                    shadow = self._shadow_buffers.get(name)
                    if shadow is None:  # buffer registered after build
                        self._shadow_buffers[name] = buf.detach().to("cpu")
                        continue
                    shadow.copy_(buf)
                    self._set_buffer(name, shadow)
            self._on_gpu = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_memory("full-topology base: parked on host", "cpu")

    def _set_buffer(self, qualified_name: str, tensor: torch.Tensor) -> None:
        module_path, _, attr = qualified_name.rpartition(".")
        module = self.model.get_submodule(module_path) if module_path else self.model
        setattr(module, attr, tensor)

    # ── per-miner graft ──────────────────────────────────────────────────
    def graft_from_path(self, path: str, *, generation: int | None = None) -> nn.Module:
        """Validate, back up, overlay. Returns the model to evaluate.

        Caller must hold `self.lock`. `generation`, when given, must match
        the current one — the stale-orphan guard; checked first, before any
        read of the shard, so a stale caller cannot touch the base at all.
        """
        if generation is not None and generation != self._generation:
            raise ShardRejected(
                f"graft generation {generation} is stale (current "
                f"{self._generation}) — the round moved on while this eval "
                "was queued"
            )
        if self._backup:
            raise RuntimeError(
                "graft_from_path called with a graft still applied — "
                "restore_grafted must run between miners"
            )
        sd = load_state_dict_from_path(path)
        if len(sd) == 0:
            raise ValueError(f"Checkpoint at {path} has empty model_state_dict")

        touched: list[tuple[str, int, str]] = []  # (module_path, gid, param)
        for key in sd:
            match = _SHARD_KEY_RE.match(key)
            if match is None:
                raise ShardRejected(
                    f"shard key {key!r} is not a per-expert projection; "
                    "in-place graft refuses keys it cannot restore"
                )
            prefix = match.group("prefix")
            layer_match = _LAYER_RE.search(prefix)
            if layer_match is None:
                raise ShardRejected(f"shard key {key!r} has no layer id")
            layer, gid = int(layer_match.group(1)), int(match.group("gid"))
            if gid not in self._allowed_by_layer.get(layer, set()):
                raise ShardRejected(
                    f"shard writes expert {gid} on layer {layer}, outside "
                    "this validator's group"
                )
            touched.append((prefix[:-1], gid, match.group("name")))

        backup: list[tuple[nn.Module, str, int, torch.Tensor]] = []
        with torch.no_grad():
            for module_path, gid, name in touched:
                experts = self.model.get_submodule(module_path)
                local = int(experts.global_to_local_map[gid])
                rows = getattr(experts, name).data
                backup.append(
                    (experts, name, local, self._stage_backup(module_path, gid, name, rows[local]))
                )
            self._backup = backup
            try:
                self.model.load_state_dict(sd, strict=False)
            except Exception:
                # A half-applied overlay must not outlive this call: the
                # caller's finally only runs restore once the graft has been
                # handed back, so self-heal here and re-raise.
                self.restore_grafted()
                raise
        # Per-miner memory, because the round that motivated the pool above
        # also grew host RSS 31.5 -> 47.3 GB across four eval loader builds
        # and then died entering the merge. Whatever is growing, it shows up
        # between these two lines or it does not show up in the eval at all.
        _log_memory("full-topology base: grafted", self._device())
        return self.model

    def _stage_backup(
        self, module_path: str, gid: int, name: str, row: torch.Tensor
    ) -> torch.Tensor:
        """Copy one expert's rows into pooled host storage and return it.

        Always lands on CPU regardless of where the base currently is: the
        backup exists to be read once, at restore, and holding it in VRAM was
        what left the eval window with 213 MB free.
        """
        key = (module_path, gid, name)
        staged = self._backup_pool.get(key)
        if staged is None or staged.shape != row.shape or staged.dtype != row.dtype:
            staged = torch.empty(row.shape, dtype=row.dtype, device="cpu")
            self._backup_pool[key] = staged
        staged.copy_(row)
        return staged

    def restore_grafted(self) -> None:
        """Put the backed-up rows back. The base is bit-identical to
        pre-graft afterwards — asserted by test, relied on by every
        subsequent miner's eval.

        The saved rows live on the host; `copy_` handles the transfer back
        onto whichever device the base is on.
        """
        had_backup = bool(self._backup)
        with torch.no_grad():
            for experts, name, local, saved in self._backup:
                getattr(experts, name).data[local].copy_(saved)
        self._backup = []
        if had_backup:
            _log_memory("full-topology base: restored", self._device())

    def _device(self):
        return next(self.model.parameters()).device


class _RoundGraftHandle:
    """Duck-types the controller interface `evaluate_one_miner_sync` uses,
    with the round's generation baked in. Everything else delegates."""

    def __init__(self, base: FullTopologyEvalBase, generation: int):
        self._base = base
        self._generation = generation

    @property
    def lock(self) -> threading.Lock:
        return self._base.lock

    def graft_from_path(self, path: str) -> nn.Module:
        return self._base.graft_from_path(path, generation=self._generation)

    def restore_grafted(self) -> None:
        self._base.restore_grafted()

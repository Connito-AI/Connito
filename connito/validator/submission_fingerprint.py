"""Weight-delta sketches for near-duplicate submission detection.

Every miner in an expert group trains from the same frozen base model, so
two independent submissions differ in the *direction* their weights moved,
not in the weights themselves. A copy-then-perturb submission, by contrast,
has (almost exactly) the same delta as its source plus a small random
nudge — the two delta vectors stay nearly parallel no matter how the
perturbation is scaled, as long as it is small enough not to destroy the
copied model's val_loss.

Comparing full deltas would mean holding every submission in memory at
finalize, so we compare fixed coordinate samples instead: for each tensor
shared between the submission and the base model, a deterministic,
seed-derived subset of coordinates is read from both and their difference
concatenated into a single 1-D sketch (~32k floats by default). Cosine
similarity between two sketches then estimates cosine similarity between
the full deltas with sampling error ~1/sqrt(budget) ≈ 0.006.

The seed comes from the round's combined seed, which is not known to
miners at upload time — so an adversary cannot concentrate their
perturbation on the sampled coordinates to evade detection. Sketches are
identical across validators for the same round because every validator
derives the same combined seed.
"""

from __future__ import annotations

import hashlib

import torch

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# Ignore deltas whose sampled norm is below this — a submission that equals
# the base model has no training signal, scores ~0 delta-loss anyway, and
# its cosine against anything is numerically meaningless.
MIN_SKETCH_NORM = 1e-8


def _coords_for_tensor(name: str, numel: int, k: int, seed: str) -> torch.Tensor:
    """`k` deterministic sample indices into a flattened tensor of size
    `numel`, derived from (seed, tensor name) — stable across validators
    and across the two sketch calls (submission + base) within a round.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    gen = torch.Generator()
    gen.manual_seed(int.from_bytes(digest[:8], "big", signed=False) % (2**63))
    if k >= numel:
        return torch.arange(numel, dtype=torch.long)
    return torch.randperm(numel, generator=gen)[:k]


def sketch_state_dict(
    sd: dict[str, torch.Tensor],
    base_sd: dict[str, torch.Tensor],
    *,
    seed: str,
    budget: int = 32768,
) -> torch.Tensor | None:
    """1-D fp32 CPU sketch of ``sd - base_sd`` over their common
    same-shape tensors. Returns ``None`` when there is nothing comparable
    (no common keys) so callers can skip recording rather than store a
    zero-length tensor.

    The per-tensor sample count is allocated proportionally to numel with
    a floor of 8, so small biases/norms still contribute and one huge
    tensor cannot monopolise the budget entirely.
    """
    common = sorted(
        k for k in sd.keys() & base_sd.keys()
        if sd[k].shape == base_sd[k].shape and sd[k].numel() > 0
    )
    if not common:
        return None

    total = sum(sd[k].numel() for k in common)
    pieces: list[torch.Tensor] = []
    for name in common:
        numel = sd[name].numel()
        k = max(8, int(budget * numel / total))
        coords = _coords_for_tensor(name, numel, k, seed)
        flat_a = sd[name].detach().reshape(-1)
        flat_b = base_sd[name].detach().reshape(-1)
        a = flat_a[coords.to(flat_a.device)].float().cpu()
        b = flat_b[coords.to(flat_b.device)].float().cpu()
        pieces.append(a - b)
    return torch.cat(pieces)


def find_near_duplicate_clusters(
    fingerprints: dict[int, torch.Tensor],
    *,
    cos_threshold: float,
) -> list[set[int]]:
    """Group UIDs whose sketches are pairwise near-parallel.

    Returns disjoint clusters of size >= 2 (union-find over flagged
    pairs). UIDs whose sketch norm is ~0 are excluded — see
    ``MIN_SKETCH_NORM``. Mismatched sketch lengths (different expert
    groups / architectures) never compare equal; they are simply skipped.
    """
    uids = sorted(fingerprints)
    usable: list[int] = []
    normed: dict[int, torch.Tensor] = {}
    for uid in uids:
        fp = fingerprints[uid]
        norm = float(torch.linalg.vector_norm(fp))
        if norm < MIN_SKETCH_NORM:
            continue
        usable.append(uid)
        normed[uid] = fp / norm

    parent: dict[int, int] = {u: u for u in usable}

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    flagged_pairs: list[tuple[int, int, float]] = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if normed[a].numel() != normed[b].numel():
                continue
            cos = float(torch.dot(normed[a], normed[b]))
            if cos >= cos_threshold:
                flagged_pairs.append((a, b, cos))
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    if not flagged_pairs:
        return []

    clusters: dict[int, set[int]] = {}
    for u in usable:
        clusters.setdefault(find(u), set()).add(u)
    result = [c for c in clusters.values() if len(c) >= 2]

    logger.warning(
        "near-duplicate submissions detected",
        clusters=[sorted(c) for c in result],
        pair_cosines={f"{a}-{b}": round(cos, 6) for a, b, cos in flagged_pairs},
        cos_threshold=cos_threshold,
    )
    return result

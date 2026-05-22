"""Bench utilities for the Connito miner.

This package contains offline tooling that helps a miner self-score
candidate checkpoints against the validator's scoring formula before
publishing on-chain. Nothing in this package is imported by the
validator/miner production code paths.

Modules are intentionally added in the same idempotent style as the rest
of the ``connito`` codebase — re-running an existing import never has
side effects.
"""

from __future__ import annotations

__all__: list[str] = []

"""Console-script shim for the owner eval daemon (parity with miner/validator CLIs)."""

from __future__ import annotations

from connito.owner_eval.run import main


if __name__ == "__main__":
    raise SystemExit(main())

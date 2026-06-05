"""Owner eval daemon.

Long-running process the subnet owner runs independently of any miner/validator.
Polls the cycle API; every ``eval_interval_cycles`` cycles it downloads the
latest merged full model (latest validator HF checkpoint), runs the enabled
benchmark suite, and emits the results as Prometheus metrics.

Launch:  ``python -m connito.owner_eval.run --path owner_eval.yaml``
One-shot (bypass the cycle gate, run once, exit — for verification/smoke):
         ``python -m connito.owner_eval.run --path owner_eval.yaml --once``
         (or set ``OWNER_EVAL_FORCE_RUN=1``)

Heavy imports (torch / bittensor / transformers / prometheus) are deferred into
the functions that use them so importing this module — e.g. to unit-test the
cycle-gate predicate — stays cheap.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from connito.shared.app_logging import configure_logging, structlog

logger = structlog.get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone owner eval pipeline daemon")
    parser.add_argument("--path", required=True, help="Path to the OwnerEvalConfig YAML file")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Bypass the cycle gate, run the suite exactly once, then exit (verification/smoke).",
    )
    return parser.parse_args(argv)


def should_run_cycle(cycle_index: int, interval: int, last_ran_cycle: int) -> bool:
    """Cycle-gate predicate, factored out for testing.

    Run when the cycle index lands on the interval boundary and we haven't
    already run for this exact cycle (the daemon polls several times per cycle).
    """
    if cycle_index < 0:
        return False
    return cycle_index % interval == 0 and cycle_index != last_ran_cycle


def _force_run_requested(args: argparse.Namespace) -> bool:
    if getattr(args, "once", False):
        return True
    return str(os.environ.get("OWNER_EVAL_FORCE_RUN", "")).lower() in ("1", "true", "yes")


def _run_once(config: Any, expert_manager: Any, subtensor: Any, wallet: Any,
              tokenizer: Any, device: Any, cycle_index: int) -> None:
    from connito.owner_eval import bootstrap
    from connito.owner_eval.runner import run_eval_suite

    model, checkpoint = bootstrap.load_latest_full_model(
        config=config,
        expert_manager=expert_manager,
        subtensor=subtensor,
        wallet=wallet,
        device=device,
    )
    try:
        revision = ("base" if config.eval_pipeline.model_source == "base"
                    else bootstrap.model_revision_label(checkpoint))
        run_eval_suite(model, tokenizer, device, config, model_revision=revision, cycle_index=cycle_index)
    finally:
        bootstrap.release_model(model)


def main_loop(config: Any, force_run: bool = False) -> None:
    from connito.shared import telemetry
    from connito.shared.chain import setup_chain_worker
    from connito.shared.cycle import get_phase_from_api
    from connito.shared.expert_manager import ExpertManager
    from connito.shared.modeling.mycelia import get_base_tokenizer
    from connito.owner_eval import bootstrap
    from connito.owner_eval.metrics.base import prep_tokenizer

    telemetry.TelemetryManager().start_server(port=config.eval_pipeline.telemetry_port)
    # "base" canary mode needs no chain identity — skip wallet/subtensor setup so
    # the publish->scrape->API path can be tested without an owner wallet.
    if config.eval_pipeline.model_source == "base":
        wallet, subtensor = None, None
        logger.info("model_source=base: skipping chain worker (canary mode)")
    else:
        wallet, subtensor, _lite = setup_chain_worker(config, serve=False)
    expert_manager = ExpertManager(config)
    tokenizer = prep_tokenizer(get_base_tokenizer(config))
    device = bootstrap.resolve_device(config)

    N = config.eval_pipeline.eval_interval_cycles
    poll_interval = config.eval_pipeline.poll_interval_sec
    last_ran_cycle = -1

    logger.info("owner eval daemon started", interval_cycles=N, poll_interval_sec=poll_interval,
                telemetry_port=config.eval_pipeline.telemetry_port, device=str(device),
                force_run=force_run)

    if force_run:
        # One-shot: evaluate immediately against whatever the latest checkpoint is.
        phase = get_phase_from_api(config)
        cycle_index = phase.cycle_index if phase is not None else -1
        _run_once(config, expert_manager, subtensor, wallet, tokenizer, device, cycle_index)
        return

    while True:
        telemetry.set_owner_eval_heartbeat()
        phase = get_phase_from_api(config)
        if phase is None:
            logger.warning("cycle API returned no phase; will retry")
        elif should_run_cycle(phase.cycle_index, N, last_ran_cycle):
            logger.info("cycle gate open; running eval suite", cycle_index=phase.cycle_index)
            try:
                _run_once(config, expert_manager, subtensor, wallet, tokenizer, device, phase.cycle_index)
            except Exception as exc:  # noqa: BLE001 — keep the daemon alive across run failures
                logger.warning("owner eval run failed", cycle_index=phase.cycle_index, error=str(exc),
                               exc_info=True)
                telemetry.inc_error("owner_eval", "run_failed")
            last_ran_cycle = phase.cycle_index
        time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    from connito.owner_eval import bootstrap

    configure_logging()
    args = parse_args(argv)
    config = bootstrap.load_config(args.path)
    main_loop(config, force_run=_force_run_requested(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

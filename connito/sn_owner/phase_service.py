import json
from contextlib import asynccontextmanager
from pathlib import Path

import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException

from connito.shared.app_logging import configure_logging
from connito.shared.config import CycleCfg, OwnerConfig, parse_args
from connito.sn_owner.cycle import PhaseManager, PhaseResponse
from connito.sn_owner.init_peer_store import get_init_peer_ids
from connito.validator.inter_validator_connection import structlog


phase_manager: PhaseManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the async subtensor and the PhaseManager bound to it.

    Kept inside lifespan (rather than __main__) because `AsyncSubtensor`
    needs an event loop for `initialize()`. The sync subtensor that used to
    back PhaseManager is gone — concurrent `recv()` from the threadpool was
    raising `websockets.ConcurrencyError`.
    """
    global phase_manager
    async_subtensor = bittensor.AsyncSubtensor(network=config.chain.lite_network)
    await async_subtensor.initialize()
    phase_manager = PhaseManager(config.cycle, async_subtensor)
    try:
        yield
    finally:
        close = getattr(async_subtensor, "close", None)
        if close is not None:
            try:
                await close()
            except Exception as e:
                logger.warning("AsyncSubtensor close failed", error=str(e))


app = FastAPI(title="Phase Service", lifespan=lifespan)


@app.get("/get_phase", response_model=PhaseResponse)
async def read_phase():
    try:
        return await phase_manager.get_current_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/previous_phase_blocks", response_model=dict[str, tuple[int, int]])
async def prev_phase():
    try:
        return await phase_manager.current_previous_phase_block_ranges()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/blocks_until_next_phase", response_model=dict[str, tuple[int, int, int]])
async def next_phase():
    try:
        return await phase_manager.current_blocks_until_next_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_init_peer_id", response_model=list[str])
async def get_init_peer_id():
    try:
        return get_init_peer_ids(init_peer_id_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_validator_whitelist", response_model=list[str])
async def get_validator_whitelist():
    """Returns the list of hotkeys that are force-permitted as validators."""
    try:
        with open(validator_whitelist_path) as f:
            hotkeys = json.load(f)
        return hotkeys
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_cycle_config", response_model=CycleCfg)
async def get_cycle_config():
    """Returns the owner's CycleCfg. Clients can feed this dict into PhaseManager.from_dict to track phases locally."""
    try:
        with open(cycle_config_path) as f:
            return CycleCfg(**json.load(f))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="cycle config not yet written")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Phase service is running",
        "cycle_length": phase_manager.cycle_length,
        "phases": [{"index": i, "name": p["name"], "length": p["length"]} for i, p in enumerate(phase_manager.phases)],
        "usage": "GET /phase?block_height=123",
    }


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    logger = structlog.get_logger(__name__)

    global config
    global init_peer_id_path
    global validator_whitelist_path
    global cycle_config_path

    if args.path:
        config = OwnerConfig.from_path(args.path, auto_update_config=args.auto_update_config)
    else:
        config = OwnerConfig()

    config.write()

    init_peer_id_path = Path(config.run.root_path) / "init_peer_ids.json"
    validator_whitelist_path = Path(config.run.root_path) / "connito" / "sn_owner" / "validator_whitelist.json"
    cycle_config_path = Path(config.run.root_path) / "cycle_config.json"

    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)

    cycle_config_path.write_text(config.cycle.model_dump_json(indent=2))
    logger.info("wrote cycle config", path=str(cycle_config_path))

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        limit_concurrency=265,
        timeout_keep_alive=30,
        backlog=4096,
        access_log=True,
    )

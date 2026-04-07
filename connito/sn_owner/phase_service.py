import bittensor
from connito.validator.inter_validator_connection import structlog, AllowedHotkeyService
import uvicorn
from fastapi import FastAPI, HTTPException

from connito.shared.config import OwnerConfig, parse_args
from connito.sn_owner.cycle import PhaseManager, PhaseResponse
from connito.shared.app_logging import configure_logging
import multiprocessing as mp
from pathlib import Path

from connito.sn_owner.init_peer_store import add_init_peer_id, get_init_peer_ids
from connito.sn_owner.dht_init import init_dht_and_peer_id
import json

app = FastAPI(title="Phase Service")

@app.get("/get_phase", response_model=PhaseResponse)
async def read_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.get_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/previous_phase_blocks", response_model=dict[str, tuple[int, int]])
async def prev_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.previous_phase_block_ranges()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/blocks_until_next_phase", response_model=dict[str, tuple[int, int, int]])
async def next_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.blocks_until_next_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_init_peer_id", response_model=list[str])
async def get_init_peer_id():
    """
    Returns which phase we're in for the given block height.
    """
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
    global phase_manager
    global init_peer_id_path
    global validator_whitelist_path

    if args.path:
        config = OwnerConfig.from_path(args.path, auto_update_config=args.auto_update_config)
    else:
        config = OwnerConfig()

    config.write()

    init_peer_id_path = Path(config.run.root_path) / "init_peer_ids.json"
    validator_whitelist_path = Path(config.run.root_path) / "connito" / "sn_owner" / "validator_whitelist.json"

    subtensor = bittensor.Subtensor(network=config.chain.network)
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)

    phase_manager = PhaseManager(config, subtensor)
    uvicorn.run(app, host="127.0.0.1", port=8080)

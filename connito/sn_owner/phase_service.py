"""Local development harness for the subnet-owner phase service.

NOT the deployed service. Production cycle-api is deployed from the private
Connito-AI/cycle-api repository; this module exists so the phase endpoints can
be run offline — in tests, and when driving a validator or miner through a full
cycle without touching the real owner API.

Keep it dependency-light and free of validator/miner imports: it backs onto
`connito.shared.cycle.PhaseManager`, the same block arithmetic the clients use.
"""

import json
from pathlib import Path

import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException

from connito.shared.app_logging import configure_logging, structlog
from connito.shared.config import OwnerConfig, parse_args
from connito.shared.cycle import PhaseManager, PhaseResponse
from connito.sn_owner.init_peer_store import get_init_peer_ids

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

    phase_manager = PhaseManager(config, subtensor)
    uvicorn.run(app, host="127.0.0.1", port=8080)

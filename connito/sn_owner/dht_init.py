from __future__ import annotations

import hivemind
import bittensor
from pathlib import Path

from connito.shared.helper import public_multiaddrs
from connito.validator.inter_validator_connection import HotkeyAuthorizer
from connito.shared.config import OwnerConfig, parse_args
from connito.shared.app_logging import configure_logging, structlog
from connito.sn_owner.init_peer_store import add_init_peer_id, remove_init_peer_id, get_init_peer_ids
import time
import signal

configure_logging()
logger = structlog.get_logger(__name__)

def init_dht_and_peer_id(
    config: OwnerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    dht_port: int = 7002,
    public_ip: str = "",
    peer_store_path: Path | None = None,
) -> tuple[hivemind.DHT, str, bool]:
    logger.info(
        "init_dht_and_peer_id called",
        dht_port=dht_port, public_ip=public_ip, peer_store_path=str(peer_store_path),
    )

    if peer_store_path is None:
        peer_store_path = Path(config.run.root_path) / "init_peer_ids.json"
        logger.debug("using default peer_store_path", peer_store_path=str(peer_store_path))

    existing_peers = get_init_peer_ids(peer_store_path)
    logger.info("loaded existing peers", count=len(existing_peers) if existing_peers else 0, peers=existing_peers)

    authorizer = HotkeyAuthorizer(
        my_hotkey=wallet.hotkey,
        max_time_skew_s=30.0,
        subtensor=subtensor,
        config=config,
    )
    logger.debug("authorizer created", hotkey=str(wallet.hotkey))

    if existing_peers is None or len(existing_peers) == 0:
        logger.info("no existing peers found, creating new DHT", dht_port=dht_port)
        dht = hivemind.DHT(
            host_maddrs=[f"/ip4/0.0.0.0/tcp/{int(dht_port)}", f"/ip4/0.0.0.0/udp/{int(dht_port)}/quic"],
            # host_maddrs=["/ip4/127.0.0.1/tcp/7002", "/ip4/127.0.0.1/udp/7002/quic"],
            start=True,
            client_mode = False,
            authorizer=authorizer,
            bootstrap_timeout=120,
            wait_timeout=30,
        )
        logger.info("new DHT created successfully", dht_peer_id=str(dht.peer_id), dht_port=dht_port)
    else:
        logger.info("found existing_peers, joining DHT", existing_peers=existing_peers)

        last_err: Exception | None = None
        for attempt in range(1, 15):
            logger.info("DHT bootstrap attempt", attempt=attempt, max_attempts=14)
            try:
                dht = hivemind.DHT(
                    start=True,
                    client_mode = False,
                    initial_peers=existing_peers,
                    authorizer=authorizer,
                    bootstrap_timeout=120,
                    wait_timeout=30,
                    host_maddrs=[f"/ip4/0.0.0.0/tcp/{int(dht_port)}", f"/ip4/0.0.0.0/udp/{int(dht_port)}/quic"],
                )
                logger.info("DHT bootstrap succeeded", attempt=attempt)
                break
            except RuntimeError as e:
                last_err = e
                logger.exception("DHT bootstrap failed (RuntimeError)", attempt=attempt, error=str(e))
            except Exception as e:
                last_err = e
                logger.error(
                    "DHT bootstrap failed (unexpected)",
                    attempt=attempt, error=str(e), error_type=type(e).__name__,
                )

            if attempt < 15:
                sleep_time = min(5 * attempt, 10)
                logger.info("retrying DHT bootstrap", attempt=attempt, sleep_seconds=sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error("DHT bootstrap exhausted all retries", total_attempts=attempt)
                raise last_err if last_err is not None else RuntimeError("DHT bootstrap failed after retries")

    peer_id = dht.peer_id.to_base58()
    logger.info("DHT peer_id resolved", peer_id=peer_id)

    init_peer_id = f"/ip4/{public_ip}/tcp/{int(dht_port)}/p2p/{peer_id}"
    logger.info("init_peer_id constructed", init_peer_id=init_peer_id)
    return dht, init_peer_id, True


# sample:
# python connito/sn_owner/dht_init.py --path checkpoints/owner/owner/hk1/run/config.yaml \
#   --dht_port 7002 --public_ip 149.137.225.62
if __name__ == "__main__":
    args = parse_args()
    logger.info("dht_init starting", path=args.path, dht_port=args.dht_port)

    if args.path:
        config = OwnerConfig.from_path(args.path, auto_update_config=args.auto_update_config)
        logger.info("loaded config from path", path=args.path)
    else:
        config = OwnerConfig()
        logger.info("using default config")

    config.write()

    logger.info("connecting to subtensor", network=config.chain.network)
    subtensor = bittensor.Subtensor(network=config.chain.network)
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    logger.info("wallet loaded", coldkey=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)

    init_peer_id_path = Path(config.run.root_path) / "init_peer_ids.json"
    _dht, init_peer_id, started_new = init_dht_and_peer_id(
        config,
        wallet,
        subtensor,
        dht_port=args.dht_port,
        public_ip=args.dht_public_ip,
        peer_store_path=init_peer_id_path,
    )

    logger.info("dht can be discovered at", init_peer_id = init_peer_id)

    if started_new:
        add_init_peer_id(init_peer_id_path, init_peer_id)
        logger.info("registered init peer id in store", path=str(init_peer_id_path))

    _shutdown = False

    def _cleanup_and_exit(*_args):
        global _shutdown
        logger.info("shutdown signal received, cleaning up")
        _shutdown = True
        if started_new:
            remove_init_peer_id(init_peer_id_path, init_peer_id)
            logger.info("removed init peer id from store")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup_and_exit)
    signal.signal(signal.SIGTERM, _cleanup_and_exit)
    logger.info("DHT running, entering keepalive loop")

    # Keep process alive so DHT stays running; restart on failure.
    while not _shutdown:
        try:
            while not _shutdown:
                time.sleep(60)
        except SystemExit:
            raise
        except Exception as e:
            logger.warning("DHT keepalive failed, restarting", error=str(e))
            # Clean up old peer id
            if started_new:
                remove_init_peer_id(init_peer_id_path, init_peer_id)
            try:
                _dht.shutdown()
            except Exception:
                pass
            time.sleep(5)
            try:
                _dht, init_peer_id, started_new = init_dht_and_peer_id(
                    config,
                    wallet,
                    subtensor,
                    dht_port=args.dht_port,
                    public_ip=args.public_ip,
                    peer_store_path=init_peer_id_path,
                )
                logger.info("DHT restarted successfully", init_peer_id=init_peer_id)
                if started_new:
                    add_init_peer_id(init_peer_id_path, init_peer_id)
            except Exception as restart_err:
                logger.error("DHT restart failed, retrying in 30s", error=str(restart_err))
                time.sleep(30)

    if started_new:
        remove_init_peer_id(init_peer_id_path, init_peer_id)

# llm_weightnet/shared/logging.py
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging() -> None:
    """
    Pretty console in dev, JSON in prod. Force with pretty=True/False.
    Env overrides:
      LOG_FORMAT=pretty|json
      LOG_LEVEL=DEBUG|INFO|...
      LOG_UTC=1  (timestamp in UTC)
    """
    level = "INFO"
    fmt = "pretty"
    use_utc = True

    # # 1) stdlib baseline so third-party libs (uvicorn, requests) show up
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")

    # Silence noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("datasets.builder").setLevel(logging.WARNING)

    # 2) structlog processors (common)
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=use_utc)
    common = [
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        processors = common + [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        # Pretty, aligned columns. pad_event aligns the event text; key=value follow neatly.
        processors = common + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=28,  # adjust to your taste
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Module-level logger you can import directly
structlog.configure_once(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ANSI escape for dim dark-gray text — bypasses structlog so the style isn't overridden
_DIM = "\033[90m"
_RESET = "\033[0m"


def log_phase(msg: str, **kwargs) -> None:
    """Log a message that always prints in dim (darker) colour."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    line = f"{ts} [info     ] {msg}"
    if extra:
        line += f" {extra}"
    print(f"{_DIM}{line}{_RESET}", flush=True)

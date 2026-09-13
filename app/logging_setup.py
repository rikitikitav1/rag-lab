import logging
import sys

import structlog
from redaction import redact


# an error text is the one place a key can reach a log line, quoted back by the server that got it
def _redacted(_logger, _method, event: dict) -> dict:
    return {k: redact(v) if isinstance(v, str) else v for k, v in event.items()}


def configure(level: str = "INFO"):
    log_level = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redacted,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


get_logger = structlog.get_logger

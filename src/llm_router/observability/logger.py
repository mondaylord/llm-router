"""Thin structlog wrapper.

JSON output in production, key-value in dev. The router emits one log
event per decision; downstream eval pipelines consume those events.

Importing this module is cheap; configuration happens via
`configure_logging()`. If never called, structlog defaults apply
(human-readable to stderr).
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)

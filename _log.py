"""Structured logging via structlog.

Two modes:
- pretty: human-readable, colored if stderr is a TTY (default for CLI)
- json:   one line per event, machine-parseable (default for Docker / production)

Choose via `configure_logging("pretty"|"json"|"silent")`, or by setting the
DEEP_RESEARCH_LOG env var. CLI users get pretty by default; the Docker image
sets DEEP_RESEARCH_LOG=json so logs are pipe-friendly.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def configure_logging(mode: str | None = None) -> None:
    if mode is None:
        mode = os.environ.get("DEEP_RESEARCH_LOG", "pretty").lower()
    if mode not in ("pretty", "json", "silent"):
        mode = "pretty"

    if mode == "silent":
        logging.getLogger().setLevel(logging.CRITICAL)
        structlog.configure(
            processors=[],
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
        return

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if mode == "json":
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(),
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


# Module-level logger handle. Callers do `from _log import log` then `log.info(...)`.
log = structlog.get_logger("deep_research")

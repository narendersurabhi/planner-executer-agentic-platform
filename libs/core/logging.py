from __future__ import annotations

import logging
from typing import Any, Dict

import structlog


def configure_logging(service_name: str) -> None:
    logging.basicConfig(level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logger = structlog.get_logger(service=service_name)
    logger.info("logging_configured")


def log_event(logger: structlog.BoundLogger, event_type: str, payload: Dict[str, Any]) -> None:
    logger.info(event_type, **payload)


def get_logger(service_name: str) -> structlog.BoundLogger:
    return structlog.get_logger(service=service_name)

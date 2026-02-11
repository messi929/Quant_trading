"""Logging utilities with file and console output."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
) -> None:
    """Configure loguru logger with console and file handlers."""
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if console:
        logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path / "quant_{time:YYYY-MM-DD}.log",
            format=fmt,
            level=level,
            rotation="1 day",
            retention="30 days",
            compression="zip",
        )

        logger.add(
            log_path / "error_{time:YYYY-MM-DD}.log",
            format=fmt,
            level="ERROR",
            rotation="1 day",
            retention="90 days",
        )

    logger.info("Logger initialized")


def get_logger(name: str = __name__):
    """Get a contextualized logger."""
    return logger.bind(name=name)

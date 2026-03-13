"""
Logging configuration using Loguru.

This module sets up logging for the application with both console and file output.
"""

import sys
from pathlib import Path

from loguru import logger

from src.core.config import settings


def setup_logging() -> None:
    """Configure logging with Loguru."""
    # Remove default handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Console handler with colored output
    logger.add(
        sys.stdout,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # File handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="500 MB",
        retention="7 days",
    )

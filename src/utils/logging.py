"""Standardized logging utility for CyberTrace."""

import logging
import sys
import os

def get_logger(name: str = "CyberTrace") -> logging.Logger:
    """Return a configured logger with clean timestamped formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

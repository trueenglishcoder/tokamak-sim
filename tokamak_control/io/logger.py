from __future__ import annotations

import logging
from typing import Optional

_ROOT_LOGGER_NAME = "tokamak_control"
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, *, name: Optional[str] = None) -> logging.Logger:
    """Configure the project logger tree and return the requested logger."""
    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    root_logger.propagate = False

    if name is None or name == _ROOT_LOGGER_NAME:
        return root_logger
    return get_logger(name)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the root project logger or a named child logger."""
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        logger_name = name
    else:
        logger_name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)

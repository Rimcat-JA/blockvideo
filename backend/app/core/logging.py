"""Structured logging helpers — never log secrets or full scripts."""
from __future__ import annotations

import sys

from loguru import logger

from app.core.security import redact


class SafeLogger:
    """Wrap loguru so any %s interpolation is automatically redacted."""

    def __init__(self, bound: object | None = None) -> None:
        """Create an optionally bound logger facade."""
        self._logger = logger if bound is None else logger.bind(bound=bound)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        """Write an informational message after redacting its format string."""
        self._logger.opt(depth=1).info(self._safe(message), *args, **kwargs)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        """Write a warning message through the redacting logger wrapper."""
        self._logger.opt(depth=1).warning(self._safe(message), *args, **kwargs)

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        """Write an error message without exposing key-shaped text."""
        self._logger.opt(depth=1).error(self._safe(message), *args, **kwargs)

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        """Write a debug message through the safe logger."""
        self._logger.opt(depth=1).debug(self._safe(message), *args, **kwargs)

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        """Write an exception message with safe logger traceback behavior."""
        self._logger.opt(depth=1).exception(self._safe(message), *args, **kwargs)

    @staticmethod
    def _safe(message: object) -> object:
        """Redact string messages while leaving structured arguments unchanged."""
        if isinstance(message, str):
            return redact(message)
        return message


def configure_logging(level: str = "INFO") -> None:
    """Install loguru sinks; safe to call multiple times."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )


log = SafeLogger()

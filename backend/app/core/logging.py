"""Redacting Loguru facade used by the backend.

Imports:
    ``sys`` provides the stderr sink installed by ``configure_logging``.
    ``loguru.logger`` supplies structured logging and ``{name}`` formatting.
    ``redact`` removes token-shaped text from format strings before emission.

Only the message/format-string argument is redacted.  Structured keyword or
positional values are passed to Loguru unchanged, so callers must avoid
passing raw secrets as logging arguments as well.
"""
from __future__ import annotations

import sys

from loguru import logger

from app.core.security import redact


class SafeLogger:
    """Wrap Loguru and redact token-shaped text in message strings.

    Loguru uses brace-style placeholders such as ``{project_id}``; this class
    does not implement printf-style interpolation or recursively redact
    structured arguments.

    Attributes:
        _logger: The unbound or context-bound Loguru logger used for output.

    """

    def __init__(self, bound: object | None = None) -> None:
        """Create an optionally bound logger facade.

        Args:
            bound: Optional value attached to the Loguru context as the
                ``bound`` field.  ``None`` uses the module logger directly.

        """
        self._logger = logger if bound is None else logger.bind(bound=bound)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        """Emit an INFO record after redacting the message template.

        Args:
            message: Loguru brace-format message template.
            args: Positional values consumed by the template.
            kwargs: Named values consumed by the template.

        """
        self._logger.opt(depth=1).info(self._safe(message), *args, **kwargs)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        """Emit a WARNING record through the same redaction path as ``info``."""
        self._logger.opt(depth=1).warning(self._safe(message), *args, **kwargs)

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        """Emit an ERROR record after redacting key-shaped text in the template."""
        self._logger.opt(depth=1).error(self._safe(message), *args, **kwargs)

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        """Emit a DEBUG record through the safe logger facade."""
        self._logger.opt(depth=1).debug(self._safe(message), *args, **kwargs)

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        """Emit an ERROR record with Loguru's exception/traceback context."""
        self._logger.opt(depth=1).exception(self._safe(message), *args, **kwargs)

    @staticmethod
    def _safe(message: object) -> object:
        """Redact a string template and leave non-string values untouched.

        This small boundary keeps redaction from changing Loguru's handling of
        exception objects or other structured values.
        """
        if isinstance(message, str):
            return redact(message)
        return message


def configure_logging(level: str = "INFO") -> None:
    """Replace Loguru sinks with the application's stderr configuration.

    Args:
        level: Minimum Loguru level accepted by the sink, for example
            ``"INFO"`` or ``"DEBUG"``.

    Side Effects:
        Removes all existing Loguru sinks and installs one stderr sink with
        timestamps, source location, and disabled backtraces/diagnostics.
        Calling this function again replaces the previous sink rather than
        duplicating log lines.

    """
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


# Shared unbound facade imported by services, workers, and API routes.
log = SafeLogger()

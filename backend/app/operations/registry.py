"""Explicit operation handler registry; catalog strings are never evaluated."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.models.project import Project
from app.operations.contracts import OperationResult

OperationHandler = Callable[[Session, Project, dict[str, Any]], OperationResult]


class RegistryError(ValueError):
    """Raised when handler registration is duplicate or incomplete."""


class HandlerRegistry:
    """Own the only mapping from definition handler keys to callables."""

    def __init__(self) -> None:
        self._handlers: dict[str, OperationHandler] = {}

    def register(self, key: str, handler: OperationHandler) -> None:
        """Register one unique callable under a stable key."""
        if key in self._handlers:
            raise RegistryError(f"duplicate handler key: {key}")
        self._handlers[key] = handler

    def require(self, key: str) -> OperationHandler:
        """Return a registered callable or fail without dynamic lookup."""
        try:
            return self._handlers[key]
        except KeyError as exc:
            raise RegistryError(f"handler is not registered: {key}") from exc

    def keys(self) -> frozenset[str]:
        """Return the immutable set of registered keys."""
        return frozenset(self._handlers)

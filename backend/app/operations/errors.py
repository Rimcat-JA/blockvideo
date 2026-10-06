"""Domain rejections independent of HTTP and persistence orchestration."""
from __future__ import annotations

from app.operations.contracts import Readiness, ReadinessResult


class OperationError(ValueError):
    """Bounded operation rejection with a machine-readable reason."""

    def __init__(
        self, reason_code: str, message: str, *,
        readiness: Readiness | None = None, result: ReadinessResult | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.readiness = readiness
        self.result = result

"""Provider-neutral error contract shared outside provider implementations."""
from __future__ import annotations


class ProviderError(RuntimeError):
    """Represent a sanitized upstream or media-processing failure."""

    def __init__(
        self,
        message: str,
        *,
        safe: bool = True,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.safe = safe
        self.original = original

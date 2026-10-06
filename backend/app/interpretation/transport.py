"""Replaceable inference interface, independent of operation execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["system", "user"]
    content: str


class StructuredAdapter(Protocol):
    """One bounded inference; implementations report safe InterpretationErrors."""

    async def complete(
        self, messages: tuple[ModelMessage, ...], schema: dict[str, Any],
    ) -> str: ...

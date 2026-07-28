"""Secrets handling — keep API keys out of logs, exceptions, and persistence."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Heuristic to detect strings that look like API keys.
_KEY_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|"
    r"sk-or-[A-Za-z0-9_\-]{8,}|"
    r"sk-ant-[A-Za-z0-9_\-]{8,}|"
    r"AIza[0-9A-Za-z_\-]{20,}|"
    r"gho_[A-Za-z0-9]{20,}|"
    r"[A-Za-z0-9_\-]{40,})"
)


def mask(value: str | None) -> str:
    """Mask a secret value for safe display.

    Returns '***' for empty values, otherwise shows the first 4 and last 2
    characters with the middle replaced by asterisks.
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-2:]} (len={len(value)})"


def is_likely_key(value: str) -> bool:
    if not value:
        return False
    return bool(_KEY_PATTERN.search(value))


def redact(text: str) -> str:
    """Replace anything that looks like an API key in free-form text with a mask."""
    if not text:
        return text

    def _sub(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) <= 8:
            return "***"
        return f"{token[:4]}***{token[-2:]}"

    return _KEY_PATTERN.sub(_sub, text)


@dataclass
class SecretBundle:
    """Per-project temporary API key bundle kept in memory only."""

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    image_api_key: str | None = None
    image_base_url: str | None = None
    image_model: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "llm": {
                "has_key": bool(self.llm_api_key),
                "base_url": self.llm_base_url,
                "model": self.llm_model,
                "key_preview": mask(self.llm_api_key),
            },
            "image": {
                "has_key": bool(self.image_api_key),
                "base_url": self.image_base_url,
                "model": self.image_model,
                "key_preview": mask(self.image_api_key),
            },
        }


class SecretStore:
    """In-memory store keyed by project_id. Never persists to disk."""

    def __init__(self) -> None:
        self._store: dict[int, SecretBundle] = {}

    def set(self, project_id: int, bundle: SecretBundle) -> None:
        self._store[project_id] = bundle

    def get(self, project_id: int) -> SecretBundle | None:
        return self._store.get(project_id)

    def drop(self, project_id: int) -> None:
        self._store.pop(project_id, None)

    def summary(self, project_id: int) -> dict[str, Any]:
        bundle = self.get(project_id)
        if bundle is None:
            return {"llm": {}, "image": {}}
        return bundle.to_summary()


# Module-level singleton so handlers share one store.
secret_store = SecretStore()
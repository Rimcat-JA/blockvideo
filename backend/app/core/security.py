"""Secret detection, redaction, and process-local BYOK storage.

Imports:
    ``re`` supplies the heuristic token matcher.
    ``dataclass`` defines the small provider-configuration container.
    ``Any`` describes JSON-like diagnostic summaries.

The module deliberately does not encrypt or persist API keys.  Callers keep
raw values only in ``SecretStore`` for the lifetime of this Python process;
diagnostic methods expose booleans and masked previews instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Heuristic matcher for common provider tokens and long token-shaped strings.
# It is intentionally conservative and is not a credential validator.
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

    Args:
        value: Secret text, or ``None`` when no key was supplied.

    Returns:
        An empty string for ``None`` or an empty string; ``"***"`` for short
        values; otherwise a preview containing the first four and last two
        characters plus the original length.  The full secret is never
        returned.

    """
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-2:]} (len={len(value)})"


def is_likely_key(value: str) -> bool:
    """Return whether free-form text contains a token-shaped substring.

    Args:
        value: Text to inspect.  Empty text is treated as non-secret.

    Returns:
        ``True`` when ``_KEY_PATTERN`` finds a likely provider key.  This is a
        format heuristic, not proof that the value is valid or active.

    """
    if not value:
        return False
    return bool(_KEY_PATTERN.search(value))


def redact(text: str) -> str:
    """Replace likely API keys in free-form text with non-secret previews.

    Args:
        text: Log or exception text that may contain a provider token.

    Returns:
        The original text when no token-shaped value is found, otherwise text
        with every matching token replaced by a short preview.  Non-string
        objects are not accepted because callers should redact before logging.

    """
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
    """Per-project provider configuration kept in memory only.

    Attributes:
        llm_api_key: API key used for the project's language-model provider.
        llm_base_url: OpenAI-compatible chat endpoint base URL.
        llm_model: Main language-model identifier.
        image_api_key: API key used for image generation, if configured.
        image_base_url: Image API base URL.
        image_model: Image-generation model identifier.

    The values are raw secrets or provider settings.  Use ``to_summary`` when
    exposing this bundle to diagnostics; never serialize the dataclass itself.

    """

    # Main LLM provider settings; the key is never written to the database.
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    # Image provider settings; the image key follows the same lifetime rules.
    image_api_key: str | None = None
    image_base_url: str | None = None
    image_model: str | None = None

    def to_summary(self) -> dict[str, Any]:
        """Return provider metadata with masked key previews.

        Returns:
            A JSON-compatible mapping containing ``has_key``, endpoint, model,
            and masked-preview values for the LLM and image providers.

        Side Effects:
            None.  The bundle remains unchanged and no secret is persisted.

        """
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
    """Process-local mapping from project IDs to temporary secret bundles.

    The store is intentionally a plain dictionary: it is used by request
    handlers and worker tasks in one process, is not shared between processes,
    and is cleared only when an entry is dropped or the process exits.
    """

    def __init__(self) -> None:
        """Initialize an empty process-local project-to-secrets mapping."""
        self._store: dict[int, SecretBundle] = {}

    def set(self, project_id: int, bundle: SecretBundle) -> None:
        """Associate or replace a project's temporary provider bundle.

        Args:
            project_id: Database identifier used as the lookup key.
            bundle: Raw provider settings retained only in this process.

        Side Effects:
            Replaces any existing bundle for ``project_id`` in memory.

        """
        self._store[project_id] = bundle

    def get(self, project_id: int) -> SecretBundle | None:
        """Return a project's bundle without modifying the store.

        Args:
            project_id: Database identifier to look up.

        Returns:
            The stored ``SecretBundle`` or ``None`` when no in-memory entry
            exists, including after process restart.

        """
        return self._store.get(project_id)

    def drop(self, project_id: int) -> None:
        """Remove a project's temporary secrets if they exist.

        Args:
            project_id: Database identifier whose entry should be forgotten.

        Side Effects:
            Deletes the dictionary entry; missing IDs are intentionally benign.

        """
        self._store.pop(project_id, None)

    def summary(self, project_id: int) -> dict[str, Any]:
        """Return a safe diagnostic summary for one project.

        Args:
            project_id: Database identifier to summarize.

        Returns:
            ``SecretBundle.to_summary()`` when present, or empty LLM/image
            mappings when the project has no in-memory credentials.

        """
        bundle = self.get(project_id)
        if bundle is None:
            return {"llm": {}, "image": {}}
        return bundle.to_summary()


# Shared process-local singleton used by project routes and provider factories.
secret_store = SecretStore()

"""Pair provider credentials with the endpoint they belong to.

Imports:
    ``dataclass`` defines the resolved credential record.
    ``Literal`` names where a credential came from.
    ``Settings`` supplies server-wide fallbacks from the environment.
    ``SecretBundle`` holds a project's in-memory BYOK values.

A server key from ``.env`` is only ever sent to the server's own endpoint.
When a project names a different base URL, only the project's own key may be
used there; otherwise a blank BYOK key would forward the server key (and the
script) to whatever host the project supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.core.security import SecretBundle

# Default image endpoint when neither the project nor the server names one.
DEFAULT_IMAGE_BASE_URL = "https://api.openai.com/v1"
# Default image model when neither the project nor the server names one.
DEFAULT_IMAGE_MODEL = "gpt-image-1"

CredentialOrigin = Literal["project", "global", "missing"]


@dataclass(frozen=True)
class ResolvedCredential:
    """One provider's key, endpoint and model resolved as a consistent pair.

    Attributes:
        api_key: Key to send, or ``None`` when no usable key exists.
        base_url: Endpoint the key will be sent to.
        model: Model identifier for that endpoint.
        origin: ``project`` for a BYOK key, ``global`` for the server key.

    """

    api_key: str | None
    base_url: str | None
    model: str | None
    origin: CredentialOrigin


def _same_endpoint(left: str | None, right: str | None) -> bool:
    """Compare base URLs ignoring a trailing slash and letter case."""
    if not left or not right:
        return False
    return left.rstrip("/").lower() == right.rstrip("/").lower()


def _resolve(
    project_key: str | None,
    project_url: str | None,
    project_model: str | None,
    server_key: str | None,
    server_url: str | None,
    server_model: str | None,
) -> ResolvedCredential:
    """Choose a key and endpoint that come from the same owner."""
    base_url = project_url or server_url
    model = project_model or server_model
    if project_key:
        return ResolvedCredential(project_key, base_url, model, "project")
    if project_url and not _same_endpoint(project_url, server_url):
        # A foreign endpoint never receives the server's key.
        return ResolvedCredential(None, base_url, model, "missing")
    if server_key:
        return ResolvedCredential(server_key, base_url, model, "global")
    return ResolvedCredential(None, base_url, model, "missing")


def resolve_llm_credential(secrets: SecretBundle | None, settings: Settings) -> ResolvedCredential:
    """Resolve the chat model key, endpoint and model for one project.

    Args:
        secrets: The project's BYOK bundle, if any.
        settings: Server settings providing ``LLM_*`` fallbacks.

    Returns:
        The paired credential; ``api_key`` is ``None`` when none may be used.

    """
    return _resolve(
        secrets.llm_api_key if secrets else None,
        secrets.llm_base_url if secrets else None,
        secrets.llm_model if secrets else None,
        settings.llm_api_key,
        settings.llm_base_url,
        settings.llm_model,
    )


def resolve_image_credential(secrets: SecretBundle | None, settings: Settings) -> ResolvedCredential:
    """Resolve the image key, endpoint and model for one project.

    Args:
        secrets: The project's BYOK bundle, if any.
        settings: Server settings providing ``IMAGE_*`` fallbacks.

    Returns:
        The paired credential with the OpenAI defaults filled in.

    """
    server_url = settings.image_base_url or DEFAULT_IMAGE_BASE_URL
    resolved = _resolve(
        secrets.image_api_key if secrets else None,
        secrets.image_base_url if secrets else None,
        secrets.image_model if secrets else None,
        settings.image_api_key,
        server_url,
        settings.image_model,
    )
    return ResolvedCredential(
        resolved.api_key,
        resolved.base_url or server_url,
        resolved.model or DEFAULT_IMAGE_MODEL,
        resolved.origin,
    )

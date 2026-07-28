"""Vendor-neutral image-generation interface and validation error.

Imports:
    ``abc`` defines the required asynchronous provider method.
    ``Path`` identifies the output artifact location.
    ``ProviderError`` provides the shared safe upstream-error type.
"""
from __future__ import annotations

import abc
from pathlib import Path

from app.providers.llm import ProviderError


class ImageProvider(abc.ABC):
    """Generates a still image matching a textual prompt.

    Implementations must write the result to ``output_path`` and never log
    the prompt's underlying script text or any API key.

    Attributes:
        name: Stable provider identifier used by factories and diagnostics.

    """

    name: str = "abstract"

    @abc.abstractmethod
    async def generate_image(
        self, prompt: str, width: int, height: int, output_path: Path
    ) -> Path:
        """Generate one image and write it to ``output_path``.

        Args:
            prompt: Visual description sent to the provider.
            width: Requested output width in pixels.
            height: Requested output height in pixels.
            output_path: Destination path that the implementation must create.

        Returns:
            The created output path.

        Raises:
            ProviderError: When the upstream service rejects or cannot create
                the requested image.

        """
        ...


class ImageValidationError(ProviderError):
    """Raised when an upstream image provider rejects the prompt or payload."""

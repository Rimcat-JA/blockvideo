"""Image provider abstraction."""
from __future__ import annotations

import abc
from pathlib import Path

from app.providers.llm import ProviderError


class ImageProvider(abc.ABC):
    """Generates a still image matching a textual prompt.

    Implementations must write the result to ``output_path`` and never log
    the prompt's underlying script text or any API key.
    """

    name: str = "abstract"

    @abc.abstractmethod
    async def generate_image(
        self, prompt: str, width: int, height: int, output_path: Path
    ) -> Path:
        """Generate an image and write it to the requested output path."""
        ...


class ImageValidationError(ProviderError):
    """Raised when the image prompt is rejected by the upstream provider."""

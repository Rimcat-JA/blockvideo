"""Fake image provider — draws a stylised placeholder PNG using PIL."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.providers.image import ImageProvider


class FakeImageProvider(ImageProvider):
    """Draw deterministic placeholder slides without a network request."""

    name = "fake"

    def __init__(self) -> None:
        """Create an empty call log for test assertions."""
        self.calls: list[tuple[str, int, int, str]] = []

    async def generate_image(
        self, prompt: str, width: int, height: int, output_path: Path
    ) -> Path:
        """Render a labeled placeholder image derived from the prompt hash."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:6]
        self.calls.append((prompt, width, height, h))

        # Try several font candidates that ship with most systems + render.
        img = Image.new("RGB", (width, height), color=(245, 247, 250))
        draw = ImageDraw.Draw(img)
        # Border
        draw.rectangle(
            [(20, 20), (width - 20, height - 20)],
            outline=(15, 118, 110),
            width=6,
        )
        font = self._pick_font(size=64)
        small = self._pick_font(size=36)
        heading = "FAKE IMAGE"
        sub = f"# {h}  {width}x{height}"
        # Draw title
        bbox = draw.textbbox((0, 0), heading, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - tw) // 2, (height // 2) - th), heading, fill=(15, 23, 42), font=font)
        bbox = draw.textbbox((0, 0), sub, font=small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - tw) // 2, (height // 2) + 20), sub, fill=(71, 85, 105), font=small)
        # Prompt truncated footer
        footer = prompt[:120] + ("…" if len(prompt) > 120 else "")
        bbox = draw.textbbox((0, 0), footer, font=small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((width - tw) // 2, height - th - 60),
            footer,
            fill=(100, 116, 139),
            font=small,
        )
        img.save(output_path, format="PNG")
        return output_path

    @staticmethod
    def _pick_font(size: int) -> ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

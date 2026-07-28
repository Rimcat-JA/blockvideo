"""Mermaid diagram renderer via mermaid-cli (mmdc).

Renders a Mermaid source string to a PNG by invoking ``mmdc`` as a subprocess
with an argument array (never a shell string). The Mermaid source is written
to a temporary ``.mmd`` file so no LLM-controlled text ever reaches a shell.

When mmdc is unavailable or rendering fails, callers should fall back to the
built-in PIL diagram renderer in :mod:`app.services.image_renderer`.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings, resolve_mmdc
from app.core.logging import log


@dataclass
class MermaidRenderResult:
    """Dimensions and output path returned by the async Mermaid wrapper."""

    output_path: Path
    width: int
    height: int


def mmdc_available() -> bool:
    """Return True if a usable mmdc executable is resolvable."""
    exe = resolve_mmdc()
    if exe == "mmdc":
        return shutil.which("mmdc") is not None
    return Path(exe).exists() or shutil.which(exe) is not None


def _escape_subprocess_arg(value: str) -> str:
    """Pass an argv element through without shell escaping.

    No shell is involved; we pass argv elements verbatim. Kept as a hook
    so callers see that no shell escaping is needed here.
    """
    return value


async def render_mermaid_to_png(
    source: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    theme: str = "default",
    background: str = "transparent",
    timeout: float | None = None,
) -> MermaidRenderResult:
    """Async wrapper — delegates to :func:`render_mermaid_to_png_sync`."""
    import asyncio

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: render_mermaid_to_png_sync(
            source,
            output_path,
            width=width,
            height=height,
            theme=theme,
            background=background,
            timeout=timeout,
        ),
    )
    return MermaidRenderResult(output_path=output_path, width=width, height=height)


def render_mermaid_to_png_sync(
    source: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    theme: str = "default",
    background: str = "transparent",
    timeout: float | None = None,
) -> Path:
    """Render ``source`` (Mermaid) to ``output_path`` as a PNG synchronously.

    Uses ``subprocess.run`` (argv array, ``shell=False``). Raises
    ``RuntimeError`` on any failure so the caller can fall back to the PIL
    diagram renderer. The Mermaid source is written to a temp ``.mmd`` file
    so no LLM-controlled text reaches a shell.
    """
    import subprocess
    import tempfile

    if not source.strip():
        raise RuntimeError("mermaid source is empty")

    exe = resolve_mmdc()
    if exe == "mmdc" and shutil.which("mmdc") is None:
        npx = shutil.which("npx")
        if npx is None:
            raise RuntimeError("mmdc も npx も見つかりません")
        base_argv: list[str] = [npx, "--yes", "mmdc"]
    else:
        base_argv = [exe]

    settings = get_settings()
    puppeteer_config = settings.mermaid_puppeteer_config
    timeout_seconds = timeout if timeout is not None else settings.mermaid_render_timeout

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as src_fp:
        src_fp.write(source)
        src_path = Path(src_fp.name)

    argv = base_argv + [
        "-i", str(src_path),
        "-o", str(output_path),
        "-w", str(width),
        "-H", str(height),
        "-t", theme,
        "-b", background,
    ]
    if puppeteer_config is not None and Path(puppeteer_config).exists():
        argv += ["--puppeteerConfigFile", str(puppeteer_config)]
    argv = [str(a) for a in argv if a not in (None, "")]

    log.info("mmdc を実行: {cmd} ...", cmd=" ".join(argv[:3]))
    try:
        try:
            proc = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"mmdc 実行がタイムアウトしました ({timeout_seconds}s)"
            ) from exc
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace")[:400] if proc.stderr else ""
            raise RuntimeError(f"mmdc 失敗 (rc={proc.returncode}): {err}")
        if not output_path.exists():
            raise RuntimeError("mmdc が出力ファイルを生成しませんでした")
        return output_path
    finally:
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass

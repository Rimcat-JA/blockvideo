"""Mermaid diagram renderer via Mermaid CLI (``mmdc``).

Renders a Mermaid source string to a PNG by invoking ``mmdc`` as a subprocess
with an argument array (never a shell string). The Mermaid source is written
to a temporary ``.mmd`` file so no LLM-controlled text ever reaches a shell.

When ``mmdc`` is unavailable or rendering fails, callers should fall back to
the built-in PIL diagram renderer in :mod:`app.services.image_renderer`.

Imports:
    ``shutil`` checks ``mmdc``/``npx`` availability.
    ``tempfile`` stores source outside shell arguments during rendering.
    Dataclasses/paths describe the result and destination.
    Configuration/logging modules resolve executables and record safe progress.
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
    """Dimensions and output path returned by Mermaid rendering.

    Attributes:
        output_path: PNG written by Mermaid CLI.
        width, height: Requested canvas dimensions reported to the caller.

    """

    output_path: Path
    width: int
    height: int


def mmdc_available() -> bool:
    """Return whether configured ``mmdc`` is discoverable.

    Returns:
        ``True`` for an explicit existing/path-resolvable executable or a
        globally installed bare ``mmdc``.  This probe does not test whether
        Puppeteer can launch successfully.

    """
    exe = resolve_mmdc()
    if exe == "mmdc":
        return shutil.which("mmdc") is not None
    return Path(exe).exists() or shutil.which(exe) is not None


def _escape_subprocess_arg(value: str) -> str:
    """Return an argv element unchanged because no shell is used.

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
    """Render Mermaid source without blocking the event loop.

    Args:
        source: Mermaid graph source.
        output_path: PNG destination.
        width: Requested CLI viewport width hint.
        height: Requested CLI viewport height hint.
        theme: Mermaid CLI theme name.
        background: Mermaid CLI background value.
        timeout: Optional per-process timeout override.

    Returns:
        ``MermaidRenderResult`` for the requested output path and dimensions.

    Raises:
        RuntimeError: Propagated from the synchronous renderer for missing
            executables, process failure, timeout, or missing output.

    """
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

    Args:
        source: Mermaid graph source written to a temporary ``.mmd`` file.
        output_path: PNG destination.
        width: Requested CLI viewport width hint.
        height: Requested CLI viewport height hint.
        theme: Mermaid CLI theme name.
        background: Mermaid CLI background value.
        timeout: Optional process timeout; otherwise settings default.

    Returns:
        The supplied ``output_path`` after CLI creation.

    Raises:
        RuntimeError: If source is empty, the executable cannot be resolved,
            the subprocess times out/fails, or no output file is produced.

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

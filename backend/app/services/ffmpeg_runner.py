"""FFmpeg/FFprobe command construction and asynchronous execution.

We *never* assemble shell strings; everything goes through subprocess with a
list of arguments to prevent shell-injection from LLM-controlled fields.
Imports:
    ``asyncio`` runs subprocesses without blocking the event loop.
    ``json`` parses FFprobe's duration response.
    ``shutil`` checks executable availability.
    ``time`` measures command duration.
    Dataclasses/paths represent clip inputs and output files.

All subprocesses receive argument arrays with ``shell=False`` semantics.  The
module does not implement a crossfade despite accepting the configuration
value: positive ``crossfade_seconds`` currently follows the same concat-copy
path as zero, and the function documentation makes that limitation explicit.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import resolve_ffmpeg, resolve_ffprobe
from app.core.logging import log
from app.providers.llm import ProviderError


# Letterbox colour for the slide region. Matches ``diagram_renderer.BG``
# (#F8FAFC) so any residual padding blends into the slide instead of showing
# as black bars. The subtitle band below stays black for contrast.
SLIDE_BG_HEX = "0xF8FAFC"


@dataclass
class BlockClip:
    """Description of one block's media inputs and output.

    Attributes:
        image_path: Primary slide image.
        audio_path: Narration WAV input.
        duration_ms: Target encoded clip duration.
        output_path: Destination MP4 path.
        subtitle_path: Optional per-block ASS file to burn in.

    """

    image_path: Path
    audio_path: Path
    duration_ms: int
    output_path: Path
    subtitle_path: Path | None = None


def _slide_fit_chain(width: int, height: int, band_height: int) -> str:
    """Filter chain that fits one slide image into the frame.

    With a subtitle band the slide occupies the upper
    ``height - band_height`` region and the lower band is left black for the
    burned-in captions; without one the slide fills the frame.

    Args:
        width: Final video frame width in pixels.
        height: Final video frame height in pixels.
        band_height: Caption band height; non-positive means no reserved band.

    Returns:
        A single FFmpeg video-filter expression that scales and letterboxes a
        slide, optionally reserving a black subtitle region.

    """
    if band_height > 0:
        slide_h = max(120, height - band_height)
        return (
            f"scale={width}:{slide_h}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{slide_h}:(ow-iw)/2:(oh-ih)/2:{SLIDE_BG_HEX},"
            f"pad={width}:{height}:0:0:black,"
            f"setsar=1"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:{SLIDE_BG_HEX},"
        f"setsar=1"
    )


def build_block_video_args(
    *,
    image: Path | None = None,
    audio: Path,
    duration_ms: int,
    output: Path,
    ffmpeg: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    subtitle_path: Path | None = None,
    subtitle_band_height: int = 0,
    slides: list[tuple[Path, int]] | None = None,
) -> list[str]:
    """Build the ffmpeg argv for one block's images+audio -> .mp4.

    ``slides`` is an ordered list of ``(image, duration_ms)``. A block whose
    script contained several visuals shows them in sequence underneath its
    narration, rather than only the first — the rest would otherwise never
    reach the screen. Passing a single ``image`` is the one-slide shorthand.

    When ``subtitle_path`` is given (and the file exists), the slide is
    scaled into the upper ``height - subtitle_band_height`` region and the
    lower band is left as a black bar where the burned-in subtitles render.
    The subtitle file is an .ass produced per-block (start=0).

    We never assemble shell strings — filter graphs are single argv elements
    and the subtitle path is escaped for the libass filter only.

    Args:
        image: Single-slide shorthand used when ``slides`` is absent.
        audio: Narration input path.
        duration_ms: Target clip length, including trailing hold.
        output: Destination MP4 path.
        ffmpeg: Executable name/path.
        width: Output video width in pixels.
        height: Output video height in pixels.
        fps: Output frame rate.
        subtitle_path: Optional existing ASS file.
        subtitle_band_height: Reserved lower-frame height when subtitles exist.
        slides: Ordered ``(image, duration_ms)`` pairs for multi-slide blocks.

    Returns:
        FFmpeg argv suitable for ``asyncio.create_subprocess_exec``.

    Raises:
        ValueError: If neither ``image`` nor ``slides`` supplies an input.

    """
    duration_seconds = max(0.1, duration_ms / 1000.0)
    if not slides:
        if image is None:
            raise ValueError("either image or slides is required")
        slides = [(image, duration_ms)]

    burn_subs = bool(subtitle_path and subtitle_path.exists())
    band = subtitle_band_height if burn_subs else 0
    fit = _slide_fit_chain(width, height, band)
    sub_filter = f",subtitles={_escape_sub_path(subtitle_path)}" if burn_subs else ""

    args: list[str] = [ffmpeg, "-y"]
    for slide_image, slide_ms in slides:
        args += [
            "-loop", "1",
            "-framerate", str(fps),
            "-t", f"{max(0.1, slide_ms / 1000.0):.3f}",
            "-i", str(slide_image),
        ]
    args += ["-i", str(audio)]

    if len(slides) == 1:
        args += ["-vf", fit + sub_filter]
    else:
        # Fit each slide identically, then concatenate and burn subtitles on
        # the joined stream so cue timings stay relative to the whole block.
        parts = [f"[{i}:v]{fit}[v{i}]" for i in range(len(slides))]
        joined = "".join(f"[v{i}]" for i in range(len(slides)))
        parts.append(f"{joined}concat=n={len(slides)}:v=1:a=0{sub_filter}[vout]")
        args += [
            "-filter_complex", ";".join(parts),
            "-map", "[vout]",
            "-map", f"{len(slides)}:a",
        ]

    args += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-profile:v", "high",
        "-movflags", "+faststart",
        "-r", str(fps),
        # Pad the audio with silence so the clip runs for the full display
        # duration. Without this the old ``-shortest`` cut every block at the
        # last syllable, which silently discarded the trailing hold between
        # blocks — ``-t`` alone cannot extend past the shortest input.
        "-af", "apad",
        "-t", f"{duration_seconds:.3f}",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "192k",
        str(output),
    ]
    return args


def _escape_sub_path(path: Path) -> str:
    r"""Escape an .ass path for the libass ``subtitles=`` filter on Windows.

    The libass filter uses ``:`` as an option separator, so the drive-letter
    colon must be escaped (``\\:``) and the whole path wrapped in single
    quotes (libass treats them literally). Forward slashes are used in place
    of backslashes. This is filter-syntax escaping only — no shell is
    involved (the whole ``-vf`` value is one argv element).

    Args:
        path: ASS path embedded in the libass filter value.

    Returns:
        Filter-syntax escaped path, with forward slashes and escaped drive
        colon.  It is not shell escaping because no shell is used.

    """
    p = str(path).replace("\\", "/").replace(":", "\\:")
    return f"'{p}'"


def build_concat_args(*, list_file: Path, output: Path, ffmpeg: str, crossfade_seconds: float) -> list[str]:
    """Build FFmpeg argv that concatenates block MP4s.

    With ``crossfade_seconds == 0`` the concat demuxer copies streams without
    re-encoding.  Positive values are accepted for API compatibility but are
    currently ignored and use the same concat-copy command; no crossfade is
    emitted by this implementation.

    Args:
        list_file: FFmpeg concat-demuxer file.
        output: Final output path.
        ffmpeg: Executable name/path.
        crossfade_seconds: Reserved future crossfade setting.

    Returns:
        FFmpeg argv for the current concat-copy implementation.

    """
    if crossfade_seconds <= 0:
        return [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output),
        ]
    # simpler safe default: no crossfade even if requested, to avoid
    # requiring decoding all blocks twice. Users can override later.
    return [
        ffmpeg,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output),
    ]


async def run_ffmpeg(args: list[str], *, log_path: Path | None = None) -> None:
    """Execute ffmpeg, writing the log to ``log_path`` if provided.

    Args:
        args: Complete FFmpeg argv; ``args[0]`` is the executable.
        log_path: Optional combined stdout/stderr file, defaulting to
            ``.ffmpeg.log`` in the current directory.

    Raises:
        ProviderError: If the executable is unavailable, the process times
            out, or exits non-zero.

    Side Effects:
        Creates the log directory/file and writes the encoded output selected
        by the supplied command.

    """
    exe = args[0]
    if not shutil.which(exe) and not Path(exe).exists():
        raise ProviderError(
            f"ffmpegが見つかりません ({exe})。FFmpegをインストールしてください。",
            safe=True,
        )
    output_path = log_path or Path(".ffmpeg.log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("ffmpegを実行: {cmd} ...", cmd=" ".join(args[:6]))
    started = time.monotonic()
    with output_path.open("wb") as fp:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=fp,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=1800.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise ProviderError("ffmpeg実行がタイムアウトしました", safe=True) from None
    elapsed = time.monotonic() - started
    if rc != 0:
        raise ProviderError(
            f"ffmpeg失敗 (rc={rc})。詳細は {output_path} を確認してください。",
            safe=True,
        )
    log.info("ffmpeg成功 elapsed_seconds={elapsed}", elapsed=round(elapsed, 2))


async def ffprobe_duration_ms(path: Path) -> int:
    """Run FFprobe and return a media duration in milliseconds.

    Args:
        path: Audio/video file passed to FFprobe.

    Returns:
        Rounded duration from FFprobe's ``format.duration`` field.

    Raises:
        ProviderError: If FFprobe is unavailable or returns a non-zero status.
        json.JSONDecodeError: If FFprobe emits invalid JSON.

    """
    ffprobe = resolve_ffprobe()
    if not shutil.which(ffprobe) and not Path(ffprobe).exists():
        raise ProviderError(
            f"ffprobeが見つかりません ({ffprobe})。FFmpegをインストールしてください。",
            safe=True,
        )
    proc = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise ProviderError(
            f"ffprobe失敗 (rc={proc.returncode}): {stderr.decode('utf-8', 'replace')[:200]}",
            safe=True,
        )
    payload = json.loads(stdout.decode("utf-8") or "{}")
    duration = float(payload.get("format", {}).get("duration", 0.0))
    return int(round(duration * 1000))


def ffprobe_available() -> bool:
    """Return whether the configured FFprobe executable is discoverable."""
    ffprobe = resolve_ffprobe()
    return bool(shutil.which(ffprobe)) or Path(ffprobe).exists()


def ffmpeg_available() -> bool:
    """Return whether the configured FFmpeg executable is discoverable."""
    ffmpeg = resolve_ffmpeg()
    return bool(shutil.which(ffmpeg)) or Path(ffmpeg).exists()


def write_concat_list(items: list[Path], list_file: Path) -> None:
    """Write absolute media paths in concat-demuxer syntax.

    Args:
        items: Ordered media files to concatenate.
        list_file: Destination list file.

    Side Effects:
        Creates the destination directory and writes one escaped ``file`` line
        per input.  The order is preserved exactly.

    """
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with list_file.open("w", encoding="utf-8") as fp:
        for item in items:
            safe = str(item.resolve()).replace("'", "\\'")
            fp.write(f"file '{safe}'\n")

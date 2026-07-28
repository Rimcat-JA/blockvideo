"""Precise SICP-style diagram rendering with PIL.

Mermaid flowcharts cannot express box-and-pointer (cons cell) diagrams or
environment-model diagrams — the two staples of SICP-style teaching
material. Rather than coaxing Mermaid into an approximation, the LLM emits a
*structured spec* (see :mod:`app.services.stage_schemas`) and this module
draws it exactly.

Two diagram families are supported:

``pointer_diagram``
    Cons cells drawn as the classic two-slot box. ``car`` holds a value,
    ``cdr`` either points right to the next cell, terminates with the nil
    slash, or loops back to an earlier cell (drawn as a routed arrow beneath
    the row, which is how circular lists read best).

``env_diagram``
    Environment frames with their bindings, parent-frame pointers, and
    procedure objects (the classic two-circle blob with params/body and an
    environment pointer).

Both renderers lay the content out at a generous natural size, measure the
resulting bounding box, then scale it once to fit the target canvas. The
output is therefore **always exactly ``width`` x ``height``** on the theme
background, which matters downstream: ffmpeg letterboxes anything that does
not match the frame aspect, and an undersized diagram would otherwise be
blur-upscaled between black bars.
Imports:
    ``math`` draws arrow geometry.
    ``unicodedata`` measures East Asian display width.
    Dataclasses/paths/types describe outputs and loose model-produced specs.
    PIL creates canvases, text, fonts, and drawing primitives.
    ``log`` is available for defensive renderer diagnostics.

Module state:
    Theme colors and layout constants below define the visual language and the
    natural supersampled canvas size.  They are immutable conventions, not
    per-request configuration.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.core.logging import log


# --------------------------------------------------------------------------
# Theme colors — match the "落ち着いた大学講義風" visual style.
# --------------------------------------------------------------------------

BG = (248, 250, 252)
INK = (15, 23, 42)
MUTED = (100, 116, 139)
ACCENT = (0, 139, 139)
ACCENT_SOFT = (204, 233, 233)
BOX_FILL = (255, 255, 255)
BOX_EDGE = (51, 65, 85)
ARROW = (30, 41, 59)
CYCLE_ARROW = (190, 88, 24)
GROUP_EDGE = (203, 213, 225)

# Natural layout constants. Deliberately generous: the layout is drawn at
# roughly 2x the size it will usually occupy, then scaled down once to fit.
# Supersampling this way means a diagram that needs to grow to fill the frame
# still has real pixels behind it instead of being blur-upscaled.
SLOT = 168         # width of one cons-cell slot; a cell is 2 slots wide
CELL_H = 168
CELL_GAP = 200     # horizontal gap between adjacent cells
ROW_GAP = 210      # vertical gap between groups
CYCLE_LANE = 150   # vertical space reserved under a row for loop-back arrows
ROOT_RISE = 190    # height of the root pointer stub above a cell
PAD = 90           # padding inside the measured content box
LINE_W = 6


@dataclass
class DiagramResult:
    """Path and dimensions produced by a structured diagram renderer.

    Attributes:
        output_path: PNG destination written by the renderer.
        width, height: Exact final canvas dimensions.

    """

    output_path: Path
    width: int
    height: int


_FONT_CANDIDATES = [
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_MONO_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _font(size: int, mono: bool = False):
    """Load the first usable proportional or monospace system font.

    Args:
        size: Requested point/pixel size passed to PIL.
        mono: Select the monospace candidate list when true.

    Returns:
        A PIL font object; PIL's built-in default is the final fallback.

    """
    for path in (_MONO_CANDIDATES if mono else _FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Measure text using PIL and a defensive fallback estimate."""
    if not text:
        return (0, 0)
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return (box[2] - box[0], box[3] - box[1])
    except Exception:  # pragma: no cover - defensive
        return (len(text) * font.size // 2, font.size)


def _has_cjk(text: str) -> bool:
    """Return whether text contains kana, CJK, or full-width characters."""
    return any(
        "　" <= ch <= "ヿ"      # kana + CJK punctuation
        or "一" <= ch <= "鿿"   # kanji
        or "＀" <= ch <= "￯"   # full-width forms
        for ch in text
    )


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int,
              start: int = 34, mono: bool = False):
    """Largest font at which *text* fits inside (max_w, max_h).

    The monospace candidates are ASCII-only code fonts (Consolas et al), so
    any text containing Japanese falls back to the proportional CJK stack —
    otherwise every kana renders as a tofu box.
    """
    if mono and _has_cjk(text):
        mono = False
    size = start
    while size > 12:
        f = _font(size, mono=mono)
        w, h = _text_size(draw, text, f)
        if w <= max_w and h <= max_h:
            return f
        size -= 2
    return _font(12, mono=mono)


def _centered(draw: ImageDraw.ImageDraw, text: str, cx: int, cy: int, font,
              fill=INK) -> None:
    """Draw text centered around a supplied point."""
    w, h = _text_size(draw, text, font)
    draw.text((cx - w / 2, cy - h / 2), text, font=font, fill=fill)


def _rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline,
             width: int = LINE_W) -> None:
    """Draw a themed rounded rectangle used by diagram nodes and frames."""
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline,
                           width=width)


def _arrow_head(draw: ImageDraw.ImageDraw, tip: tuple[float, float],
                angle: float, color, size: int = 30) -> None:
    """Draw a triangular arrowhead at a line endpoint."""
    a1 = angle + math.radians(150)
    a2 = angle - math.radians(150)
    p1 = (tip[0] + size * math.cos(a1), tip[1] + size * math.sin(a1))
    p2 = (tip[0] + size * math.cos(a2), tip[1] + size * math.sin(a2))
    draw.polygon([tip, p1, p2], fill=color)


def _arrow(draw: ImageDraw.ImageDraw, start, end, color=ARROW,
           width: int = LINE_W, head: int = 30) -> None:
    """Draw one straight arrow between two points."""
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    _arrow_head(draw, end, angle, color, head)


def _polyline_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]],
                    color=ARROW, width: int = LINE_W, head: int = 30) -> None:
    """Draw a routed arrow through a sequence of points."""
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")
    p0, p1 = points[-2], points[-1]
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    _arrow_head(draw, p1, angle, color, head)


def _dot(draw: ImageDraw.ImageDraw, center, r: int = 14, fill=ARROW) -> None:
    """Draw a filled endpoint marker used in procedure and reference arrows."""
    cx, cy = center
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


# --------------------------------------------------------------------------
# Canvas composition
# --------------------------------------------------------------------------

def _compose(content: Image.Image, width: int, height: int,
             title: str, caption: str | None) -> Image.Image:
    """Scale *content* to fit and centre it on a fixed themed canvas.

    Never upscales past 1.0 — the natural layout is already drawn large, so
    a scale factor above 1 would only soften the result.
    """
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    top = 34
    title_h = 0
    if title:
        tfont = _fit_font(draw, title, width - 160, 78, start=64)
        tw, th = _text_size(draw, title, tfont)
        draw.text(((width - tw) / 2, top), title, font=tfont, fill=ACCENT)
        title_h = th + 26
        draw.line([(width / 2 - tw / 2, top + th + 14),
                   (width / 2 + tw / 2, top + th + 14)],
                  fill=ACCENT_SOFT, width=4)

    cap_h = 0
    cap_font = None
    if caption:
        cap_font = _fit_font(draw, caption, width - 200, 52, start=34)
        _, cap_h = _text_size(draw, caption, cap_font)
        cap_h += 30

    avail_w = width - 100
    avail_h = height - top - title_h - cap_h - 60
    scale = min(avail_w / content.width, avail_h / content.height, 1.0)
    new_size = (max(1, int(content.width * scale)), max(1, int(content.height * scale)))
    resized = content.resize(new_size, Image.LANCZOS)

    ox = int((width - resized.width) / 2)
    oy = int(top + title_h + (avail_h - resized.height) / 2)
    canvas.paste(resized, (ox, oy), resized if resized.mode == "RGBA" else None)

    if caption and cap_font is not None:
        cw, _ = _text_size(draw, caption, cap_font)
        draw.text(((width - cw) / 2, height - cap_h - 6), caption,
                  font=cap_font, fill=MUTED)
    return canvas


def compose_on_canvas(content: Image.Image, *, width: int, height: int,
                      title: str = "", caption: str | None = None) -> Image.Image:
    """Compose content onto an exact-size themed canvas.

    Used by the Mermaid path too: ``mmdc`` treats ``-w``/``-H`` as viewport
    hints, so it emits images at whatever size the graph happens to need.
    Composing them here guarantees a full-frame, correctly-padded slide.

    Args:
        content: Already-rendered RGBA/RGB image to fit.
        width: Exact output width in pixels.
        height: Exact output height in pixels.
        title: Optional top heading.
        caption: Optional bottom caption.

    Returns:
        New RGB image with the content scaled/centered and no more than 1x
        enlarged.

    """
    return _compose(content, width, height, title, caption)


def _blank_layer(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a transparent RGBA drawing layer and its PIL draw object."""
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


# --------------------------------------------------------------------------
# Box-and-pointer (cons cell) diagrams
# --------------------------------------------------------------------------

def _slot_value(node: Any) -> tuple[str, str]:
    """Normalise a car/cdr node into (kind, text).

    Accepts the schema form ``{"kind": ..., "text"/"target": ...}`` and also
    tolerates a bare string, which models reach for often enough that being
    strict here would cost diagrams for no benefit.
    """
    if node is None:
        return ("nil", "")
    if isinstance(node, str):
        s = node.strip()
        if s.lower() in {"nil", "null", "()", "'()", "empty"}:
            return ("nil", "")
        return ("value", s)
    if isinstance(node, dict):
        kind = str(node.get("kind") or "").strip().lower()
        if kind == "ref":
            return ("ref", str(node.get("target") or "").strip())
        if kind == "nil":
            return ("nil", "")
        if kind == "value":
            return ("value", str(node.get("text") or "").strip())
        if node.get("target"):
            return ("ref", str(node["target"]).strip())
        if node.get("text") is not None:
            return ("value", str(node["text"]).strip())
    return ("nil", "")


def _draw_cons_group(draw: ImageDraw.ImageDraw, group: dict, x0: int, y0: int) -> tuple[int, int]:
    """Draw one labelled group of cells. Returns (width, height) consumed."""
    cells = [c for c in (group.get("cells") or []) if isinstance(c, dict) and c.get("id")]
    if not cells:
        return (0, 0)

    label = (group.get("label") or "").strip()
    label_h = 0
    if label:
        lf = _font(58)
        _, lh = _text_size(draw, label, lf)
        draw.text((x0, y0), label, font=lf, fill=MUTED)
        label_h = lh + 26

    row_y = y0 + label_h + ROOT_RISE
    positions: dict[str, tuple[int, int]] = {}
    cell_w = SLOT * 2

    for i, cell in enumerate(cells):
        cx = x0 + i * (cell_w + CELL_GAP)
        positions[str(cell["id"])] = (cx, row_y)

    # --- cells ---
    for cell in cells:
        cid = str(cell["id"])
        cx, cy = positions[cid]
        box = [cx, cy, cx + cell_w, cy + CELL_H]
        _rounded(draw, box, 14, BOX_FILL, BOX_EDGE, LINE_W)
        draw.line([(cx + SLOT, cy), (cx + SLOT, cy + CELL_H)],
                  fill=BOX_EDGE, width=LINE_W)

        car_kind, car_text = _slot_value(cell.get("car"))
        if car_kind == "nil":
            draw.line([(cx + 18, cy + CELL_H - 18), (cx + SLOT - 18, cy + 18)],
                      fill=BOX_EDGE, width=LINE_W)
        elif car_text:
            f = _fit_font(draw, car_text, SLOT - 34, CELL_H - 40, start=62)
            _centered(draw, car_text, cx + SLOT // 2, cy + CELL_H // 2, f, INK)

        cdr_kind, _ = _slot_value(cell.get("cdr"))
        if cdr_kind == "nil":
            draw.line([(cx + SLOT + 18, cy + CELL_H - 18),
                       (cx + cell_w - 18, cy + 18)], fill=BOX_EDGE, width=LINE_W)
        elif cdr_kind == "ref":
            _dot(draw, (cx + SLOT + SLOT // 2, cy + CELL_H // 2))
        else:
            _, txt = _slot_value(cell.get("cdr"))
            if txt:
                f = _fit_font(draw, txt, SLOT - 34, CELL_H - 40, start=62)
                _centered(draw, txt, cx + SLOT + SLOT // 2, cy + CELL_H // 2, f, INK)

    # --- cdr arrows ---
    order = {str(c["id"]): i for i, c in enumerate(cells)}
    lane_used = 0
    for cell in cells:
        cid = str(cell["id"])
        kind, target = _slot_value(cell.get("cdr"))
        if kind != "ref" or target not in positions:
            continue
        cx, cy = positions[cid]
        tx, ty = positions[target]
        src = (cx + SLOT + SLOT // 2, cy + CELL_H // 2)
        if order[target] > order[cid]:
            # forward link: straight arrow into the target's left edge
            _arrow(draw, (src[0] + 16, src[1]), (tx - 16, ty + CELL_H // 2))
        else:
            # backward link (cycle): route below the row so it stays readable
            lane_used = CYCLE_LANE
            lane_y = cy + CELL_H + CYCLE_LANE - 34
            pts = [
                src,
                (src[0], lane_y),
                (tx + SLOT // 2, lane_y),
                (tx + SLOT // 2, ty + CELL_H + 18),
            ]
            _polyline_arrow(draw, pts, color=CYCLE_ARROW)

    # --- root pointers ---
    for root in (group.get("roots") or []):
        if not isinstance(root, dict):
            continue
        target = str(root.get("target") or "").strip()
        if target not in positions:
            continue
        tx, ty = positions[target]
        name = (root.get("label") or "").strip()
        anchor_x = tx + SLOT // 2
        # Reserve the top of the stub for the label so the arrow shaft never
        # runs through the glyphs.
        stub_top = ty - ROOT_RISE + 26
        if name:
            f = _font(56)
            w, h = _text_size(draw, name, f)
            draw.text((anchor_x - w / 2, stub_top - 12), name, font=f, fill=ACCENT)
            stub_top += h + 26
        _arrow(draw, (anchor_x, stub_top), (anchor_x, ty - 14), color=ACCENT,
               width=LINE_W, head=26)

    total_w = len(cells) * cell_w + (len(cells) - 1) * CELL_GAP
    total_h = label_h + ROOT_RISE + CELL_H + lane_used
    return (total_w, total_h)


# Grid fonts for author-drawn slides. Consolas has uniform advances and the
# box-drawing set but no CJK and no ▶; MS Gothic has everything but renders
# CJK at exactly twice the ASCII width. Neither alone gives an aligned grid,
# so characters are placed per cell and the font is chosen per character.
_GRID_MONO = [
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_GRID_WIDE = [
    "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load(paths: list[str], size: int):
    """Load the first available font path or PIL's built-in fallback font."""
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


_NOTDEF_CACHE: dict[int, bytes] = {}


def _render_glyph(font, ch: str) -> bytes:
    """Rasterize one glyph into a comparable bitmap for coverage checks."""
    img = Image.new("L", (72, 72), 0)
    ImageDraw.Draw(img).text((4, 4), ch, font=font, fill=255)
    return img.tobytes()


def _has_glyph(font, ch: str) -> bool:
    """Whether *font* actually draws *ch* rather than a .notdef box.

    PIL exposes no glyph-coverage API, so the character is compared against
    a codepoint guaranteed to be unassigned: identical bitmaps mean tofu.
    """
    key = id(font)
    notdef = _NOTDEF_CACHE.get(key)
    if notdef is None:
        notdef = _render_glyph(font, "\uffff")
        _NOTDEF_CACHE[key] = notdef
    return _render_glyph(font, ch) != notdef


def _cells(ch: str) -> int:
    """How many grid cells a character occupies.

    East Asian "ambiguous" characters — the box-drawing set, ●, arrows — are
    treated as narrow, which is how a monospaced editor shows them and
    therefore how the art was drawn.
    """
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def render_verbatim_slide(
    body: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    title: str = "",
    caption: str | None = None,
) -> None:
    """Draw slide content exactly as the script author wrote it.

    Nothing here interprets the text. When a model designs a diagram from a
    schema it has to invent identifiers for anything the script left implicit,
    and it does: a real project came back with 115 labels — ``hdr``, ``k2``,
    ``2nd1``, ``local*`` — that appear nowhere in the script the viewer is
    listening to. Text the author supplied cannot invent anything.

    Alignment is the whole point of hand-drawn diagrams, so the block is laid
    out as one monospaced unit: every line is drawn at the same left edge and
    a fixed line pitch, with leading spaces intact. The font is the largest
    that fits, which keeps a four-line sketch readable and still admits a
    thirty-line listing.

    Args:
        body: Author-authored monospaced slide content.
        output_path: PNG destination.
        width: Exact output canvas width in pixels.
        height: Exact output canvas height in pixels.
        title: Optional heading above the drawing.
        caption: Optional explanatory caption below it.

    Side Effects:
        Creates the destination directory and writes a PNG.  Content is
        rendered faithfully apart from tab expansion and common indentation
        removal.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = (body or "").replace("\t", "    ").rstrip().splitlines() or [""]
    # Trim the common indentation so a block indented inside the script does
    # not sit against the right edge of the slide.
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if indents:
        cut = min(indents)
        lines = [l[cut:] if l.strip() else "" for l in lines]

    inner_w = max(1, width - PAD * 2)
    inner_h = max(1, height - PAD * 2 - (110 if title else 0) - (70 if caption else 0))

    columns = max(sum(_cells(ch) for ch in line) for line in lines) or 1

    size = 64
    while size > 12:
        mono = _load(_GRID_MONO, size)
        cell_w = mono.getlength("0") or size * 0.55
        # The font's own line height, not a padded one: box-drawing glyphs are
        # cut to join at exactly that pitch, and any extra leading leaves the
        # verticals of a hand-drawn box visibly disconnected.
        pitch = sum(mono.getmetrics())
        if cell_w * columns <= inner_w and pitch * len(lines) <= inner_h:
            break
        size -= 2
    else:
        mono = _load(_GRID_MONO, 12)
        cell_w = mono.getlength("0") or 7
        pitch = sum(mono.getmetrics())

    # Sized so one full-width glyph spans exactly two cells, keeping Japanese
    # labels on the same grid as the box-drawing around them.
    wide = _load(_GRID_WIDE, max(8, int(round(cell_w * 2))))

    block_w = max(1, int(cell_w * columns) + 4)
    block_h = pitch * len(lines)
    content, draw = _blank_layer(block_w, block_h)
    for row, line in enumerate(lines):
        col = 0
        for ch in line:
            if ch != " ":
                # Consolas keeps every advance identical, so it is preferred
                # wherever it has the glyph; anything it lacks (▶, kana) is
                # drawn from the wide font at its cell position.
                font = mono if _has_glyph(mono, ch) else wide
                draw.text((col * cell_w, row * pitch), ch, font=font, fill=INK)
            col += _cells(ch)

    _compose(content, width, height, title, caption).save(output_path)


def render_pointer_diagram(spec: dict, output_path: Path, *, width: int,
                           height: int) -> DiagramResult:
    """Render a structured box-and-pointer diagram to a fixed canvas.

    Args:
        spec: Mapping containing ``groups`` or a compatible top-level ``cells``
            list, optional roots/title/caption.
        output_path: PNG destination.
        width: Exact final canvas width in pixels.
        height: Exact final canvas height in pixels.

    Returns:
        ``DiagramResult`` describing the written image.

    Raises:
        ValueError: If no cells exist or the measured layout is empty.

    """
    groups = [g for g in (spec.get("groups") or []) if isinstance(g, dict)]
    if not groups:
        # tolerate a flat spec with cells at the top level
        if spec.get("cells"):
            groups = [{"label": "", "cells": spec["cells"], "roots": spec.get("roots") or []}]
        else:
            raise ValueError("pointer_diagram spec has no cells")

    # Measure with a throwaway layer, then draw for real.
    probe, pdraw = _blank_layer(4000, 3000)
    sizes = [_draw_cons_group(pdraw, g, 0, 0) for g in groups]
    content_w = max((w for w, _ in sizes), default=0)
    content_h = sum(h for _, h in sizes) + ROW_GAP * max(0, len(groups) - 1)
    if content_w <= 0 or content_h <= 0:
        raise ValueError("pointer_diagram produced an empty layout")

    layer, draw = _blank_layer(content_w + PAD * 2, content_h + PAD * 2)
    y = PAD
    for group, (gw, gh) in zip(groups, sizes):
        if gh == 0:
            continue
        _draw_cons_group(draw, group, PAD + (content_w - gw) // 2, y)
        y += gh + ROW_GAP

    canvas = _compose(layer, width, height,
                      str(spec.get("title") or "").strip(),
                      str(spec.get("caption") or "").strip() or None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return DiagramResult(output_path=output_path, width=width, height=height)


# --------------------------------------------------------------------------
# Environment-model diagrams
# --------------------------------------------------------------------------

FRAME_W = 620
BIND_H = 62
FRAME_PAD = 26
FRAME_GAP_Y = 150
PROC_R = 42


def _frame_height(frame: dict) -> int:
    """Calculate an environment frame's natural height from its bindings."""
    bindings = [b for b in (frame.get("bindings") or []) if isinstance(b, dict)]
    return FRAME_PAD * 2 + 54 + max(1, len(bindings)) * BIND_H


def render_env_diagram(spec: dict, output_path: Path, *, width: int,
                       height: int) -> DiagramResult:
    """Render an environment-model diagram of frames and procedures.

    Args:
        spec: Mapping containing non-empty ``frames`` and optional procedures,
            title, and caption.
        output_path: PNG destination.
        width: Exact final canvas width in pixels.
        height: Exact final canvas height in pixels.

    Returns:
        ``DiagramResult`` describing the written image.

    Raises:
        ValueError: If the specification has no identified frames.

    """
    frames = [f for f in (spec.get("frames") or []) if isinstance(f, dict) and f.get("id")]
    if not frames:
        raise ValueError("env_diagram spec has no frames")
    procs = [p for p in (spec.get("procedures") or []) if isinstance(p, dict) and p.get("id")]

    heights = [_frame_height(f) for f in frames]
    col_h = sum(heights) + FRAME_GAP_Y * max(0, len(frames) - 1)
    proc_col_w = 520 if procs else 0
    content_w = FRAME_W + (proc_col_w + 140 if procs else 0)
    proc_h = sum(180 for _ in procs) + 60 * max(0, len(procs) - 1)
    content_h = max(col_h, proc_h)

    layer, draw = _blank_layer(content_w + PAD * 2, content_h + PAD * 2)

    positions: dict[str, tuple[int, int, int]] = {}
    y = PAD
    for frame, fh in zip(frames, heights):
        x = PAD
        fid = str(frame["id"])
        positions[fid] = (x, y, fh)
        label = (frame.get("label") or fid).strip()

        _rounded(draw, [x, y, x + FRAME_W, y + fh], 14, BOX_FILL, ACCENT, LINE_W)
        lf = _fit_font(draw, label, FRAME_W - 40, 46, start=38)
        draw.text((x + FRAME_PAD, y + FRAME_PAD - 6), label, font=lf, fill=ACCENT)
        draw.line([(x + FRAME_PAD, y + FRAME_PAD + 44),
                   (x + FRAME_W - FRAME_PAD, y + FRAME_PAD + 44)],
                  fill=GROUP_EDGE, width=3)

        by = y + FRAME_PAD + 58
        for b in (frame.get("bindings") or []):
            if not isinstance(b, dict):
                continue
            name = str(b.get("name") or "").strip()
            value = str(b.get("value") or "").strip()
            nf = _fit_font(draw, name, 220, BIND_H - 16, start=34, mono=True)
            draw.text((x + FRAME_PAD + 6, by + 8), name, font=nf, fill=INK)
            draw.text((x + FRAME_PAD + 236, by + 8), ":", font=nf, fill=MUTED)
            if value:
                vf = _fit_font(draw, value, FRAME_W - 320, BIND_H - 16, start=34, mono=True)
                draw.text((x + FRAME_PAD + 266, by + 8), value, font=vf, fill=INK)
            by += BIND_H
        y += fh + FRAME_GAP_Y

    # parent-frame pointers (child -> parent, drawn upward on the left rail)
    for frame in frames:
        parent = str(frame.get("parent") or "").strip()
        if not parent or parent not in positions:
            continue
        cx, cy, _ch = positions[str(frame["id"])]
        px, py, ph = positions[parent]
        rail = cx - 34
        pts = [
            (cx, cy + 40),
            (rail, cy + 40),
            (rail, py + ph - 30),
            (px, py + ph - 30),
        ]
        _polyline_arrow(draw, pts, color=MUTED, width=3, head=18)

    # procedure objects
    if procs:
        px0 = PAD + FRAME_W + 140
        py = PAD
        for proc in procs:
            label = (proc.get("label") or "").strip()
            params = (proc.get("params") or "").strip()
            body = (proc.get("body") or "").strip()
            cy = py + PROC_R + 10
            _rounded(draw, [px0, py, px0 + proc_col_w, py + 160], 14,
                     BOX_FILL, BOX_EDGE, 3)
            draw.ellipse([px0 + 26, cy - PROC_R, px0 + 26 + PROC_R * 2, cy + PROC_R],
                         outline=BOX_EDGE, width=LINE_W)
            draw.ellipse([px0 + 40 + PROC_R * 2, cy - PROC_R,
                          px0 + 40 + PROC_R * 4, cy + PROC_R],
                         outline=BOX_EDGE, width=LINE_W)
            _dot(draw, (px0 + 26 + PROC_R, cy))
            _dot(draw, (px0 + 40 + PROC_R * 3, cy))
            if label:
                f = _fit_font(draw, label, proc_col_w - 60, 42, start=34)
                draw.text((px0 + 26, py + 122), label, font=f, fill=ACCENT)
            detail = f"params: {params}" if params else ""
            if body:
                detail = (detail + "   " if detail else "") + f"body: {body}"
            if detail:
                f = _fit_font(draw, detail, proc_col_w - 60, 38, start=28, mono=True)
                draw.text((px0 + 26, py + 122 + 44), detail, font=f, fill=MUTED)

            env = str(proc.get("env") or "").strip()
            if env in positions:
                ex, ey, eh = positions[env]
                _polyline_arrow(
                    draw,
                    [(px0 + 40 + PROC_R * 3, cy), (px0 + 40 + PROC_R * 3, cy - 70),
                     (ex + FRAME_W + 20, cy - 70), (ex + FRAME_W + 12, ey + eh // 2)],
                    color=MUTED, width=3, head=18,
                )
            py += 220

    canvas = _compose(layer, width, height,
                      str(spec.get("title") or "").strip(),
                      str(spec.get("caption") or "").strip() or None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return DiagramResult(output_path=output_path, width=width, height=height)

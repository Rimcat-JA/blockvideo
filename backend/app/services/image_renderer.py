"""Local PIL image rendering for every supported visual type.

Renders text, title, comparison, code, formula, author-drawn, structured
diagram, and Mermaid visuals to PNG.  PIL performs the local drawing; Mermaid
CLI is used only for Mermaid graph sources before a PIL canvas composition.
We never ship fonts in the repo — the renderer relies on system-installed fonts.

Imports:
    ``re`` parses the small Mermaid fallback subset.
    Dataclasses/paths/types describe render results and loose plan payloads.
    PIL performs image, text, font, and drawing operations.
    Model/services supply visual enum values, structured diagram renderers,
    Mermaid CLI, and safe diagnostics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.core.logging import log
from app.models.block import VisualType
from app.services.diagram_renderer import (
    BG as DIAGRAM_BG,
    compose_on_canvas,
    render_env_diagram,
    render_pointer_diagram,
    render_verbatim_slide,
)
from app.services.mermaid_renderer import mmdc_available, render_mermaid_to_png_sync


@dataclass
class RenderResult:
    """Path and dimensions of one rendered slide image.

    Attributes:
        output_path: PNG written by the selected renderer.
        width, height: Exact output dimensions requested by the pipeline.

    """

    output_path: Path
    width: int
    height: int


def _pick_font(size: int) -> ImageFont.ImageFont:
    """Choose the first installed font capable of rendering common UI text."""
    candidates = [
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Wrap *text* to fit *max_width* using actual glyph metrics.

    Falls back to character-based wrap when metrics are unavailable.
    """
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            candidate = current + ch
            try:
                bbox = draw.textbbox((0, 0), candidate, font=font)
                width = bbox[2] - bbox[0]
            except Exception:
                width = len(candidate)
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def _fill_background(img: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    """Fill a slide canvas with the shared diagram background color."""
    draw.rectangle([(0, 0), img.size], fill=DIAGRAM_BG)


def render_text_slide(
    output_path: Path,
    *,
    width: int,
    height: int,
    heading: str,
    body: str,
    bg_color: tuple[int, int, int] = (248, 250, 252),
    accent: tuple[int, int, int] = (15, 118, 110),
    fg: tuple[int, int, int] = (15, 23, 42),
) -> RenderResult:
    """Render a heading and wrapped explanatory body as a light slide.

    Args:
        output_path: PNG destination.
        width: Output width in pixels.
        height: Output height in pixels.
        heading: Main visible heading.
        body: Explanatory text, limited to the first eight wrapped lines.
        bg_color: RGB background color before the shared fill.
        accent: RGB color for the accent bar.
        fg: RGB color for heading text.

    Returns:
        ``RenderResult`` for the written PNG.

    """
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    _fill_background(img, draw)
    # accent bar
    draw.rectangle([(80, 140), (260, 200)], fill=accent)
    # heading
    h_font = _pick_font(88)
    h_lines = _wrap_lines(draw, heading or "", h_font, width - 320)
    y = 220
    for line in h_lines[:3]:
        draw.text((80, y), line, fill=fg, font=h_font)
        y += 110
    # body
    b_font = _pick_font(56)
    b_lines = _wrap_lines(draw, body or "", b_font, width - 320)
    yy = y + 30
    for line in b_lines[:8]:
        draw.text((120, yy), line, fill=(51, 65, 85), font=b_font)
        yy += 80
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return RenderResult(output_path=output_path, width=width, height=height)


def render_title_slide(
    output_path: Path,
    *,
    width: int,
    height: int,
    title: str,
    subtitle: str,
    chapter_label: str | None = None,
) -> RenderResult:
    """Render a chapter/title slide with an optional label.

    Args:
        output_path: PNG destination.
        width: Output width in pixels.
        height: Output height in pixels.
        title: Main title text.
        subtitle: Optional subtitle below the title.
        chapter_label: Optional small label above the title.

    Returns:
        ``RenderResult`` for the written PNG.

    """
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    _fill_background(img, draw)
    # accent stripe
    draw.rectangle([(0, 0), (width, 24)], fill=(15, 118, 110))
    if chapter_label:
        cfont = _pick_font(48)
        draw.text((120, 200), chapter_label, fill=(100, 116, 139), font=cfont)
    hfont = _pick_font(120)
    lines = _wrap_lines(draw, title, hfont, width - 240)
    y = 360
    for line in lines[:3]:
        draw.text((120, y), line, fill=(15, 23, 42), font=hfont)
        y += 150
    if subtitle:
        sfont = _pick_font(56)
        slines = _wrap_lines(draw, subtitle, sfont, width - 240)
        for line in slines[:2]:
            draw.text((120, y + 30), line, fill=(71, 85, 105), font=sfont)
            y += 70
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return RenderResult(output_path=output_path, width=width, height=height)


def render_comparison(
    output_path: Path,
    *,
    width: int,
    height: int,
    heading: str,
    left: dict[str, str],
    right: dict[str, str],
) -> RenderResult:
    """Render two labeled panels for a comparison plan.

    Args:
        output_path: PNG destination.
        width: Output width in pixels.
        height: Output height in pixels.
        heading: Comparison heading.
        left: Left panel mapping with optional ``title`` and ``content``.
        right: Right panel mapping with optional ``title`` and ``content``.

    Returns:
        ``RenderResult`` for the written PNG.

    """
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    _fill_background(img, draw)
    # heading
    hfont = _pick_font(72)
    lines = _wrap_lines(draw, heading or "比較", hfont, width - 240)
    draw.text((120, 80), lines[0] if lines else "", fill=(15, 23, 42), font=hfont)
    panel_y = 240
    panel_h = height - 320
    margin = 80
    gap = 80
    panel_w = (width - margin * 2 - gap) // 2
    # left panel
    draw.rectangle([(margin, panel_y), (margin + panel_w, panel_y + panel_h)], outline=(15, 118, 110), width=4)
    # right panel
    draw.rectangle(
        [(margin + panel_w + gap, panel_y), (margin + panel_w * 2 + gap, panel_y + panel_h)],
        outline=(15, 118, 110),
        width=4,
    )
    tfont = _pick_font(56)
    cfont = _pick_font(46)
    draw.text((margin + 24, panel_y + 24), left.get("title", ""), fill=(15, 23, 42), font=tfont)
    draw.text(
        (margin + panel_w + gap + 24, panel_y + 24),
        right.get("title", ""),
        fill=(15, 23, 42),
        font=tfont,
    )
    body_y = panel_y + 24 + 80
    # left content
    for line in _wrap_lines(draw, left.get("content", ""), cfont, panel_w - 48)[:8]:
        draw.text((margin + 24, body_y), line, fill=(51, 65, 85), font=cfont)
        body_y += 64
    body_y = panel_y + 24 + 80
    for line in _wrap_lines(draw, right.get("content", ""), cfont, panel_w - 48)[:8]:
        draw.text(
            (margin + panel_w + gap + 24, body_y),
            line,
            fill=(51, 65, 85),
            font=cfont,
        )
        body_y += 64
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return RenderResult(output_path=output_path, width=width, height=height)


def render_code_slide(
    output_path: Path,
    *,
    width: int,
    height: int,
    heading: str,
    code: str,
    language: str = "text",
) -> RenderResult:
    """Render syntax-neutral code with line numbers and a language tag.

    Args:
        output_path: PNG destination.
        width: Output width in pixels.
        height: Output height in pixels.
        heading: Visible code-slide heading.
        code: Source text to display; long output is clipped to the canvas.
        language: Optional uppercase language badge.

    Returns:
        ``RenderResult`` for the written PNG.

    """
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    # heading
    hfont = _pick_font(64)
    draw.text((80, 60), heading or "コード", fill=(226, 232, 240), font=hfont)
    # language tag
    if language:
        tag = f"  {language.upper()}  "
        lfont = _pick_font(36)
        bbox = draw.textbbox((0, 0), tag, font=lfont)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle(
            [(width - tw - 80 - 30, 60), (width - 80, 60 + th + 24)],
            fill=(15, 118, 110),
        )
        draw.text((width - tw - 80 - 12, 60 + 12), tag.strip(), fill=(255, 255, 255), font=lfont)
    # code block
    code_font = _pick_font(46)
    code = code or ""
    code_area_y = 200
    code_area_x = 80
    code_area_w = width - 160
    line_height = 64
    max_lines = (height - code_area_y - 80) // line_height
    lines = code.splitlines() or [""]
    # line numbers gutter
    n_lines = min(len(lines), max_lines)
    gfont = _pick_font(36)
    draw.text((code_area_x, code_area_y), "  1", fill=(148, 163, 184), font=gfont)
    _wrap_lines(draw, lines[0] if lines else "", code_font, code_area_w - 100)
    y = code_area_y
    drawn = 0
    line_no = 1
    for i in range(n_lines):
        line_text = lines[i]
        wrapped = _wrap_lines(draw, line_text, code_font, code_area_w - 100)
        for wline in wrapped:
            if drawn >= max_lines:
                break
            # line number
            draw.text(
                (code_area_x, y),
                f"{line_no:>3}",
                fill=(100, 116, 139),
                font=gfont,
            )
            # code
            draw.text(
                (code_area_x + 100, y),
                wline,
                fill=(226, 232, 240),
                font=code_font,
            )
            y += line_height
            drawn += 1
        line_no += 1
        if drawn >= max_lines:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return RenderResult(output_path=output_path, width=width, height=height)


def render_diagram(
    output_path: Path,
    *,
    width: int,
    height: int,
    heading: str,
    diagram_source: str,
) -> RenderResult:
    """Render Mermaid source with CLI-first and PIL fallback behavior.

    Primary path: invoke ``mmdc`` (mermaid-cli) to render the Mermaid source
    faithfully — branching, trees, labeled edges, subgraphs and all. This is
    what the LLM is asked to produce.

    Fallback: if mmdc is unavailable or fails, fall back to the built-in PIL
    renderer which only understands a tiny subset of Mermaid (vertical chain
    of ``[..]`` boxes with ``-->`` edges). That keeps the pipeline working
    without a Node toolchain, at the cost of diagram fidelity.

    Args:
        output_path: PNG destination.
        width: Exact output width in pixels.
        height: Exact output height in pixels.
        heading: Visible heading.
        diagram_source: Mermaid source; empty/unsupported input is shown as
            text in the fallback path.

    Returns:
        ``RenderResult`` for either the Mermaid or fallback PNG.

    Side Effects:
        May create a temporary PNG, invoke ``mmdc``/``npx``, log a warning on
        failure, and write the final PNG.

    """
    source = (diagram_source or "").strip()
    if source and mmdc_available():
        import tempfile

        # mmdc sizes the canvas to the graph, so -w/-H only cap it. Render
        # oversized, then compose onto the real frame: that keeps the result
        # crisp instead of blur-upscaling a 613x181 PNG to fill 1920x880.
        tmp_png = Path(tempfile.mkstemp(suffix=".png")[1])
        try:
            render_mermaid_to_png_sync(
                source,
                tmp_png,
                width=max(width, 2400),
                height=max(height, 1400),
                theme="default",
                background="transparent",
            )
            with Image.open(tmp_png) as raw:
                content = raw.convert("RGBA")
                canvas = compose_on_canvas(
                    content, width=width, height=height, title=heading or "図"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, format="PNG")
            return RenderResult(output_path=output_path, width=width, height=height)
        except Exception as exc:  # noqa: BLE001 - any failure -> PIL fallback
            log.warning(
                "mermaid レンダリング失敗、PIL フォールバックに切り替えます: {err}",
                err=exc,
            )
        finally:
            try:
                tmp_png.unlink(missing_ok=True)
            except OSError:
                pass

    # ----- Fallback: built-in PIL renderer (subset of Mermaid) -----
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    _fill_background(img, draw)
    hfont = _pick_font(60)
    draw.text((80, 80), heading or "図", fill=(15, 23, 42), font=hfont)
    boxes, edges = _parse_simple_mermaid(diagram_source or "")
    if boxes:
        _draw_boxes_and_edges(img, draw, boxes, edges)
    else:
        # fall back to monospaced text rendering
        f = _pick_font(36)
        for i, line in enumerate((diagram_source or "").splitlines()[:18]):
            draw.text((80, 220 + i * 50), line, fill=(51, 65, 85), font=f)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return RenderResult(output_path=output_path, width=width, height=height)


def _embed_heading(mermaid_source: str, heading: str) -> str:
    """Prepend a Mermaid ``title``/``accTitle`` to the rendered image.

    If the source already
    starts with a graph/flowchart directive, inject the title line after it.
    """
    lines = mermaid_source.splitlines()
    if lines and lines[0].strip().lower().startswith(("graph", "flowchart")):
        return lines[0] + "\n" + f"    accTitle: {heading}\n" + "\n".join(lines[1:])
    return f"flowchart TD\n    accTitle: {heading}\n" + mermaid_source


_NODE_RX = re.compile(r"([A-Za-z0-9_]+)\s*\[([^\]]+)\]")
_EDGE_RX = re.compile(r"([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)")


def _parse_simple_mermaid(source: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Extract simple box nodes and arrows for the no-mmdc fallback renderer."""
    boxes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for raw in source.splitlines():
        line = raw.strip()
        if line.startswith(("graph", "flowchart", "subgraph", "end", "%%")):
            continue
        for m in _NODE_RX.finditer(line):
            boxes.setdefault(m.group(1), m.group(2))
        for m in _EDGE_RX.finditer(line):
            edges.append((m.group(1), m.group(2)))
            boxes.setdefault(m.group(1), m.group(1))
            boxes.setdefault(m.group(2), m.group(2))
    return boxes, edges


def _draw_boxes_and_edges(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    boxes: dict[str, str],
    edges: list[tuple[str, str]],
) -> None:
    """Draw the small vertical graph subset understood by the PIL fallback."""
    if not boxes:
        return
    keys = list(boxes.keys())
    n = len(keys)
    box_w = 360
    box_h = 100
    y0 = 280
    gap = 140
    positions: dict[str, tuple[int, int]] = {}
    # layout: simple vertical chain if there are edges, else grid.
    if edges:
        # Build a topological-ish chain by following edges.
        children: dict[str, list[str]] = {k: [] for k in keys}
        for a, b in edges:
            children[a].append(b)
        # roots = nodes with no incoming edges
        incoming = {k: 0 for k in keys}
        for a, b in edges:
            incoming[b] += 1
        roots = [k for k in keys if incoming[k] == 0]
        chain: list[str] = []
        visited = set()
        for r in roots or keys:
            stack = [r]
            while stack and len(chain) < n:
                cur = stack.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                chain.append(cur)
                stack.extend(children.get(cur, []))
        # append any leftovers
        for k in keys:
            if k not in visited:
                chain.append(k)
    else:
        chain = keys
    f = _pick_font(40)
    for i, k in enumerate(chain):
        cx = (img.width - box_w) // 2
        cy = y0 + i * (box_h + gap)
        if cy + box_h > img.height - 60:
            break
        draw.rectangle(
            [(cx, cy), (cx + box_w, cy + box_h)],
            fill=(255, 255, 255),
            outline=(15, 118, 110),
            width=4,
        )
        label = boxes.get(k, k)
        if len(label) > 18:
            label = label[:17] + "…"
        bbox = draw.textbbox((0, 0), label, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (cx + (box_w - tw) // 2, cy + (box_h - th) // 2 - 4),
            label,
            fill=(15, 23, 42),
            font=f,
        )
        positions[k] = (cx, cy)
    # draw edges (straight lines + arrowhead)
    for a, b in edges:
        if a not in positions or b not in positions:
            continue
        ax, ay = positions[a][0] + box_w // 2, positions[a][1] + box_h
        bx, by = positions[b][0] + box_w // 2, positions[b][1]
        draw.line([(ax, ay), (bx, by)], fill=(15, 118, 110), width=4)
        # arrow head
        draw.polygon(
            [(bx - 10, by - 20), (bx + 10, by - 20), (bx, by)],
            fill=(15, 118, 110),
        )


def render_formula(
    output_path: Path,
    *,
    width: int,
    height: int,
    heading: str,
    formula: str,
    note: str = "",
) -> RenderResult:
    """Render a formula panel as styled PIL text.

    For the MVP we don't bundle a TeX engine; we render formula as styled
    text using PIL with italics and a serif font fallback. (We tried to
    render KaTeX-equivalent via SVG here, but PIL-only rendering keeps the
    MVP dependency-free.)

    Args:
        output_path: PNG destination.
        width: Output width in pixels.
        height: Output height in pixels.
        heading: Formula heading.
        formula: Formula text, clipped after 30 characters.
        note: Optional explanatory text below the formula.

    Returns:
        ``RenderResult`` for the written PNG.

    """
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    _fill_background(img, draw)
    hfont = _pick_font(64)
    draw.text((80, 100), heading or "数式", fill=(15, 23, 42), font=hfont)
    # formula
    ffont = _pick_font(120)
    label = formula or ""
    if len(label) > 30:
        label = label[:29] + "…"
    bbox = draw.textbbox((0, 0), label, font=ffont)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (width - tw) // 2
    cy = (height - th) // 2
    draw.rectangle(
        [(cx - 60, cy - 40), (cx + tw + 60, cy + th + 40)],
        outline=(15, 118, 110),
        width=4,
    )
    draw.text((cx, cy), label, fill=(15, 23, 42), font=ffont)
    if note:
        nfont = _pick_font(40)
        lines = _wrap_lines(draw, note, nfont, width - 200)
        y = cy + th + 80
        for line in lines[:3]:
            draw.text((120, y), line, fill=(71, 85, 105), font=nfont)
            y += 56
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return RenderResult(output_path=output_path, width=width, height=height)


def render_visual_plan(
    plan_payload: dict[str, Any],
    output_path: Path,
    *,
    width: int,
    height: int,
    fallback_summary: str = "",
) -> RenderResult:
    """Dispatch a validated-ish plan payload to its renderer.

    Args:
        plan_payload: Mapping with ``visual_type`` and renderer-specific fields.
        output_path: PNG destination.
        width: Exact output width in pixels.
        height: Exact output height in pixels.
        fallback_summary: Text used when a plan omits visible summary/body.

    Returns:
        The selected renderer's ``RenderResult``.  Invalid/missing visual types
        fall back to a readable text slide rather than raising where possible;
        structured diagram failures also degrade to text slides.

    """
    vtype = plan_payload.get("visual_type") or VisualType.text_slide.value
    heading = plan_payload.get("heading") or ""
    if vtype == VisualType.title_slide.value:
        return render_title_slide(
            output_path,
            width=width,
            height=height,
            title=heading,
            subtitle=plan_payload.get("visual_summary") or fallback_summary,
        )
    if vtype == VisualType.verbatim_slide.value:
        # Author-drawn content: laid out on a fixed character grid so the
        # box-drawing lines the author aligned stay aligned.
        render_verbatim_slide(
            plan_payload.get("verbatim") or "",
            output_path,
            width=width,
            height=height,
            title=heading,
        )
        return RenderResult(output_path=output_path, width=width, height=height)
    if vtype == VisualType.code_slide.value:
        return render_code_slide(
            output_path,
            width=width,
            height=height,
            heading=heading or "コード",
            code=plan_payload.get("code") or "",
            language=plan_payload.get("language") or "text",
        )
    if vtype == VisualType.formula.value:
        return render_formula(
            output_path,
            width=width,
            height=height,
            heading=heading or "数式",
            formula=plan_payload.get("formula") or "",
            note=plan_payload.get("visual_summary") or "",
        )
    if vtype == VisualType.pointer_diagram.value:
        spec = dict(plan_payload.get("pointer_diagram") or {})
        spec.setdefault("title", heading)
        try:
            res = render_pointer_diagram(spec, output_path, width=width, height=height)
            return RenderResult(res.output_path, res.width, res.height)
        except Exception as exc:  # noqa: BLE001 - degrade to a readable slide
            log.warning("pointer_diagram 描画失敗、text_slide に降格: {err}", err=exc)
            return render_text_slide(
                output_path, width=width, height=height,
                heading=heading or "構造",
                body=plan_payload.get("visual_summary") or fallback_summary,
            )
    if vtype == VisualType.env_diagram.value:
        spec = dict(plan_payload.get("env_diagram") or {})
        spec.setdefault("title", heading)
        try:
            res = render_env_diagram(spec, output_path, width=width, height=height)
            return RenderResult(res.output_path, res.width, res.height)
        except Exception as exc:  # noqa: BLE001 - degrade to a readable slide
            log.warning("env_diagram 描画失敗、text_slide に降格: {err}", err=exc)
            return render_text_slide(
                output_path, width=width, height=height,
                heading=heading or "環境モデル",
                body=plan_payload.get("visual_summary") or fallback_summary,
            )
    if vtype == VisualType.diagram.value:
        return render_diagram(
            output_path,
            width=width,
            height=height,
            heading=heading or "図",
            diagram_source=plan_payload.get("diagram") or "",
        )
    if vtype == VisualType.comparison.value:
        return render_comparison(
            output_path,
            width=width,
            height=height,
            heading=heading or "比較",
            left=plan_payload.get("left_panel") or {"title": "左", "content": ""},
            right=plan_payload.get("right_panel") or {"title": "右", "content": ""},
        )
    # default text slide
    return render_text_slide(
        output_path,
        width=width,
        height=height,
        heading=heading or "要点",
        body=plan_payload.get("visual_summary") or fallback_summary,
    )

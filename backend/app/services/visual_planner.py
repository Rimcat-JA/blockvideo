"""Visual plan generation.

The planner decides, per block, what should be on screen. The important
judgement it makes is *which* visual form fits the narration:

- Cons cells / list structure / ``set-cdr!`` / circular lists become a
  ``pointer_diagram`` — a structured spec the renderer draws as a real
  SICP box-and-pointer diagram. Mermaid flowcharts cannot express these.
- Environment frames, bindings and procedure objects become an
  ``env_diagram``.
- Code examples become a ``code_slide``. Code set as a *diagram* reads badly
  and was a recurring failure mode, so both the prompt and a server-side
  guard steer it back to a slide.
- Only genuine process/branch flows stay as Mermaid ``diagram``.
"""
from __future__ import annotations

import json
import re
import unicodedata

from pydantic import ValidationError

from app.core.logging import log
from app.models.block import VisualType
from app.providers.llm import LLMProvider, LLMRequest, LLMMessage
from app.services.stage_schemas import GlobalVisualStylePayload, VisualPlan, VisualPlanPayload


VISUAL_PLAN_SYSTEM = (
    "あなたは日本語の計算機科学教材（SICP系）の映像ディレクター兼作図者です。"
    "ナレーション1ブロックに対して、理解を最も助ける画面を1つだけ設計し、"
    "JSON のみで返してください。図は曖昧な絵ではなく、"
    "描画エンジンがそのまま作図できる構造化データとして出力します。"
)


_SLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["value", "ref", "nil"]},
        "text": {"type": ["string", "null"]},
        "target": {"type": ["string", "null"]},
    },
    "required": ["kind"],
    "additionalProperties": False,
}

POINTER_DIAGRAM_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "title": {"type": ["string", "null"]},
        "caption": {"type": ["string", "null"]},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": ["string", "null"]},
                    "roots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "target": {"type": "string"},
                            },
                            "required": ["label", "target"],
                            "additionalProperties": False,
                        },
                    },
                    "cells": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "car": _SLOT_SCHEMA,
                                "cdr": _SLOT_SCHEMA,
                            },
                            "required": ["id", "car", "cdr"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["cells"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}

# Anthropic's structured outputs reject any 'object' that does not set
# additionalProperties explicitly to false, so the comparison panels need a
# declared shape rather than a bare {"type": "object"}.
PANEL_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "title": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["title", "content"],
    "additionalProperties": False,
}

ENV_DIAGRAM_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "title": {"type": ["string", "null"]},
        "caption": {"type": ["string", "null"]},
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "parent": {"type": ["string", "null"]},
                    "bindings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["name", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "label", "bindings"],
                "additionalProperties": False,
            },
        },
        "procedures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "params": {"type": ["string", "null"]},
                    "body": {"type": ["string", "null"]},
                    "env": {"type": ["string", "null"]},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["frames"],
    "additionalProperties": False,
}


async def generate_global_style(
    provider: LLMProvider, *, project_title: str
) -> str:
    """Ask the LLM for one style description shared by all project blocks."""
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=VISUAL_PLAN_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"動画タイトル: {project_title}\n"
                    "全ブロックで統一する global_visual_style を1つ定義してください。"
                    "落ち着いた大学講義風で、16:9、白〜薄グレー背景、"
                    "青緑をアクセント、等幅フォントでコード、過剰装飾なし、"
                    "高い可読性、日本語の教材資料風。"
                    "出力は JSON のみ: {\"global_visual_style\": \"...\"}"
                ),
            ),
        ],
        temperature=0.2,
        max_tokens=800,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "global_style",
                "schema": {
                    "type": "object",
                    "properties": {"global_visual_style": {"type": "string"}},
                    "required": ["global_visual_style"],
                    "additionalProperties": False,
                },
            },
        },
    )
    data = await provider.chat_json(request)
    return GlobalVisualStylePayload.model_validate(data).global_visual_style


async def generate_title(provider: LLMProvider, *, script: str) -> str:
    """Derive a short video title from the script (Quick Generate flow)."""
    excerpt = script.strip()[:3000]
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content="あなたは日本語教材動画の編集者です。JSON のみで返します。",
            ),
            LLMMessage(
                role="user",
                content=(
                    "次の台本にふさわしい動画タイトルを1つ考えてください。"
                    "40文字以内、日本語、内容を具体的に表すもの。記号での装飾は不要。\n\n"
                    f"```text\n{excerpt}\n```\n\n"
                    '出力は JSON のみ: {"title": "..."}'
                ),
            ),
        ],
        temperature=0.3,
        max_tokens=300,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "title",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
        },
    )
    data = await provider.chat_json(request)
    title = str((data or {}).get("title") or "").strip()
    if not title:
        raise RuntimeError("title generation returned an empty title")
    return title[:255]


_CODE_MARKERS = re.compile(
    r"\(define\s|\(lambda\s|\(let\s|\(cons\s|\(set-cdr!|\(set-car!|\(if\s|"
    r"\bdef\s+\w+\(|\bfunction\s+\w+\(|=>|;;|\breturn\b|\bclass\s+\w+"
)


def _looks_like_code(text: str) -> bool:
    """Return whether a 'diagram' body is really a code listing.

    Mermaid sources always start with a graph directive; anything that opens
    with S-expressions or statements is code the model mislabelled.
    """
    body = (text or "").strip()
    if not body:
        return False
    first = body.splitlines()[0].strip().lower()
    if first.startswith(("graph", "flowchart", "sequencediagram", "classdiagram",
                         "statediagram", "erdiagram", "journey", "gantt", "pie",
                         "mindmap", "timeline")):
        return False
    return bool(_CODE_MARKERS.search(body))


def _normalize_plan_payload(data: object) -> object:
    """Repair the two shapes models reach for instead of the schema.

    ``diagram`` is the Mermaid *source string*; the structured pointer/env
    specs come from a separate call. Models that pick ``pointer_diagram``
    frequently write the spec object straight into ``diagram`` — or hang it
    off the top level next to ``plan`` — which fails validation and costs the
    block its visual. Both are unambiguous, so move them rather than retry.
    """
    if not isinstance(data, dict):
        return data
    plan = data.get("plan")
    if not isinstance(plan, dict):
        return data

    # A structured spec parked in the Mermaid string field.
    spec = plan.get("diagram")
    if isinstance(spec, dict):
        kind = plan.get("visual_type")
        if kind in ("pointer_diagram", "env_diagram") and not plan.get(kind):
            plan[kind] = spec
        plan["diagram"] = None

    # Fields hung off the top level instead of inside ``plan``.
    for key in ("diagram", "pointer_diagram", "env_diagram"):
        stray = data.pop(key, None)
        if isinstance(stray, dict) and key != "diagram" and not plan.get(key):
            plan[key] = stray

    return {"plan": plan}


_FENCE_RX = re.compile(r"```([A-Za-z0-9_+-]*)\n?([\s\S]*?)```")


# ```slide 見出し  …author-drawn slide content…  ```
# The heading runs to end of line so it can be Japanese, which the generic
# fence regex (info strings are ASCII identifiers) would swallow into the body.
_SLIDE_FENCE_RX = re.compile(r"```slide[ \t]*([^\n]*)\n([\s\S]*?)```")


def extract_authored_slide(source_text: str) -> tuple[str, str] | None:
    """Return ``(heading, body)`` for a slide the script author drew, if any.

    A block carrying one of these needs no visual planning: the author has
    already said exactly what the slide shows, so the model is not asked to
    design anything and cannot invent labels the narration never mentions.
    """
    match = _SLIDE_FENCE_RX.search(source_text or "")
    if not match:
        return None
    body = (match.group(2) or "").strip("\n")
    if not body.strip():
        return None
    return (match.group(1) or "").strip(), body


_BORDER_CHARS = "─━┌┏┐┓└┗┘┛├┣┤┫┬┳┴┻┼╋═╔╗╚╝"


def _cell_width(line: str) -> int:
    """Return display width using two columns for full-width East Asian glyphs."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in line)


def slide_alignment_issues(body: str) -> list[str]:
    """Lines whose box does not close where its border says it should.

    Models write ``│ table │`` under a ``┌───────────┐`` and miscount the
    padding, so the right-hand wall lands inside the box. The renderer is
    faithful and draws exactly that, which makes a script-side mistake look
    like a rendering bug — worth naming rather than guessing a repair, since
    padding it automatically would be inventing layout the author did not ask
    for.
    """
    lines = (body or "").splitlines()
    borders = [
        i for i, l in enumerate(lines)
        if any(c in _BORDER_CHARS for c in l) and "─" in l
    ]
    issues: list[str] = []
    for top, bottom in zip(borders, borders[1:]):
        width = _cell_width(lines[top].rstrip())
        if width != _cell_width(lines[bottom].rstrip()):
            continue  # not a matched pair of walls
        for i in range(top + 1, bottom):
            inner = lines[i].rstrip()
            if "│" in inner and _cell_width(inner) != width:
                issues.append(
                    f"{i + 1}行目 幅{_cell_width(inner)} (枠線は{width}): {inner.strip()[:40]}"
                )
    return issues


def authored_plan(heading: str, body: str) -> VisualPlan:
    """Build an author-drawn slide plan locally without an LLM call."""
    return VisualPlan(
        visual_type=VisualType.verbatim_slide,
        heading=heading or None,
        verbatim=body,
    )


def extract_code_block(source_text: str) -> tuple[str, str] | None:
    """Return ``(code, language)`` for the first real code fence, if any.

    Fences holding hand-drawn ASCII diagrams (```` ```text ```` with box
    characters) are not code and must not become code slides — those belong
    to the structured diagram renderers.
    """
    for match in _FENCE_RX.finditer(source_text or ""):
        lang = (match.group(1) or "").strip().lower()
        body = (match.group(2) or "").strip()
        if not body:
            continue
        if _looks_like_code(body):
            return body, lang or "text"
    return None


def extract_all_fences(source_text: str) -> list[tuple[str, str]]:
    """Every fenced block in document order, as ``(body, language)``.

    Includes hand-drawn ``text`` fences: they were drawn to be looked at, and
    a monospaced slide is a faithful way to show them. Used to build the
    extra slides a block cycles through, so a script that packs several
    listings between two sentences does not lose all but the first.
    """
    out: list[tuple[str, str]] = []
    for match in _FENCE_RX.finditer(source_text or ""):
        lang = (match.group(1) or "").strip().lower()
        body = (match.group(2) or "").strip()
        if body:
            out.append((body, lang or "text"))
    return out


def build_slide_sequence(
    plan_payload: dict, source_text: str, *, max_slides: int = 6
) -> list[dict]:
    """Plans for every slide a block should show, in order.

    The planner describes the block's narration with one visual; the script
    may additionally contain listings and diagrams that deserve screen time.
    This pairs the planner's choice with the leftover fences without spending
    another LLM call on each.
    """
    primary = dict(plan_payload or {})
    sequence = [primary]

    if primary.get("visual_type") == VisualType.verbatim_slide.value:
        # The author said what this block shows. Appending the block's other
        # fences would contradict that, and the ```slide fence itself would
        # come back as a duplicate code slide.
        return sequence

    primary_code = (primary.get("code") or "").strip()
    for body, lang in extract_all_fences(source_text):
        if len(sequence) >= max_slides:
            break
        if body == primary_code:
            continue  # already on the primary slide
        sequence.append(
            {
                "visual_type": "code_slide",
                "heading": primary.get("heading") or "",
                "code": body,
                "language": lang,
            }
        )
    return sequence


def _enforce_visual_type(
    plan: VisualPlan, *, block_index: int, source_text: str = ""
) -> VisualPlan:
    """Server-side guard for the choices models most often get wrong."""
    vt = plan.visual_type.value
    if vt == VisualType.verbatim_slide.value:
        return plan  # authored, not chosen — nothing to second-guess

    # A block that carries an actual code listing should show it. Models often
    # prefer a diagram because the narration around the code is conceptual,
    # which leaves the listing invisible even though the script included it.
    code_block = extract_code_block(source_text)
    if code_block and vt not in ("code_slide", "formula"):
        code, lang = code_block
        log.info(
            "block={idx} コードブロックを検出したため {old} -> code_slide",
            idx=block_index, old=vt,
        )
        plan.code = code
        plan.language = plan.language or lang
        plan.visual_type = type(plan.visual_type).code_slide
        vt = "code_slide"

    # A diagram whose body is code belongs on a code slide.
    if vt == "diagram" and _looks_like_code(plan.diagram or ""):
        log.info(
            "block={idx} diagram本文がコードのため code_slide に変更", idx=block_index
        )
        plan.code = plan.code or plan.diagram
        plan.diagram = None
        plan.visual_type = type(plan.visual_type).code_slide
        vt = "code_slide"

    # Structured diagram types must carry their spec; otherwise downgrade so
    # the renderer never receives an empty diagram.
    if vt == "pointer_diagram" and not (plan.pointer_diagram or {}).get("groups"):
        log.warning(
            "block={idx} pointer_diagram に groups が無いため text_slide に降格",
            idx=block_index,
        )
        plan.visual_type = type(plan.visual_type).text_slide
    elif vt == "env_diagram" and not (plan.env_diagram or {}).get("frames"):
        log.warning(
            "block={idx} env_diagram に frames が無いため text_slide に降格",
            idx=block_index,
        )
        plan.visual_type = type(plan.visual_type).text_slide
    elif vt == "code_slide" and not (plan.code or "").strip():
        plan.visual_type = type(plan.visual_type).text_slide

    return plan


_USER_PROMPT = """\
global_visual_style: {style}
ブロック番号: {index}
tts_text: 「{tts}」
source_text: 「{source}」

このナレーションに最も適した画面を1つ設計してください。

## visual_type の選び方（重要・上から順に判定）

1. **pointer_diagram** — cons セル・ペア・リスト構造・car/cdr・ポインタ・
   `set-cdr!` / `set-car!` による破壊的変更・循環リスト・共有構造の話題。
   SICP の「箱とポインタ図」をそのまま構造化データで出す。
   - `groups` は1〜3個。状態変化を語るなら「変更前」「変更後」で2つに分ける。
   - `cells[].id` は一意。`car`/`cdr` は
     `{{"kind":"value","text":"a"}}` / `{{"kind":"ref","target":"c2"}}` /
     `{{"kind":"nil"}}` のいずれか。
   - リスト終端は必ず `{{"kind":"nil"}}`。循環は最初のセルへの `ref`。
   - 変数名から出る矢印は `roots`（例 `{{"label":"x","target":"c1"}}`）。

2. **env_diagram** — 環境・フレーム・束縛・スコープ・`define` による変数束縛・
   手続きオブジェクト・親環境へのポインタの話題。
   - `frames[].parent` で親環境を指す。グローバル環境の parent は null。
   - 手続きオブジェクトは `procedures`（`params` / `body` / `env` を埋める）。

3. **code_slide** — コード例そのものを見せたいとき。**コードを図にしてはいけない。**
   `code` にコード本文、`language` に `scheme` などを入れる。

4. **diagram** — 上記に当てはまらず、純粋な手順・分岐・状態遷移を示すときのみ。
   Mermaid 構文（先頭行は `flowchart TD` か `flowchart LR`）。
   データ構造やメモリ上の参照関係にこれを使わないこと。

## この応答で図の中身は書かないこと
`pointer_diagram` / `env_diagram` を選んだ場合、この応答では `visual_type` と
`heading` を返すだけでよく、図の構造は次の質問で設計します。
`diagram` フィールドは **Mermaid の文字列専用** です。ここにセルや
フレームのオブジェクトを入れないでください（null のままにする）。

5. **formula** / **comparison** / **title_slide** / **text_slide** — 上記以外。

## 共通ルール
- `heading` は必ず短い日本語で入れる。
- ラベルは短く。セルの car は1〜6文字程度に収める。
- `ai_image` は使わない。
- 出力は JSON のみ: {{"plan": {{...}}}}
"""


_DIAGRAM_PROMPTS = {
    "pointer_diagram": (
        "次のナレーションを SICP の「箱とポインタ図」として設計してください。\n"
        "- `groups` は1〜3個。状態変化を語る内容なら「変更前」「変更後」で分ける。\n"
        "- `cells[].id` は一意。`car` / `cdr` は次のいずれか:\n"
        "  `{{\"kind\":\"value\",\"text\":\"a\"}}` / `{{\"kind\":\"ref\",\"target\":\"c2\"}}` / `{{\"kind\":\"nil\"}}`\n"
        "- リスト終端は必ず `{{\"kind\":\"nil\"}}`。循環は前のセルへの `ref`。\n"
        "- 変数名から出る矢印は `roots`（例 `{{\"label\":\"x\",\"target\":\"c1\"}}`）。\n"
        "- セルの car ラベルは1〜6文字程度に収める。\n"
    ),
    "env_diagram": (
        "次のナレーションを SICP の「環境モデル図」として設計してください。\n"
        "- `frames[].parent` で親環境を指す。グローバル環境の parent は null。\n"
        "- 各フレームの `bindings` に変数名と値を入れる。\n"
        "- 手続きオブジェクトは `procedures` に `params` / `body` / `env` を埋める。\n"
        "- ラベルは短く。\n"
    ),
}


async def _design_diagram(
    provider: LLMProvider,
    *,
    kind: str,
    schema: dict,
    block_index: int,
    tts_text: str,
    source_text: str,
    heading: str,
) -> dict | None:
    """Second planning call: design one structured diagram.

    Kept separate from the visual-type choice because Anthropic's constrained
    decoding rejects the combined schema outright ("the compiled grammar is
    too large"). Splitting also lets the diagram prompt be specific instead of
    one clause inside a menu, which measurably improves the layouts.
    """
    # The spec is nullable at the top level for the combined schema; here it
    # is the whole response, so require the object.
    body = {k: v for k, v in schema.items() if k != "type"}
    body["type"] = "object"

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=VISUAL_PLAN_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"{_DIAGRAM_PROMPTS[kind]}\n"
                    f"見出し: {heading}\n"
                    f"ナレーション: 「{tts_text}」\n"
                    f"原文: 「{source_text}」\n\n"
                    "出力は JSON のみ。"
                ),
            ),
        ],
        temperature=0.2,
        max_tokens=4000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": kind, "schema": body},
        },
    )
    for attempt in (1, 2):
        try:
            data = await provider.chat_json(request)
        except Exception as exc:  # noqa: BLE001 - downgraded by the caller's guard
            log.warning(
                "block={idx} {kind} の設計に失敗 (attempt {n}): {err}",
                idx=block_index, kind=kind, n=attempt, err=exc,
            )
            continue
        if isinstance(data, dict):
            return data
    return None


async def generate_visual_plan(
    provider: LLMProvider,
    *,
    block_index: int,
    tts_text: str,
    source_text: str,
    global_style: str,
) -> VisualPlan:
    """Choose the visual for one block, then design it if it is a diagram."""
    schema = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "properties": {
                    "visual_type": {
                        "type": "string",
                        "enum": [
                            "code_slide",
                            "pointer_diagram",
                            "env_diagram",
                            "diagram",
                            "formula",
                            "comparison",
                            "title_slide",
                            "text_slide",
                        ],
                    },
                    "heading": {"type": ["string", "null"]},
                    "visual_summary": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                    "formula": {"type": ["string", "null"]},
                    "diagram": {"type": ["string", "null"]},
                    "left_panel": PANEL_SCHEMA,
                    "right_panel": PANEL_SCHEMA,
                },
                "required": ["visual_type", "heading"],
                "additionalProperties": False,
            }
        },
        "required": ["plan"],
        "additionalProperties": False,
    }
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=VISUAL_PLAN_SYSTEM),
            LLMMessage(
                role="user",
                content=_USER_PROMPT.format(
                    style=global_style,
                    index=block_index,
                    tts=tts_text,
                    source=source_text,
                ),
            ),
        ],
        temperature=0.3,
        max_tokens=3000,
        response_format={"type": "json_schema", "json_schema": {"name": "plan", "schema": schema}},
    )
    # One retry: malformed JSON from the planner is occasional and transient,
    # and without a retry a single blip silently costs that block its diagram.
    payload = None
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            data = await provider.chat_json(request)
            payload = VisualPlanPayload.model_validate(_normalize_plan_payload(data))
            break
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            last_exc = exc
            log.warning(
                "block={idx} visual plan 検証失敗 (attempt {n}): {err}",
                idx=block_index, n=attempt, err=exc,
            )
    if payload is None:
        raise RuntimeError(f"visual plan 検証失敗: {last_exc}") from last_exc

    plan = payload.plan
    kind = plan.visual_type.value
    if kind in ("pointer_diagram", "env_diagram"):
        spec = await _design_diagram(
            provider,
            kind=kind,
            schema=POINTER_DIAGRAM_SCHEMA if kind == "pointer_diagram" else ENV_DIAGRAM_SCHEMA,
            block_index=block_index,
            tts_text=tts_text,
            source_text=source_text,
            heading=plan.heading or "",
        )
        if kind == "pointer_diagram":
            plan.pointer_diagram = spec
        else:
            plan.env_diagram = spec

    plan = _enforce_visual_type(
        plan, block_index=block_index, source_text=source_text
    )
    # ai_image is no longer offered to the model; keep the field clear so the
    # image stage never tries to call an image provider.
    if plan.visual_type.value != "ai_image":
        plan.image_prompt = None
    return plan

"""Fake LLM provider — deterministic results for tests and demo mode.

Behavior:
    * Splits source_script into target-sized chunks around punctuation.
    * Returns visual plans based on simple keyword heuristics.
    * Synthesises a stable global_visual_style.

The fake provider never makes a network call and is the only provider that
the integration tests can rely on without environment variables.
"""
from __future__ import annotations

import hashlib
import json
import re

from app.providers.llm import LLMProvider, LLMRequest, LLMResponse


class FakeLLMProvider(LLMProvider):
    """Deterministic offline LLM used by tests and the demo command."""

    name = "fake"

    def __init__(self) -> None:
        """Create a provider with an inspectable record of fake calls."""
        self.calls: list[tuple[str, str, int]] = []  # (purpose, hash, n_messages)

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Infer the prompt purpose and return a deterministic JSON response."""
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        purpose = self._detect_purpose(last_user)
        h = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
        self.calls.append((purpose, h, len(request.messages)))

        if purpose == "split":
            content = self._build_split_payload(last_user)
        elif purpose == "global_style":
            content = json.dumps(
                {"global_visual_style": self._default_global_style()}, ensure_ascii=False
            )
        elif purpose == "visual_plan":
            content = json.dumps({"plan": self._build_visual_plan(last_user)}, ensure_ascii=False)
        elif purpose == "title":
            content = json.dumps({"title": "サンプル動画"}, ensure_ascii=False)
        else:
            content = json.dumps({"echo": True}, ensure_ascii=False)
        return LLMResponse(content=content, raw={"fake": True, "purpose": purpose})

    @staticmethod
    def _detect_purpose(text: str) -> str:
        """Classify a prompt using the markers used by the fake workflows."""
        # Order matters: the visual-plan prompt quotes the project's
        # global_visual_style, so it must be recognised before the
        # global-style prompt or every plan request is misrouted.
        if "blocks" in text and "source_text" in text:
            return "split"
        if "visual_type" in text:
            return "visual_plan"
        if "global_visual_style" in text:
            return "global_style"
        if '"title"' in text and "動画タイトル" in text:
            return "title"
        return "unknown"

    @staticmethod
    def _default_global_style() -> str:
        """Return the stable visual style used by fake planning."""
        return (
            "大学講義用のシンプルなスライド。白または薄いグレー(%)の背景、"
            "青緑(#0F766E)をアクセント、等幅フォントでコード、"
            "日本語は見出しと本文で2段階のフォントサイズ、"
            "過度な装飾やグラデーションは禁止。16:9。"
        )

    @staticmethod
    def _split_text_into_blocks(
        text: str, *, target: int = 90, max_chars: int = 130, min_chars: int = 50
    ) -> list[str]:
        """Deterministic punctuation-based splitter used as the fake output.

        This is also the deterministic *fallback* used when the real LLM
        fails to produce a usable split.
        """
        # Normalize whitespace: replace any whitespace run with a single space.
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        # Split into sentences on 。？！. but keep the delimiter.
        parts = re.findall(r"[^。？！.!?\n]+[。？！.!?]?", normalized)
        chunks: list[str] = []
        buf = ""
        for part in parts:
            if not part.strip():
                continue
            cand = (buf + part).strip() if buf else part.strip()
            if len(cand) <= max_chars:
                buf = cand
                continue
            if buf:
                chunks.append(buf)
                buf = part.strip()
            else:
                # Single sentence too long: split by だが/そして etc is too
                # brittle; just emit as-is so caller can recover.
                chunks.append(part.strip())
                buf = ""
        if buf:
            chunks.append(buf)
        # If too many tiny chunks, try to merge under target.
        merged: list[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) < min_chars and len(merged[-1]) + len(chunk) <= target:
                merged[-1] = merged[-1] + chunk
            else:
                merged.append(chunk)
        # Make absolutely sure each block is under max_chars (last resort:
        # break on a space if needed but try not to).
        result: list[str] = []
        for chunk in merged:
            if len(chunk) <= max_chars:
                result.append(chunk)
                continue
            # fallback: hard split on max_chars, prefer sentence boundaries.
            for i in range(0, len(chunk), max_chars):
                result.append(chunk[i : i + max_chars])
        return result

    @classmethod
    def _build_split_payload(cls, last_user: str) -> str:
        """Extract a script from a split prompt and serialize fake blocks."""
        # Try to extract the script from the prompt heuristically.
        m = re.search(r"```(?:text|script)?\s*(.*?)```", last_user, flags=re.DOTALL)
        if m:
            script = m.group(1)
        else:
            # The prompt ends with the script after a header marker; grab the last paragraph.
            script = last_user.split("\n\n")[-1]
        # Some prompts include a stray wrapper. The last line of the prompt
        # might be a sentinel like "SCRIPT:" — handle that.
        if "SCRIPT:" in script:
            script = script.split("SCRIPT:", 1)[1]
        blocks = cls._split_text_into_blocks(script)
        payload = {
            "blocks": [
                {"index": i, "source_text": b, "tts_text": b} for i, b in enumerate(blocks)
            ]
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_visual_plan(user_prompt: str) -> dict:
        """Choose a small visual-plan fixture from narration keywords."""
        # Heuristic visual-plan selection based on TTS text. Returns the full
        # plan structure expected by Pydantic.
        text_match = re.search(r"「(.+?)」", user_prompt, flags=re.DOTALL)
        text = text_match.group(1) if text_match else ""
        text.lower()
        if any(kw in text for kw in ["コード", "プログラム", "関数", "code"]):
            return {
                "visual_type": "code_slide",
                "heading": "サンプルコード",
                "visual_summary": "コード例を表示する",
                "image_prompt": None,
                "code": "# 例\ndef hello():\n    return 'world'\n",
                "language": "python",
                "formula": None,
                "diagram": None,
                "left_panel": None,
                "right_panel": None,
            }
        if any(kw in text for kw in ["= ", "＝", "sum", "∑"]):
            return {
                "visual_type": "formula",
                "heading": "数式",
                "visual_summary": "数式を表示する",
                "image_prompt": None,
                "code": None,
                "language": None,
                "formula": "f(x) = x^2 + 2x + 1",
                "diagram": None,
                "left_panel": None,
                "right_panel": None,
            }
        if any(kw in text for kw in ["cons", "セル", "car", "cdr", "ポインタ", "リスト"]):
            return {
                "visual_type": "pointer_diagram",
                "heading": "リスト構造",
                "visual_summary": "cons セルの箱とポインタ図",
                "image_prompt": None,
                "code": None,
                "language": None,
                "formula": None,
                "diagram": None,
                "pointer_diagram": {
                    "title": "リスト構造",
                    "caption": None,
                    "groups": [
                        {
                            "label": None,
                            "roots": [{"label": "x", "target": "c1"}],
                            "cells": [
                                {
                                    "id": "c1",
                                    "car": {"kind": "value", "text": "a", "target": None},
                                    "cdr": {"kind": "ref", "text": None, "target": "c2"},
                                },
                                {
                                    "id": "c2",
                                    "car": {"kind": "value", "text": "b", "target": None},
                                    "cdr": {"kind": "nil", "text": None, "target": None},
                                },
                            ],
                        }
                    ],
                },
                "env_diagram": None,
                "left_panel": None,
                "right_panel": None,
            }
        if any(kw in text for kw in ["図", "図解", "フロー", "アーキテクチャ"]):
            return {
                "visual_type": "diagram",
                "heading": "構成図",
                "visual_summary": "Mermaid 図で表示する",
                "image_prompt": None,
                "code": None,
                "language": None,
                "formula": None,
                "diagram": "graph TD\n  A[入力] --> B[処理]\n  B --> C[出力]",
                "left_panel": None,
                "right_panel": None,
            }
        if any(kw in text for kw in ["比較", "対比", "versus", "vs", "一方"]):
            return {
                "visual_type": "comparison",
                "heading": "比較",
                "visual_summary": "左右に並べて比較する",
                "image_prompt": None,
                "code": None,
                "language": None,
                "formula": None,
                "diagram": None,
                "left_panel": {"title": "案A", "content": "簡単な実装"},
                "right_panel": {"title": "案B", "content": "拡張可能な実装"},
            }
        if any(kw in text for kw in ["はじめに", "本章", "概要", "今回"]):
            return {
                "visual_type": "title_slide",
                "heading": "本節の概要",
                "visual_summary": "章タイトル",
                "image_prompt": None,
                "code": None,
                "language": None,
                "formula": None,
                "diagram": None,
                "left_panel": None,
                "right_panel": None,
            }
        return {
            "visual_type": "text_slide",
            "heading": text[:24] or "要点",
            "visual_summary": "テキストで要点を表示",
            "image_prompt": None,
            "code": None,
            "language": None,
            "formula": None,
            "diagram": None,
            "left_panel": None,
            "right_panel": None,
        }

"""Unit tests for the structured diagram renderers and the planner guard.

The renderers must always emit an image at exactly the requested canvas size:
anything else gets letterboxed (and blur-upscaled) by ffmpeg downstream.
"""
from __future__ import annotations

import pytest
from PIL import Image

from app.services.diagram_renderer import render_env_diagram, render_pointer_diagram
from app.services.stage_schemas import VisualPlan
from app.services.stage_schemas import VisualPlanPayload
from app.services.visual_planner import (
    _enforce_visual_type,
    _looks_like_code,
    _normalize_plan_payload,
)


SPEC = {"groups": [{"cells": [{"id": "c1", "car": {"kind": "value", "text": "a"},
                              "cdr": {"kind": "nil"}}]}]}


def _plan(payload):
    return VisualPlanPayload.model_validate(_normalize_plan_payload(payload)).plan


def test_spec_written_into_the_mermaid_field_is_moved() -> None:
    """Models pick pointer_diagram then park the spec in ``diagram``."""
    plan = _plan({"plan": {"visual_type": "pointer_diagram", "heading": "h",
                           "diagram": SPEC}})
    assert plan.visual_type.value == "pointer_diagram"
    assert plan.pointer_diagram == SPEC
    assert plan.diagram is None


def test_spec_hung_off_the_top_level_is_moved_into_plan() -> None:
    plan = _plan({"plan": {"visual_type": "pointer_diagram", "heading": "h"},
                  "pointer_diagram": SPEC})
    assert plan.pointer_diagram == SPEC


def test_stray_object_in_diagram_is_cleared_not_fatal() -> None:
    plan = _plan({"plan": {"visual_type": "text_slide", "heading": "h"},
                  "diagram": SPEC})
    assert plan.visual_type.value == "text_slide"
    assert plan.diagram is None


def test_valid_mermaid_string_is_untouched() -> None:
    src = "flowchart TD\n  A --> B"
    plan = _plan({"plan": {"visual_type": "diagram", "heading": "h", "diagram": src}})
    assert plan.diagram == src


def test_normalize_passes_through_non_dict() -> None:
    assert _normalize_plan_payload([1, 2]) == [1, 2]


POINTER_SPEC = {
    "title": "リスト構造",
    "caption": "cdr でつながる",
    "groups": [
        {
            "label": "変更前",
            "roots": [{"label": "x", "target": "c1"}],
            "cells": [
                {"id": "c1", "car": {"kind": "value", "text": "a"},
                 "cdr": {"kind": "ref", "target": "c2"}},
                {"id": "c2", "car": {"kind": "value", "text": "b"},
                 "cdr": {"kind": "nil"}},
            ],
        }
    ],
}

ENV_SPEC = {
    "title": "環境モデル",
    "frames": [
        {"id": "g", "label": "グローバル環境", "parent": None,
         "bindings": [{"name": "x", "value": "10"}]},
        {"id": "e1", "label": "E1", "parent": "g",
         "bindings": [{"name": "y", "value": "5"}]},
    ],
    "procedures": [
        {"id": "p1", "label": "square", "params": "x", "body": "(* x x)", "env": "g"}
    ],
}


@pytest.mark.parametrize("size", [(1920, 880), (1920, 1080), (1280, 720)])
def test_pointer_diagram_fills_requested_canvas(tmp_path, size) -> None:
    out = tmp_path / "p.png"
    width, height = size
    render_pointer_diagram(POINTER_SPEC, out, width=width, height=height)
    assert Image.open(out).size == (width, height)


def test_env_diagram_fills_requested_canvas(tmp_path) -> None:
    out = tmp_path / "e.png"
    render_env_diagram(ENV_SPEC, out, width=1920, height=880)
    assert Image.open(out).size == (1920, 880)


def test_pointer_diagram_accepts_a_cycle(tmp_path) -> None:
    """A cdr pointing back to an earlier cell must not blow up layout."""
    spec = {
        "groups": [
            {
                "cells": [
                    {"id": "c1", "car": {"kind": "value", "text": "a"},
                     "cdr": {"kind": "ref", "target": "c2"}},
                    {"id": "c2", "car": {"kind": "value", "text": "b"},
                     "cdr": {"kind": "ref", "target": "c1"}},
                ]
            }
        ]
    }
    out = tmp_path / "cycle.png"
    render_pointer_diagram(spec, out, width=1920, height=880)
    assert Image.open(out).size == (1920, 880)


def test_pointer_diagram_tolerates_bare_string_slots(tmp_path) -> None:
    """Models sometimes write "a" / "nil" instead of the object form."""
    spec = {"cells": [{"id": "c1", "car": "a", "cdr": "nil"}]}
    out = tmp_path / "loose.png"
    render_pointer_diagram(spec, out, width=1920, height=880)
    assert Image.open(out).size == (1920, 880)


def test_pointer_diagram_rejects_empty_spec(tmp_path) -> None:
    with pytest.raises(ValueError):
        render_pointer_diagram({"groups": []}, tmp_path / "x.png", width=1920, height=880)


def test_env_diagram_rejects_empty_spec(tmp_path) -> None:
    with pytest.raises(ValueError):
        render_env_diagram({"frames": []}, tmp_path / "x.png", width=1920, height=880)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("flowchart TD\n  A --> B", False),
        ("graph LR\n  A --> B", False),
        ("(define (make-cycle x) (set-cdr! (last-pair x) x) x)", True),
        ("def hello():\n    return 1", True),
        ("", False),
    ],
)
def test_looks_like_code(body, expected) -> None:
    assert _looks_like_code(body) is expected


def test_guard_moves_code_from_diagram_to_code_slide() -> None:
    plan = VisualPlan(
        visual_type="diagram",
        heading="コード",
        diagram="(define (square x) (* x x))",
    )
    result = _enforce_visual_type(plan, block_index=0)
    assert result.visual_type.value == "code_slide"
    assert result.code == "(define (square x) (* x x))"
    assert result.diagram is None


def test_guard_downgrades_pointer_diagram_without_a_spec() -> None:
    plan = VisualPlan(visual_type="pointer_diagram", heading="構造")
    result = _enforce_visual_type(plan, block_index=0)
    assert result.visual_type.value == "text_slide"


SCHEME_BLOCK = (
    "実際のコードはこうです。\n"
    "```scheme\n(define (make-cycle x)\n  (set-cdr! (last-pair x) x)\n  x)\n```"
)
ASCII_BLOCK = "こうなります。\n```text\n先頭 → 次\n ↑    │\n └────┘\n```"


def test_code_in_the_script_is_shown_as_a_slide() -> None:
    """A block carrying a real listing must display it, even when the model
    preferred a diagram for the surrounding narration."""
    plan = VisualPlan(visual_type="pointer_diagram", heading="循環")
    result = _enforce_visual_type(plan, block_index=0, source_text=SCHEME_BLOCK)
    assert result.visual_type.value == "code_slide"
    assert "make-cycle" in (result.code or "")
    assert result.language == "scheme"


def test_ascii_art_does_not_become_a_code_slide() -> None:
    """Hand-drawn diagrams belong to the diagram renderers, not code slides."""
    plan = VisualPlan(
        visual_type="pointer_diagram",
        heading="循環",
        pointer_diagram={"groups": [{"cells": [{"id": "c1", "car": "a", "cdr": "nil"}]}]},
    )
    result = _enforce_visual_type(plan, block_index=0, source_text=ASCII_BLOCK)
    assert result.visual_type.value == "pointer_diagram"


def test_guard_is_a_noop_without_a_code_fence() -> None:
    plan = VisualPlan(
        visual_type="pointer_diagram",
        heading="循環",
        pointer_diagram={"groups": [{"cells": [{"id": "c1", "car": "a", "cdr": "nil"}]}]},
    )
    result = _enforce_visual_type(plan, block_index=0, source_text="ただの文章です。")
    assert result.visual_type.value == "pointer_diagram"


def test_guard_keeps_a_valid_mermaid_diagram() -> None:
    plan = VisualPlan(
        visual_type="diagram", heading="流れ", diagram="flowchart TD\n  A --> B"
    )
    result = _enforce_visual_type(plan, block_index=0)
    assert result.visual_type.value == "diagram"

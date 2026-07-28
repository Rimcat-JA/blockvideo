"""Tests for the visual plan schemas and content hashing."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.block import VisualType
from app.services.hashing import hash_value, short_hash
from app.services.stage_schemas import GlobalVisualStylePayload, VisualPlan, VisualPlanPayload


def test_visual_plan_rejects_unknown_types() -> None:
    with pytest.raises(ValidationError):
        VisualPlan.model_validate({"visual_type": "bogus"})


def test_visual_plan_accepts_all_supported_types() -> None:
    for value in VisualType:
        plan = VisualPlan.model_validate({"visual_type": value.value})
        assert plan.visual_type == value


def test_visual_plan_image_prompt_only_for_ai_image() -> None:
    plan = VisualPlan.model_validate(
        {"visual_type": "ai_image", "image_prompt": "a colorful diagram"}
    )
    assert plan.image_prompt == "a colorful diagram"


def test_visual_plan_rejects_unexpected_panel_keys() -> None:
    payload = {
        "visual_type": "comparison",
        "left_panel": {"title": "L", "content": "x", "evil": "<script>"},
        "right_panel": {"title": "R", "content": "y", "javascript": "alert(1)"},
    }
    parsed = VisualPlan.model_validate(payload)
    assert "evil" not in parsed.left_panel
    assert "javascript" not in parsed.right_panel


def test_global_visual_style_payload_rejects_long_strings() -> None:
    with pytest.raises(ValidationError):
        GlobalVisualStylePayload.model_validate({"global_visual_style": "x" * 3000})


def test_visual_plan_payload_requires_plan() -> None:
    with pytest.raises(ValidationError):
        VisualPlanPayload.model_validate({"notplan": "x"})


def test_hash_deterministic_and_unique() -> None:
    h1 = hash_value("block-1", "text here", {"visual_type": "title_slide"})
    h2 = hash_value("block-1", "text here", {"visual_type": "title_slide"})
    h3 = hash_value("block-1", "text here", {"visual_type": "ai_image"})
    h4 = hash_value("block-2", "text here", {"visual_type": "title_slide"})
    assert h1 == h2
    assert h1 != h3
    assert h1 != h4
    assert len(short_hash(h1)) == 16


def test_hash_ignores_dict_order() -> None:
    h1 = hash_value({"a": 1, "b": 2})
    h2 = hash_value({"b": 2, "a": 1})
    assert h1 == h2

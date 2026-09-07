"""Narration focus must preserve content, grid geometry, and full-slide fit."""
from __future__ import annotations

import pytest
from PIL import Image, ImageChops

from app.services.diagram_renderer import BG, INK, render_verbatim_slide
from app.services.image_renderer import render_visual_plan
from app.services.visual_focus import focus_terms, literal_ranges, matching_terms


ART = "┌───────┬─────────┐\n│ key   │ value   │\n└───────┴─────────┘"


def _render(tmp_path, plan, focus=None, name="slide", size=(1280, 720)):
    path = tmp_path / f"{name}.png"
    render_visual_plan(plan, path, width=size[0], height=size[1], focus_text=focus)
    with Image.open(path) as image:
        return image.convert("RGB")


def test_literal_matching_does_not_guess_or_match_identifier_fragments():
    plan = {"visual_type": "verbatim_slide", "verbatim": "key value table x"}
    assert focus_terms(plan, "monkeyとvaluableとstableとindexです。") == ()
    assert focus_terms(plan, "keyからvalueを取得します。") == ("key", "value")
    assert focus_terms(plan, "キーと値を見ます。") == ()
    assert focus_terms(plan, "xの参照先です。") == ("x",)
    assert focus_terms(plan, "KEYです。") == ()


def test_japanese_labels_and_explicit_quoted_labels_are_literal():
    assert matching_terms("キー → 値の保存", "キーを調べます。") == ("キー",)
    assert matching_terms("キー → 値の保存", "「値」を見ます。") == ("値",)
    assert matching_terms("1 2 42", "1つ目です。") == ()
    assert matching_terms("値の保存", "変更します。") == ()


def test_matching_is_stable_for_variant_reuse_and_keeps_distinct_identifiers():
    plan = {"visual_type": "code_slide", "code": "key = monkey\nreturn value"}
    assert focus_terms(plan, "valueとkeyです。") == focus_terms(plan, "key、valueです。")
    assert focus_terms(plan, "monkeyとkeyです。") == ("key", "monkey")
    assert literal_ranges("key key_value set-key! key", "key") == ((0, 3), (23, 26))


def test_unrenderable_and_heading_only_cues_do_not_request_variants():
    assert focus_terms({"visual_type": "diagram", "diagram": "key"}, "keyです。") == ()
    assert focus_terms({"visual_type": "verbatim_slide", "heading": "key", "verbatim": "value"},
                       "keyです。") == ()


def test_focus_changes_background_only_without_moving_original_grid(tmp_path):
    plan = {"visual_type": "verbatim_slide", "heading": "キーと値", "verbatim": ART}
    static = _render(tmp_path, plan)
    focused = _render(tmp_path, plan, "keyを見ます。", "focus")
    diff = ImageChops.difference(static, focused).getbbox()
    assert diff is not None
    # Every original dark glyph/box pixel remains in precisely the same place.
    static_bytes, focused_bytes = static.tobytes(), focused.tobytes()
    ink = bytes(INK)
    offsets = [i for i in range(0, len(static_bytes), 3) if static_bytes[i:i + 3] == ink]
    assert offsets and all(focused_bytes[i:i + 3] == ink for i in offsets)
    # The focus stays inside the content, below the heading and above margins.
    left, top, right, bottom = diff
    assert 0 < left < right < static.width
    assert 100 < top < bottom < static.height - 20
    assert bottom - top < 180


def test_missing_term_keeps_the_exact_static_slide(tmp_path):
    plan = {"visual_type": "verbatim_slide", "heading": "図", "verbatim": ART}
    static = _render(tmp_path, plan)
    focused = _render(tmp_path, plan, "ここでは説明を続けます。", "no_match")
    assert ImageChops.difference(static, focused).getbbox() is None


def test_sparse_authored_diagrams_use_more_than_old_64_pixel_font(tmp_path):
    path = tmp_path / "large.png"
    render_verbatim_slide("┌─────┐\n│ key │\n└─────┘", path, width=1920, height=880)
    with Image.open(path) as image:
        box = ImageChops.difference(image, Image.new("RGB", image.size, BG)).getbbox()
    assert box is not None
    assert box[2] - box[0] > 400
    assert box[3] - box[1] > 240


@pytest.mark.parametrize("size", [(1920, 880), (1280, 720), (640, 360)])
def test_dense_authored_diagram_keeps_last_row_inside_canvas(tmp_path, size):
    body = "\n".join(["│ long unchanged diagram content │"] * 34 + ["│ final_marker                  │"])
    plan = {"visual_type": "verbatim_slide", "heading": "長い図", "verbatim": body}
    static = _render(tmp_path, plan, size=size)
    focused = _render(tmp_path, plan, "final_markerです。", "last", size=size)
    bounds = ImageChops.difference(static, focused).getbbox()
    assert static.size == size
    assert bounds is not None
    assert size[1] * 0.75 < bounds[1] < bounds[3] <= size[1] - 10


@pytest.mark.parametrize("visual_type,field", [("code_slide", "code"), ("text_slide", "visual_summary")])
def test_dense_code_and_text_keep_the_last_line_and_focus_it(tmp_path, visual_type, field):
    body = "\n".join([f"row_{i} = current_value" for i in range(30)] + ["last_visible_marker"])
    plan = {"visual_type": visual_type, "heading": "すべての行を表示", field: body}
    static = _render(tmp_path, plan)
    focused = _render(tmp_path, plan, "last_visible_markerです。", "last")
    bounds = ImageChops.difference(static, focused).getbbox()
    assert bounds is not None
    assert 550 < bounds[1] < bounds[3] < 710


def test_focus_argument_remains_optional_for_title_slides(tmp_path):
    result = _render(tmp_path, {"visual_type": "title_slide", "heading": "Title"}, "Titleです。")
    assert result.size == (1280, 720)

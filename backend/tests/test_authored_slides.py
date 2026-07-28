"""Slides the script author drew are shown exactly as written.

A model asked to design a diagram from a schema must invent identifiers for
whatever the script left implicit, and it does: one real 77-block project came
back with 115 labels — ``hdr``, ``k2``, ``2nd1``, ``local*``, ``∅`` — that
appear nowhere in the script the viewer is listening to. Content the author
supplied cannot invent anything, so a block carrying a ```slide fence skips
planning altogether.
"""
from __future__ import annotations

from app.models.block import VisualType
from app.services.splitter import sanitize_for_narration
from app.services.visual_planner import (
    authored_plan,
    build_slide_sequence,
    extract_authored_slide,
    extract_code_block,
)

F = "`" * 3
ART = "   table\n     │\n     ▼\n   ┌───┬───┐\n   │ ● │ ● │\n   └───┴───┘"
SRC = "テーブルは連結リストです。\n\n" + F + "slide キー→値の対応\n" + ART + "\n" + F


class TestExtraction:
    def test_the_heading_and_body_are_separated(self) -> None:
        heading, body = extract_authored_slide(SRC)
        assert heading == "キー→値の対応"
        assert body == ART

    def test_a_japanese_heading_survives(self) -> None:
        """The generic fence regex only accepts ASCII info strings."""
        heading, _ = extract_authored_slide(F + "slide 図：先頭への追加\n" + ART + "\n" + F)
        assert heading == "図：先頭への追加"

    def test_the_heading_is_optional(self) -> None:
        heading, body = extract_authored_slide(F + "slide\n" + ART + "\n" + F)
        assert heading == ""
        assert body == ART

    def test_leading_spaces_are_preserved(self) -> None:
        """Indentation is the drawing; losing it destroys the alignment."""
        _, body = extract_authored_slide(SRC)
        assert body.splitlines()[0].startswith("   table")

    def test_a_block_without_one_returns_nothing(self) -> None:
        assert extract_authored_slide("ふつうの文章です。") is None

    def test_an_empty_slide_is_not_a_slide(self) -> None:
        assert extract_authored_slide(F + "slide\n   \n" + F) is None

    def test_a_code_fence_is_not_mistaken_for_one(self) -> None:
        assert extract_authored_slide(F + "scheme\n(car x)\n" + F) is None


class TestPlanning:
    def test_the_plan_is_built_without_a_model(self) -> None:
        plan = authored_plan(*extract_authored_slide(SRC))
        assert plan.visual_type == VisualType.verbatim_slide
        assert plan.heading == "キー→値の対応"
        assert plan.verbatim == ART

    def test_the_body_is_carried_verbatim(self) -> None:
        plan = authored_plan("h", "  ┌─┐\n  │x│")
        assert plan.verbatim == "  ┌─┐\n  │x│"

    def test_an_authored_slide_never_gains_extra_slides(self) -> None:
        """Appending the block's other fences would contradict the author."""
        plan = authored_plan(*extract_authored_slide(SRC))
        assert len(build_slide_sequence(plan.model_dump(), SRC, max_slides=6)) == 1

    def test_the_slide_fence_is_not_also_shown_as_code(self) -> None:
        plan = authored_plan(*extract_authored_slide(SRC))
        sequence = build_slide_sequence(plan.model_dump(), SRC, max_slides=6)
        assert all(s.get("code") is None for s in sequence)


class TestNarration:
    def test_the_drawing_is_never_read_aloud(self) -> None:
        assert sanitize_for_narration(SRC) == "テーブルは連結リストです。"

    def test_the_heading_is_not_read_aloud_either(self) -> None:
        assert "キー→値の対応" not in sanitize_for_narration(SRC)

    def test_a_slide_fence_is_not_treated_as_a_code_listing(self) -> None:
        """Otherwise the guard would force a code slide over the drawing."""
        assert extract_code_block(F + "slide\n" + ART + "\n" + F) is None


class TestGridRendering:
    """Alignment is the whole content of a hand-drawn diagram."""

    def _render(self, tmp_path, body, name="s.png"):
        from app.services.diagram_renderer import render_verbatim_slide
        out = tmp_path / name
        render_verbatim_slide(body, out, width=1920, height=880, title="図")
        return out

    def test_it_produces_a_full_frame_slide(self, tmp_path) -> None:
        from PIL import Image
        out = self._render(tmp_path, ART)
        assert Image.open(out).size == (1920, 880)

    def test_mixed_japanese_and_box_drawing_renders(self, tmp_path) -> None:
        """The case that broke font-based layout: CJK is twice the cell width."""
        body = "変更前： table ──▶ ┌───┬───┐\n                  │ a │ 1 │"
        assert self._render(tmp_path, body, "m.png").exists()

    def test_a_glyph_the_code_font_lacks_still_draws(self, tmp_path) -> None:
        """Consolas has no ▶; it must come from the wide font, not tofu."""
        from app.services.diagram_renderer import _GRID_MONO, _has_glyph, _load
        mono = _load(_GRID_MONO, 40)
        assert not _has_glyph(mono, "▶")
        assert _has_glyph(mono, "┌")

    def test_east_asian_width_decides_cell_count(self) -> None:
        from app.services.diagram_renderer import _cells
        assert _cells("a") == 1
        assert _cells("┌") == 1, "box drawing is drawn as narrow in an editor"
        assert _cells("あ") == 2
        assert _cells("変") == 2

    def test_an_empty_body_does_not_crash(self, tmp_path) -> None:
        assert self._render(tmp_path, "", "e.png").exists()

    def test_it_creates_the_block_directory(self, tmp_path) -> None:
        """Every other renderer does; missing it failed a whole project at
        the first block whose slide was author-drawn."""
        from app.services.diagram_renderer import render_verbatim_slide
        out = tmp_path / "projects" / "0027" / "blocks" / "0001" / "image.png"
        render_verbatim_slide(ART, out, width=1920, height=880, title="図")
        assert out.exists()

    def test_a_very_wide_drawing_is_scaled_to_fit(self, tmp_path) -> None:
        from PIL import Image
        out = self._render(tmp_path, "─" * 400, "w.png")
        assert Image.open(out).size == (1920, 880)


class TestAlignmentLint:
    """A ragged box reaches the screen ragged, so name the line."""

    def test_a_short_inner_line_is_reported(self) -> None:
        from app.services.visual_planner import slide_alignment_issues
        body = " ┌───────────┐\n │ table │\n └───────────┘"
        issues = slide_alignment_issues(body)
        assert len(issues) == 1
        assert "table" in issues[0]

    def test_a_correctly_padded_box_is_silent(self) -> None:
        from app.services.visual_planner import slide_alignment_issues
        body = " ┌───────────┐\n │   table   │\n └───────────┘"
        assert slide_alignment_issues(body) == []

    def test_japanese_counts_as_two_cells(self) -> None:
        from app.services.visual_planner import slide_alignment_issues
        assert slide_alignment_issues("┌────┐\n│ 表 │\n└────┘") == []
        assert slide_alignment_issues("┌──────┐\n│ 表 │\n└──────┘") != []

    def test_connectors_outside_a_box_are_not_flagged(self) -> None:
        """A lone │ leading into a box is shorter by design."""
        from app.services.visual_planner import slide_alignment_issues
        assert slide_alignment_issues("  │\n  ▼\n┌────┐\n│ ab │\n└────┘") == []

    def test_art_without_boxes_is_never_flagged(self) -> None:
        from app.services.visual_planner import slide_alignment_issues
        assert slide_alignment_issues("key ──▶ value") == []
        assert slide_alignment_issues("") == []


class TestWhitespaceSurvivesNormalisation:
    """A drawing's spacing is its content and must not be collapsed."""

    def test_padding_inside_a_slide_is_kept(self) -> None:
        from app.services.splitter import normalize_kept
        script = "本文です。\n\n" + F + "slide 図\n┌───────────┐\n│   table   │\n└───────────┘\n" + F
        assert "│   table   │" in normalize_kept(script)

    def test_prose_runs_are_still_collapsed(self) -> None:
        from app.services.splitter import normalize_kept
        assert normalize_kept("これは    テストです。") == "これは テストです。"

    def test_indentation_of_art_survives(self) -> None:
        from app.services.splitter import normalize_kept
        script = F + "slide\n    key\n     │\n     ▼\n" + F
        assert "    key" in normalize_kept(script)

    def test_prose_around_a_fence_is_still_collapsed(self) -> None:
        from app.services.splitter import normalize_kept
        script = "前    です。\n" + F + "slide\n  ┌─┐\n" + F + "\n後    です。"
        out = normalize_kept(script)
        assert "前 です。" in out and "後 です。" in out
        assert "  ┌─┐" in out

    def test_a_ragged_box_is_no_longer_produced_by_the_pipeline(self) -> None:
        """End to end: a correctly padded script stays correctly padded."""
        from app.services.splitter import normalize_kept
        from app.services.visual_planner import (
            extract_authored_slide, slide_alignment_issues,
        )
        script = F + "slide 図\n┌───────────┐\n│   table   │\n└───────────┘\n" + F
        authored = extract_authored_slide(normalize_kept(script))
        assert slide_alignment_issues(authored[1]) == []

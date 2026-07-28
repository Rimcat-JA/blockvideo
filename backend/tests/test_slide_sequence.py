"""A block shows every visual its script contained, not just the first.

A slide holds one thing, but a script can pack several listings and diagrams
between two sentences. Those extra visuals are shown in sequence underneath
the block's narration; without this they never reach the screen at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.ffmpeg_runner import build_block_video_args
from app.services.pipeline import MIN_SLIDE_MS, _block_slides
from app.services.visual_planner import build_slide_sequence, extract_all_fences


SRC = (
    "たとえば、\n```scheme\n(cons 'a 1)\n```\n"
    "は、\n```scheme\n(a . 1)\n```\n"
    "というレコードです。\n```text\n┌───┬───┐\n│ a │ 1 │\n└───┴───┘\n```"
)


def test_all_fences_are_found_in_document_order() -> None:
    fences = extract_all_fences(SRC)
    assert [lang for _, lang in fences] == ["scheme", "scheme", "text"]
    assert fences[0][0] == "(cons 'a 1)"


def test_ascii_art_fences_are_included() -> None:
    """Hand-drawn diagrams were drawn to be looked at."""
    bodies = [b for b, _ in extract_all_fences(SRC)]
    assert any("┌" in b for b in bodies)


def test_empty_fences_are_ignored() -> None:
    assert extract_all_fences("```scheme\n\n```") == []


def test_sequence_keeps_the_planner_visual_first() -> None:
    seq = build_slide_sequence({"visual_type": "pointer_diagram", "heading": "h"}, SRC)
    assert seq[0]["visual_type"] == "pointer_diagram"
    assert len(seq) == 4
    assert all(s["visual_type"] == "code_slide" for s in seq[1:])


def test_sequence_does_not_repeat_the_primary_code() -> None:
    seq = build_slide_sequence(
        {"visual_type": "code_slide", "heading": "h", "code": "(cons 'a 1)"}, SRC
    )
    assert len(seq) == 3
    assert "(cons 'a 1)" not in [s.get("code") for s in seq[1:]]


def test_sequence_is_capped() -> None:
    many = "\n".join(f"```scheme\n(f {i})\n```" for i in range(20))
    seq = build_slide_sequence({"visual_type": "text_slide", "heading": "h"}, many)
    assert len(seq) == 6


def test_a_block_without_fences_shows_one_slide() -> None:
    seq = build_slide_sequence({"visual_type": "text_slide", "heading": "h"}, "ただの文章。")
    assert len(seq) == 1


def test_durations_are_shared_and_sum_to_the_block(tmp_path) -> None:
    primary = tmp_path / "image.png"
    primary.touch()
    for slot in (1, 2):
        (tmp_path / f"image_{slot}.png").touch()

    import app.services.pipeline as pipeline

    original = pipeline.block_image_path
    pipeline.block_image_path = lambda pid, idx, slot=0: (
        tmp_path / ("image.png" if slot == 0 else f"image_{slot}.png")
    )
    try:
        slides = _block_slides(1, 0, primary, 30_000, max_slides=3)
    finally:
        pipeline.block_image_path = original

    assert len(slides) == 3
    assert sum(ms for _, ms in slides) == 30_000
    assert all(ms >= MIN_SLIDE_MS for _, ms in slides)


def test_short_blocks_drop_extra_slides_rather_than_flashing(tmp_path) -> None:
    primary = tmp_path / "image.png"
    primary.touch()
    for slot in (1, 2, 3):
        (tmp_path / f"image_{slot}.png").touch()

    import app.services.pipeline as pipeline

    original = pipeline.block_image_path
    pipeline.block_image_path = lambda pid, idx, slot=0: (
        tmp_path / ("image.png" if slot == 0 else f"image_{slot}.png")
    )
    try:
        slides = _block_slides(1, 0, primary, 3_000)
    finally:
        pipeline.block_image_path = original

    assert len(slides) == 1, "a 3s block cannot show four slides legibly"
    assert slides[0][1] == 3_000


def test_single_slide_argv_is_unchanged() -> None:
    args = build_block_video_args(
        image=Path("i.png"), audio=Path("a.wav"), duration_ms=5_000,
        output=Path("o.mp4"), ffmpeg="ffmpeg",
    )
    assert "-vf" in args
    assert "-filter_complex" not in args


def test_multi_slide_argv_concatenates_and_maps_audio() -> None:
    args = build_block_video_args(
        slides=[(Path("a.png"), 3_000), (Path("b.png"), 2_000)],
        audio=Path("a.wav"), duration_ms=5_000,
        output=Path("o.mp4"), ffmpeg="ffmpeg",
    )
    graph = args[args.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in graph
    assert args[args.index("-map") + 1] == "[vout]"
    # audio is the input after the images
    assert "2:a" in args
    # each image input is time-limited so the concat honours the split
    assert [args[i + 1] for i, x in enumerate(args) if x == "-t"] == [
        "3.000", "2.000", "5.000",
    ]


def test_slides_require_at_least_one_image() -> None:
    with pytest.raises(ValueError):
        build_block_video_args(
            audio=Path("a.wav"), duration_ms=1_000,
            output=Path("o.mp4"), ffmpeg="ffmpeg",
        )


def _slides_with_images(tmp_path, count: int, duration_ms: int, **kwargs):
    """Build ``count`` slide images and ask the pipeline what it would show."""
    primary = tmp_path / "image.png"
    primary.touch()
    for slot in range(1, count):
        (tmp_path / f"image_{slot}.png").touch()

    import app.services.pipeline as pipeline

    original = pipeline.block_image_path
    pipeline.block_image_path = lambda pid, idx, slot=0: (
        tmp_path / ("image.png" if slot == 0 else f"image_{slot}.png")
    )
    try:
        return _block_slides(1, 0, primary, duration_ms, **kwargs)
    finally:
        pipeline.block_image_path = original


def test_the_cap_limits_how_many_slides_a_block_shows(tmp_path) -> None:
    """Six listings under forty seconds is six seconds each — too fast to read."""
    slides = _slides_with_images(tmp_path, 6, 40_000, max_slides=3)
    assert len(slides) == 3
    assert sum(ms for _, ms in slides) == 40_000


def test_the_cap_gives_each_remaining_slide_more_time(tmp_path) -> None:
    uncapped = _slides_with_images(tmp_path, 6, 40_000, max_slides=9)
    capped = _slides_with_images(tmp_path, 6, 40_000, max_slides=3)
    assert min(ms for _, ms in capped) > min(ms for _, ms in uncapped)


def test_slides_are_dropped_from_the_end(tmp_path) -> None:
    """The planner's visual comes first, the rest in the order the script has them."""
    capped = _slides_with_images(tmp_path, 5, 40_000, max_slides=2)
    assert [p.name for p, _ in capped] == ["image.png", "image_1.png"]


def test_a_cap_of_one_still_shows_the_primary_slide(tmp_path) -> None:
    slides = _slides_with_images(tmp_path, 5, 40_000, max_slides=1)
    assert len(slides) == 1
    assert slides[0][0].name == "image.png"
    assert slides[0][1] == 40_000


def test_a_block_with_fewer_slides_than_the_cap_is_untouched(tmp_path) -> None:
    slides = _slides_with_images(tmp_path, 2, 40_000, max_slides=6)
    assert len(slides) == 2


def test_the_duration_floor_still_wins_over_a_generous_cap(tmp_path) -> None:
    """A 3s block cannot show three slides however high the cap is set."""
    slides = _slides_with_images(tmp_path, 4, 3_000, max_slides=8)
    assert len(slides) == 1


class TestTheCapStopsSlidesBeingDrawn:
    """The cap belongs to the image stage, not the renderer.

    Applying it only at render time meant every extra listing was still
    composed, written to disk and then discarded — work paid for and thrown
    away, and a stale PNG left behind for the renderer to trip over.
    """

    PLAN = {"visual_type": "code_slide", "heading": "見出し", "code": "(cons 'a 1)"}

    def test_a_cap_of_one_plans_only_the_primary_visual(self) -> None:
        sequence = build_slide_sequence(self.PLAN, SRC, max_slides=1)
        assert len(sequence) == 1
        assert sequence[0] is not None
        assert sequence[0]["code"] == "(cons 'a 1)"

    def test_the_cap_limits_how_many_plans_are_produced(self) -> None:
        assert len(build_slide_sequence(self.PLAN, SRC, max_slides=2)) == 2
        assert len(build_slide_sequence(self.PLAN, SRC, max_slides=3)) == 3

    def test_a_cap_beyond_the_script_yields_only_what_exists(self) -> None:
        """SRC has three fences, one of which is already the primary."""
        assert len(build_slide_sequence(self.PLAN, SRC, max_slides=9)) == 3

    def test_extras_keep_the_order_the_script_introduces_them(self) -> None:
        sequence = build_slide_sequence(self.PLAN, SRC, max_slides=9)
        assert [s.get("language") for s in sequence[1:]] == ["scheme", "text"]

    def test_the_renderer_agrees_with_what_was_drawn(self, tmp_path) -> None:
        """Same cap on both sides, so no drawn slide goes unshown."""
        cap = 2
        sequence = build_slide_sequence(self.PLAN, SRC, max_slides=cap)
        for slot in range(len(sequence)):
            (tmp_path / ("image.png" if slot == 0 else f"image_{slot}.png")).touch()
        slides = _slides_with_images(tmp_path, len(sequence), 40_000, max_slides=cap)
        assert len(slides) == len(sequence)

"""End to end: the real models and every built-in effect on a real portrait.

Skipped without scikit-image (the test portrait) or without the models
(``AI_MAKEUP_CAM_MODELS_DIR``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam.config import SECTIONS, MakeupSettings
from makeup_cam.effects import load_builtin_effects
from makeup_cam.effects.base import Effect, FrameContext
from makeup_cam.models import FACE_LANDMARKER, SELFIE_MULTICLASS, ensure_model
from makeup_cam.pipeline import MakeupPipeline, ProcessResult
from makeup_cam.presets import list_presets, load_preset
from makeup_cam.regions import FACE_OVAL
from makeup_cam.segmentation import Segmenter
from makeup_cam.tracking import FaceTracker

# Enough frames for the tracker to settle and the makeup to fade in fully.
WARMUP_FRAMES = 15
FRAME_MS = 33
CORNER_PX = 64
# Effects must return values in [0, 1]; allow float rounding.
RANGE_TOLERANCE = 1e-4


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the user's own presets and session out of these tests."""
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))


def _checked(effect_cls: type[Effect], problems: list[str]) -> Effect:
    """An instance of ``effect_cls`` that records outputs that are not finite or leave [0, 1]."""

    class Checked(effect_cls):  # type: ignore[valid-type,misc]
        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: object) -> np.ndarray:
            out = super().apply(frame, ctx, settings)
            finite = bool(np.isfinite(out).all())
            if not finite or out.min() < -RANGE_TOLERANCE or out.max() > 1.0 + RANGE_TOLERANCE:
                problems.append(self.name)
            return out

    return Checked()


class Session:
    """A real pipeline fed one still frame as a video."""

    def __init__(self, models_dir: Path, settings: MakeupSettings, effects: list[Effect] | None = None) -> None:
        tracker = FaceTracker(ensure_model(FACE_LANDMARKER, directory=models_dir))
        segmenter = Segmenter(ensure_model(SELFIE_MULTICLASS, directory=models_dir), background=False)
        self.pipeline = MakeupPipeline(settings, tracker=tracker, segmenter=segmenter, effects=effects)
        self._frame_index = 0

    def run(self, frame: np.ndarray, frames: int) -> ProcessResult:
        result = None
        for _ in range(frames):
            result = self.pipeline.process(frame, self._frame_index * FRAME_MS)
            self._frame_index += 1
        assert result is not None
        return result


SessionFactory = Callable[..., Session]


@pytest.fixture
def session_factory(models_dir: Path) -> Iterator[SessionFactory]:
    sessions: list[Session] = []

    def make(settings: MakeupSettings, effects: list[Effect] | None = None) -> Session:
        sessions.append(Session(models_dir, settings, effects))
        return sessions[-1]

    yield make
    for session in sessions:
        session.pipeline.close()


def _face_mask(landmarks: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(landmarks[FACE_OVAL]).astype(np.int32)], 1)
    return mask.astype(bool)


def _face_change(before: np.ndarray, after: np.ndarray, landmarks: np.ndarray) -> float:
    """Mean absolute change (0..255) inside the face oval."""
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).mean(axis=2)
    return float(diff[_face_mask(landmarks, before.shape)].mean())


def _corners(image: np.ndarray) -> list[np.ndarray]:
    c = CORNER_PX
    return [image[:c, :c], image[:c, -c:], image[-c:, :c], image[-c:, -c:]]


def test_all_builtin_effects_make_up_the_face_and_leave_the_background(
    session_factory: SessionFactory, face_frame_720p: np.ndarray
) -> None:
    problems: list[str] = []
    effects = [_checked(effect_cls, problems) for effect_cls in load_builtin_effects()]
    result = session_factory(MakeupSettings(), effects).run(face_frame_720p, WARMUP_FRAMES)

    assert result.face_found
    assert result.landmarks is not None
    assert problems == [], f"effects returned non-finite or out-of-range values: {sorted(set(problems))}"
    assert _face_change(face_frame_720p, result.output, result.landmarks) > 1.0
    for before, after in zip(_corners(face_frame_720p), _corners(result.output), strict=True):
        assert np.abs(after.astype(np.int16) - before.astype(np.int16)).max() <= 1

    expected_stages = {"landmarks", "segmentation", "total"} | {effect.name for effect in effects}
    assert expected_stages <= set(result.timings_ms)
    assert all(value >= 0.0 for value in result.timings_ms.values())
    assert result.timings_ms["total"] >= result.timings_ms["landmarks"]


def test_every_builtin_preset_loads_and_runs(session_factory: SessionFactory, face_frame_720p: np.ndarray) -> None:
    builtin = [info.name for info in list_presets() if info.builtin]
    assert {"natural", "bare", "polished", "strong", "off"} <= set(builtin)

    session = session_factory(MakeupSettings())
    session.run(face_frame_720p, WARMUP_FRAMES)
    change: dict[str, float] = {}
    for name in builtin:
        session.pipeline.set_settings(load_preset(name))
        result = session.run(face_frame_720p, 4)
        assert result.face_found, name
        assert result.landmarks is not None
        change[name] = _face_change(face_frame_720p, result.output, result.landmarks)
        if name == "off":
            assert np.array_equal(result.output, face_frame_720p)

    assert change["off"] == 0.0
    assert 0.0 < change["natural"] < change["polished"]
    assert change["bare"] > 0.0
    assert change["strong"] > change["natural"]


@pytest.mark.parametrize("name", ["natural", "bare", "polished", "strong"])
def test_presets_run_all_effects_without_failures(
    session_factory: SessionFactory,
    face_frame_720p: np.ndarray,
    caplog: pytest.LogCaptureFixture,
    name: str,
) -> None:
    # The pipeline isolates effect exceptions, so a broken effect would otherwise go unnoticed.
    problems: list[str] = []
    effects = [_checked(effect_cls, problems) for effect_cls in load_builtin_effects()]
    settings = load_preset(name)
    with caplog.at_level(logging.ERROR, logger="makeup_cam"):
        result = session_factory(settings, effects).run(face_frame_720p, WARMUP_FRAMES)

    assert [record.getMessage() for record in caplog.records] == []
    assert problems == []
    assert result.landmarks is not None
    assert _face_change(face_frame_720p, result.output, result.landmarks) > 0.0
    for before, after in zip(_corners(face_frame_720p), _corners(result.output), strict=True):
        assert np.abs(after.astype(np.int16) - before.astype(np.int16)).max() <= 1


def test_every_effect_enabled_with_all_sliders_at_zero_is_bit_identical(
    session_factory: SessionFactory, face_frame_720p: np.ndarray
) -> None:
    settings = MakeupSettings()
    for section in SECTIONS:
        if section == "general":
            continue
        values = getattr(settings, section)
        for f in fields(values):
            if f.metadata["kind"] == "float" and f.metadata["min"] <= 0.0:
                setattr(values, f.name, 0.0)
    result = session_factory(settings).run(face_frame_720p, WARMUP_FRAMES)
    assert result.face_found
    assert np.array_equal(result.output, face_frame_720p)


def test_makeup_fades_out_when_the_face_leaves_and_comes_back(
    session_factory: SessionFactory, face_frame_720p: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    # A bow: the face leaves the frame, and later returns. Nothing may fail or jump.
    empty = np.full_like(face_frame_720p, 128)
    session = session_factory(load_preset("strong"))
    with caplog.at_level(logging.ERROR, logger="makeup_cam"):
        session.run(face_frame_720p, WARMUP_FRAMES)
        left = session.run(empty, 12)
        assert not left.face_found
        assert np.abs(left.output.astype(np.int16) - empty).max() <= 1
        back = session.run(face_frame_720p, WARMUP_FRAMES)
    assert back.face_found and back.landmarks is not None
    assert _face_change(face_frame_720p, back.output, back.landmarks) > 1.0
    assert [record.getMessage() for record in caplog.records] == []


def test_bypass_returns_the_original_frame(session_factory: SessionFactory, face_frame_720p: np.ndarray) -> None:
    session = session_factory(load_preset("natural"))
    made_up = session.run(face_frame_720p, WARMUP_FRAMES)
    assert not np.array_equal(made_up.output, face_frame_720p)

    session.pipeline.bypass = True
    result = session.run(face_frame_720p, 1)
    assert np.array_equal(result.output, face_frame_720p)
    assert result.face_found  # tracking keeps running so the GUI still shows the face status


def test_a_deep_bow_leaves_no_makeup_on_the_wall(session_factory: SessionFactory, face_frame_720p: np.ndarray) -> None:
    """The real model loses the face while it moves fast. Whatever is drawn then must stay on the person."""
    height, width = face_frame_720p.shape[:2]
    wall = np.full_like(face_frame_720p, (176, 190, 198))
    session = session_factory(load_preset("polished"))
    session.run(face_frame_720p, WARMUP_FRAMES)
    person = (np.abs(face_frame_720p.astype(np.int16) - face_frame_720p[0, 0]).max(axis=2) > 12).astype(np.uint8)

    def bowed(dy: float) -> tuple[np.ndarray, np.ndarray]:
        move = np.float32([[1, 0, 0], [0, 1, dy]])
        frame = cv2.warpAffine(face_frame_720p, move, (width, height), borderMode=cv2.BORDER_REPLICATE)
        here = cv2.warpAffine(person, move, (width, height)) > 0
        frame[:] = np.where(here[..., None], frame, wall)
        return frame, here

    lost = 0
    offsets = [760.0 * (i / 20.0) ** 2 for i in range(21)]
    for dy in [*offsets, *offsets[::-1]]:
        frame, here = bowed(dy)
        result = session.run(frame, 1)
        lost += not result.face_found
        far_from_person = cv2.dilate(here.astype(np.uint8), np.ones((81, 81), np.uint8)) == 0
        change = cv2.blur(np.abs(result.output.astype(np.int16) - frame.astype(np.int16)).astype(np.float32).mean(axis=2), (5, 5))
        assert change[far_from_person].max() < 3.0, f"makeup on the wall at offset {dy:.0f}"
    assert lost > 3  # the bow did take the face away

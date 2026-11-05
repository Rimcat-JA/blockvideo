"""LipEffect on the astronaut portrait (real landmarks): no-ops, locality,
teeth protection, strength scaling, color, finishes and odd geometry."""

from __future__ import annotations

from dataclasses import fields

import cv2
import numpy as np
import pytest

from makeup_cam import blend, regions
from makeup_cam.config import LipSettings, MakeupSettings
from makeup_cam.effects import lips as lips_module
from makeup_cam.effects.base import registered_effects
from makeup_cam.effects.lips import LipEffect

FINISHES = [value for value, _ in next(f for f in fields(LipSettings) if f.name == "finish").metadata["choices"]]
# Everything at full strength (finish chosen per test).
STRONG = {"opacity": 1.0, "gloss": 1.0, "feather": 1.0, "outline": 0.2}


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


def apply_lips(
    frame: np.ndarray, landmarks: np.ndarray | None, fake_ctx, intensity: float = 1.0, **values
) -> np.ndarray:
    ctx = fake_ctx(frame, landmarks, intensity=intensity)
    return LipEffect().apply(frame.copy(), ctx, LipSettings(**values))


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def lip_core(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels well inside the lips."""
    return regions.lips_mask(landmarks, full_roi(frame), outline=-0.15) > 0.99


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def test_registered_under_its_settings_section() -> None:
    assert LipEffect in registered_effects()
    assert (LipEffect.name, LipEffect.order, LipEffect.needs_face) == ("lips", 70, True)
    assert isinstance(getattr(MakeupSettings(), LipEffect.name), LipSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "use_landmarks"),
    [
        (1.0, {"opacity": 0.0, "finish": "gloss", "gloss": 1.0}, True),
        (0.0, {"opacity": 1.0, "finish": "gloss", "gloss": 1.0}, True),
        (1.0, {"opacity": 1.0}, False),
    ],
    ids=["opacity-0", "intensity-0", "no-face"],
)
def test_exact_noop(frame, face_landmarks_720p, fake_ctx, intensity, values, use_landmarks) -> None:
    landmarks = face_landmarks_720p if use_landmarks else None
    out = apply_lips(frame, landmarks, fake_ctx, intensity=intensity, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


@pytest.mark.parametrize("finish", FINISHES)
def test_output_is_a_valid_frame_for_every_finish(frame, face_landmarks_720p, fake_ctx, finish) -> None:
    out = apply_lips(frame, face_landmarks_720p, fake_ctx, finish=finish, **STRONG)
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(frame, out)[lip_core(frame, face_landmarks_720p)].mean() > 0.01


def test_changes_stay_on_the_lips(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_lips(frame, face_landmarks_720p, fake_ctx, finish="gloss", **STRONG)
    diff = change(frame, out)
    scale = regions.face_scale(face_landmarks_720p)
    near_lips = regions.roi_around(face_landmarks_720p[regions.LIPS_OUTER], frame.shape, pad=0.0, margin_px=0.3 * scale)
    outside = np.ones(diff.shape, bool)
    outside[near_lips.slices] = False
    assert diff[outside].max() == 0.0
    assert np.array_equal(out[:50, :50], frame[:50, :50])  # background corner
    assert diff[lip_core(frame, face_landmarks_720p)].mean() > 0.05


@pytest.mark.parametrize("finish", FINISHES)
def test_teeth_are_never_touched(frame, face_landmarks_720p, fake_ctx, finish) -> None:
    out = apply_lips(frame, face_landmarks_720p, fake_ctx, finish=finish, **STRONG)
    inner = regions.expand_polygon(face_landmarks_720p[regions.LIPS_INNER], -1.0)
    teeth = regions.polygon_mask(inner, full_roi(frame)) > 0.99
    assert teeth.sum() > 50  # the portrait smiles with open lips
    assert change(frame, out)[teeth].max() == 0.0


def test_strength_grows_with_intensity_and_opacity(frame, face_landmarks_720p, fake_ctx) -> None:
    core = lip_core(frame, face_landmarks_720p)

    def strength(intensity: float, opacity: float) -> float:
        out = apply_lips(frame, face_landmarks_720p, fake_ctx, intensity=intensity, opacity=opacity, color="#8E2F45")
        return float(change(frame, out)[core].mean())

    by_intensity = [strength(i, 0.8) for i in (0.25, 0.5, 1.0)]
    by_opacity = [strength(1.0, o) for o in (0.2, 0.5, 1.0)]
    assert by_intensity == sorted(by_intensity) and by_intensity[0] > 0.0
    assert by_opacity == sorted(by_opacity) and by_opacity[0] > 0.0
    assert len(set(by_intensity)) == 3 and len(set(by_opacity)) == 3


def lip_lab(image: np.ndarray, core: np.ndarray) -> np.ndarray:
    return blend.bgr_to_lab(image)[core].mean(axis=0)


def test_lipstick_sets_hue_and_lightness(frame, face_landmarks_720p, fake_ctx) -> None:
    core = lip_core(frame, face_landmarks_720p)
    original = lip_lab(frame, core)
    dark = lip_lab(apply_lips(frame, face_landmarks_720p, fake_ctx, opacity=1.0, color="#5A1A2A"), core)
    pale = lip_lab(apply_lips(frame, face_landmarks_720p, fake_ctx, opacity=1.0, color="#EBC0B4"), core)
    coral = lip_lab(apply_lips(frame, face_landmarks_720p, fake_ctx, opacity=1.0, color="#E8603C"), core)
    mauve = lip_lab(apply_lips(frame, face_landmarks_720p, fake_ctx, opacity=1.0, color="#B05090"), core)
    assert dark[0] < original[0] - 10 < original[0] + 5 < pale[0]
    assert coral[2] > original[2] + 10  # yellower (orange)
    assert mauve[2] < original[2] - 10  # bluer (purple)


def test_finish_controls_the_highlights(frame, face_landmarks_720p, fake_ctx) -> None:
    core = lip_core(frame, face_landmarks_720p)

    def top_luminance(finish: str, gloss: float) -> float:
        out = apply_lips(frame, face_landmarks_720p, fake_ctx, opacity=0.6, finish=finish, gloss=gloss)
        return float(np.percentile(blend.luminance(out)[core], 98))

    matte = top_luminance("matte", 1.0)
    satin_off = top_luminance("satin", 0.0)
    satin = top_luminance("satin", 1.0)
    gloss = top_luminance("gloss", 1.0)
    assert matte < satin_off < satin < gloss


def test_closed_mouth_paints_the_whole_lips(frame, face_landmarks_720p, fake_ctx) -> None:
    closed = face_landmarks_720p.copy()
    seam = (closed[regions.LIPS_UPPER_INNER] + closed[regions.LIPS_LOWER_INNER]) / 2
    closed[regions.LIPS_UPPER_INNER] = seam
    closed[regions.LIPS_LOWER_INNER] = seam
    out = apply_lips(frame, closed, fake_ctx, opacity=0.8, finish="gloss", gloss=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    x, y = np.round(seam[len(seam) // 2]).astype(int)
    assert change(frame, out)[y, x] > 0.01


def test_lips_at_and_beyond_the_frame_border(frame, face_landmarks_720p, fake_ctx) -> None:
    lips_bottom = face_landmarks_720p[regions.LIPS_OUTER][:, 1].max()
    partly = face_landmarks_720p + np.float32([0.0, frame.shape[0] - lips_bottom + 5.0])
    out = apply_lips(frame, partly, fake_ctx, opacity=1.0, finish="gloss", gloss=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(frame, out)[: frame.shape[0] - 60].max() == 0.0
    gone = face_landmarks_720p + np.float32([0.0, 2000.0])
    assert np.array_equal(apply_lips(frame, gone, fake_ctx, opacity=1.0), frame)


def test_lipstick_takes_on_a_warm_light(frame, face_landmarks_720p, fake_ctx) -> None:
    """A warm cast on the face (b* up) also warms the lipstick, as real light would."""
    warm = np.clip(frame * np.float32([0.8, 0.97, 1.08]), 0.0, 1.0)
    core = lip_core(frame, face_landmarks_720p)
    skin = regions.face_oval_mask(face_landmarks_720p, full_roi(frame), expand_px=-20.0) > 0.99
    skin_shift = blend.bgr_to_lab(warm)[skin][:, 2].mean() - blend.bgr_to_lab(frame)[skin][:, 2].mean()
    neutral_lips = lip_lab(apply_lips(frame, face_landmarks_720p, fake_ctx, opacity=1.0), core)
    warm_lips = lip_lab(apply_lips(warm, face_landmarks_720p, fake_ctx, opacity=1.0), core)
    assert skin_shift > 8.0
    assert warm_lips[2] - neutral_lips[2] > 0.5 * skin_shift


def zoomed(frame: np.ndarray, landmarks: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """The frame and landmarks magnified around the mouth (a face close to the camera)."""
    center = landmarks[regions.LIPS_OUTER].mean(axis=0)
    shift = np.float32([frame.shape[1] / 2, frame.shape[0] / 2]) - zoom * center
    matrix = np.float32([[zoom, 0, shift[0]], [0, zoom, shift[1]]])
    big = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    return big, (landmarks * zoom + shift).astype(np.float32)


def test_close_up_faces_are_made_up_at_reduced_resolution(frame, face_landmarks_720p, fake_ctx, monkeypatch) -> None:
    big, landmarks = zoomed(frame, face_landmarks_720p, 3.0)
    assert regions.face_scale(landmarks) > 1.5 * lips_module._MAX_WORK_SCALE
    values = {"opacity": 0.8, "finish": "gloss", "gloss": 1.0, "color": "#8E2F45"}
    fast = apply_lips(big, landmarks, fake_ctx, **values)
    monkeypatch.setattr(lips_module, "_MAX_WORK_SCALE", 1e9)
    exact = apply_lips(big, landmarks, fake_ctx, **values)
    core = lip_core(big, landmarks)
    inner = regions.expand_polygon(landmarks[regions.LIPS_INNER], -1.0)
    teeth = regions.polygon_mask(inner, full_roi(big)) > 0.99
    assert fast.min() >= 0.0 and fast.max() <= 1.0
    assert change(big, fast)[teeth].max() == 0.0
    assert change(big, fast)[core].mean() > 0.05
    assert np.abs(fast - exact)[core].mean() < 0.01


def test_unknown_finish_is_rejected(frame, face_landmarks_720p, fake_ctx) -> None:
    with pytest.raises(ValueError, match="finish"):
        apply_lips(frame, face_landmarks_720p, fake_ctx, finish="sparkle")

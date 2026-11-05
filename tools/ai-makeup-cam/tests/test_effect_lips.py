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


# ---------------------------------------------------------------------------
# The picture has the last word about the teeth
# ---------------------------------------------------------------------------

LIP_COLOR = (90 / 255, 80 / 255, 185 / 255)  # BGR, a red lip
SKIN_COLOR = (130 / 255, 160 / 255, 205 / 255)
TOOTH_COLOR = (175 / 255, 225 / 255, 235 / 255)


def painted_mouth(landmarks: np.ndarray, *, lip: tuple[float, float, float] = LIP_COLOR) -> np.ndarray:
    """A flat mouth: skin, lips, and a white band of teeth right at the mouth opening."""
    image = np.empty((720, 1280, 3), np.float32)
    image[...] = SKIN_COLOR
    roi = full_roi(image)
    lips = regions.polygon_mask(landmarks[regions.LIPS_OUTER], roi) > 0.5
    image[lips] = lip
    opening = regions.polygon_mask(landmarks[regions.LIPS_INNER], roi) > 0.5
    image[opening] = (0.05, 0.05, 0.08)
    return image


def teeth_below_the_opening(image: np.ndarray, landmarks: np.ndarray, depth: float) -> np.ndarray:
    """Paint teeth from the opening down over the lower lip's mask: the inner lip line is too high,
    as it is for a few frames when the mouth opens. Returns where the teeth are outside the opening."""
    roi = full_roi(image)
    scale = regions.face_scale(landmarks)
    opening = regions.polygon_mask(landmarks[regions.LIPS_INNER], roi) > 0.5
    grown = regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_INNER], depth * scale), roi) > 0.5
    image[grown] = TOOTH_COLOR
    return grown & ~opening


def test_teeth_inside_the_lip_mask_stay_white(face_landmarks_720p, fake_ctx) -> None:
    image = painted_mouth(face_landmarks_720p)
    teeth = teeth_below_the_opening(image, face_landmarks_720p, 0.06)
    out = LipEffect().apply(image.copy(), fake_ctx(image, face_landmarks_720p), LipSettings(opacity=1.0, finish="matte", color="#7E2A5A", feather=0.0))
    core = cv2.erode(teeth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert core.sum() > 15
    assert change(image, out)[core].max() < 0.02
    lips = (regions.lips_mask(face_landmarks_720p, full_roi(image)) > 0.99) & ~cv2.dilate(teeth.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert change(image, out)[lips].mean() > 0.03  # and the lips still get their color


def test_a_pale_lower_lip_is_not_taken_for_teeth(face_landmarks_720p, fake_ctx) -> None:
    """On dark skin the lower lip can be much paler than the upper one, nearly as pale as skin."""
    lm = face_landmarks_720p
    image = painted_mouth(lm, lip=(60 / 255, 70 / 255, 120 / 255))
    image[...] *= 0.55  # dark skin and lips, not the white of teeth
    roi = full_roi(image)
    lower = np.zeros((720, 1280), bool)
    mouth_y = float(lm[regions.LIPS_INNER, 1].mean())
    lower[int(mouth_y):] = True
    pale = (regions.lips_mask(lm, roi) > 0.5) & lower
    image[pale] = (120 / 255, 145 / 255, 175 / 255)
    teeth_below_the_opening(image, lm, 0.0)
    image[regions.polygon_mask(lm[regions.LIPS_INNER], roi) > 0.5] = TOOTH_COLOR
    out = LipEffect().apply(image.copy(), fake_ctx(image, lm), LipSettings(opacity=1.0, finish="matte", color="#B5525E", feather=0.0))
    core = pale & (regions.polygon_mask(regions.expand_polygon(lm[regions.LIPS_INNER], 2.0), roi) < 0.5)
    assert core.sum() > 50
    assert change(image, out)[core].mean() > 0.03  # no hole in the lipstick


def test_shine_on_the_lips_keeps_the_lipstick(face_landmarks_720p, fake_ctx) -> None:
    """Bright and pale like teeth, but cut off from the mouth opening by lip."""
    lm = face_landmarks_720p
    image = painted_mouth(lm)
    scale = regions.face_scale(lm)
    roi = full_roi(image)
    center = (lm[regions.LIPS_INNER_BOTTOM] + lm[regions.LIPS_BOTTOM]) / 2.0
    spot = regions.ellipse_mask(center, (0.06 * scale, 0.02 * scale), 0.0, roi) > 0.5
    image[spot] = TOOTH_COLOR
    image[regions.polygon_mask(lm[regions.LIPS_INNER], roi) > 0.5] = TOOTH_COLOR
    out = LipEffect().apply(image.copy(), fake_ctx(image, lm), LipSettings(opacity=1.0, finish="matte", color="#B5525E", feather=0.0))
    assert spot.sum() > 10
    assert change(image, out)[spot].mean() > 0.02


def test_dark_lips_are_not_lightened_much() -> None:
    limit = lips_module._limit_lightening
    assert limit(20.0, 30.0) == pytest.approx(lips_module._MAX_LIGHTEN)  # dark lips
    assert limit(20.0, 70.0) == pytest.approx(20.0)  # light lips: a nude lightens them
    assert lips_module._MAX_LIGHTEN < limit(20.0, 52.0) < 20.0
    assert limit(-15.0, 30.0) == pytest.approx(-15.0)  # darkening is never limited


def lab_of(image: np.ndarray) -> np.ndarray:
    return blend.bgr_to_lab(image)


def test_lower_teeth_with_the_lips_color_bled_into_them_stay_white(face_landmarks_720p, fake_ctx) -> None:
    """Small lower teeth in shade: hardly lighter than the lip, and webcams blur its red into them."""
    lm = face_landmarks_720p
    image = painted_mouth(lm)
    teeth = teeth_below_the_opening(image, lm, 0.06)
    lip = np.float32(LIP_COLOR)
    image[teeth] = 0.55 * lip + 0.45 * np.float32(TOOTH_COLOR)  # as light as the lip at most, half way to its red
    image = cv2.GaussianBlur(image, (0, 0), 1.0)
    before = lab_of(image)
    out = LipEffect().apply(image.copy(), fake_ctx(image, lm), LipSettings(opacity=1.0, finish="satin", gloss=0.0, color="#7E2A5A", feather=0.0))
    after = lab_of(out)
    core = cv2.erode(teeth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert core.sum() > 15
    outer = regions.lips_mask(lm, full_roi(image)) > 0.99
    outer &= ~cv2.dilate(teeth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    painted = np.abs(after - before).max(axis=2)
    assert painted[outer].mean() > 3.0 * painted[core].mean()


def test_each_lip_follows_the_lipstick_on_its_own(face_landmarks_720p, fake_ctx) -> None:
    """A dark upper and a pale lower lip must not split into a deep red and a salmon pink."""
    lm = face_landmarks_720p
    image = painted_mouth(lm, lip=(40 / 255, 40 / 255, 90 / 255))
    roi = full_roi(image)
    lips = regions.polygon_mask(lm[regions.LIPS_OUTER], roi) > 0.5
    lower = lips & (regions.polygon_mask(lm[regions.LIPS_LOWER_OUTER + regions.LIPS_LOWER_INNER[::-1]], roi) > 0.5)
    upper = lips & (regions.polygon_mask(lm[regions.LIPS_UPPER_OUTER + regions.LIPS_UPPER_INNER[::-1]], roi) > 0.5)
    image[lower] = (130 / 255, 145 / 255, 200 / 255)
    before = lab_of(image)[..., 0]
    out = LipEffect().apply(image.copy(), fake_ctx(image, lm), LipSettings(opacity=1.0, finish="satin", gloss=0.0, color="#B5525E", feather=0.0))
    after = lab_of(out)[..., 0]
    core = regions.lips_mask(lm, roi, outline=-0.15) > 0.99
    gap_before = float(before[lower & core].mean() - before[upper & core].mean())
    gap_after = float(after[lower & core].mean() - after[upper & core].mean())
    assert gap_before > 20.0
    assert gap_after < 0.6 * gap_before
    assert after[lower & core].mean() < before[lower & core].mean()  # the pale lip takes the pigment's depth
    assert after[upper & core].mean() - before[upper & core].mean() <= lips_module._MAX_LIGHTEN + 0.5


def open_mouth(lm: np.ndarray, factor: float) -> np.ndarray:
    """``lm`` with the lower lip moved down, as far as ``factor`` x the width of the mouth."""
    out = lm.copy()
    width = float(np.linalg.norm(lm[regions.MOUTH_LEFT_CORNER] - lm[regions.MOUTH_RIGHT_CORNER]))
    out[regions.LIPS_LOWER_INNER[1:-1] + regions.LIPS_LOWER_OUTER[1:-1], 1] += factor * width
    return out


def test_a_wide_open_mouth_gets_no_red_ring(face_landmarks_720p) -> None:
    lm = open_mouth(face_landmarks_720p, 0.7)
    roi = regions.Roi(400, 200, 900, 600)
    scale = regions.face_scale(lm)
    guard = lips_module._thin_guard(lm, roi, scale)
    assert guard.min() == pytest.approx(lips_module._THIN_FLOOR, abs=0.1)


def test_a_mouth_that_is_not_wide_open_keeps_all_of_its_pigment(face_landmarks_720p) -> None:
    guard = lips_module._thin_guard(face_landmarks_720p, regions.Roi(400, 200, 900, 600), regions.face_scale(face_landmarks_720p))
    assert guard == 1.0

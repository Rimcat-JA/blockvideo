"""EyeshadowEffect and EyelinerEffect on the astronaut portrait (real landmarks)
and on flat frames: no-ops, locality, eyeballs never painted, placement of the
shadow and the liner (lash line, taper, wing, lower line), blinking, head
roll, frame borders and a run through the real pipeline."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from makeup_cam import blend, regions
from makeup_cam.config import EyelinerSettings, EyeshadowSettings, MakeupSettings
from makeup_cam.effects.base import Effect, registered_effects
from makeup_cam.effects.eyes import EyelinerEffect, EyeshadowEffect

INVISIBLE = 1.0 / 255.0  # below one 8-bit output step
GRAY = 0.7
SETTINGS = {EyeshadowEffect: EyeshadowSettings, EyelinerEffect: EyelinerSettings}
EFFECTS = list(SETTINGS)
STRONG = {
    EyeshadowEffect: {"opacity": 1.0, "height": 1.0, "softness": 1.0, "shimmer": 1.0},
    EyelinerEffect: {"opacity": 1.0, "thickness": 1.0, "wing": 1.0, "lower": 1.0, "softness": 1.0},
}


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


@pytest.fixture
def gray_frame() -> np.ndarray:
    """A flat frame: the change is proportional to the makeup mask (for geometry checks)."""
    return np.full((720, 1280, 3), GRAY, np.float32)


def apply(
    effect_cls: type[Effect],
    frame: np.ndarray,
    landmarks: np.ndarray | None,
    fake_ctx,
    intensity: float = 1.0,
    **values,
) -> np.ndarray:
    ctx = fake_ctx(frame, landmarks, intensity=intensity)
    return effect_cls().apply(frame.copy(), ctx, SETTINGS[effect_cls](**values))


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def eye_frame(landmarks: np.ndarray, side: regions.Side) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(inner corner, unit inner->outer, unit up toward the brow, eye width)."""
    s = regions.side_indices(side)
    inner, outer = landmarks[s.eye_upper[0]], landmarks[s.eye_upper[-1]]
    along = (outer - inner) / np.linalg.norm(outer - inner)
    up = np.array([along[1], -along[0]])
    if np.dot(landmarks[s.brow].mean(axis=0) - (inner + outer) / 2, up) < 0:
        up = -up
    return inner, along, up, float(np.linalg.norm(outer - inner))


def pixel_coords(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    return xs + 0.5, ys + 0.5


def side_half(diff: np.ndarray, landmarks: np.ndarray, side: regions.Side) -> np.ndarray:
    """``diff`` with the other half of the face zeroed (the subject's left is on the image right)."""
    split = int(round(float(landmarks[regions.NOSE_TIP][0])))
    out = diff.copy()
    if side == "left":
        out[:, :split] = 0
    else:
        out[:, split:] = 0
    return out


def under_eye(landmarks: np.ndarray, side: regions.Side, shape: tuple[int, ...]) -> np.ndarray:
    """Bool mask of the skin under the middle of the eye (the corners may get a soft wrap or the wing)."""
    inner, along, _, width = eye_frame(landmarks, side)
    xs, ys = pixel_coords(shape[:2])
    u = ((xs - inner[0]) * along[0] + (ys - inner[1]) * along[1]) / width
    zone = regions.under_eye_mask(landmarks, side, regions.Roi(0, 0, shape[1], shape[0])) > 0.5
    return zone & (u > 0.15) & (u < 0.85)


def centroid(weights: np.ndarray) -> np.ndarray:
    xs, ys = pixel_coords(weights.shape)
    total = weights.sum()
    return np.array([(xs * weights).sum() / total, (ys * weights).sum() / total])


def sample(diff: np.ndarray, point: np.ndarray) -> float:
    x, y = np.floor(point).astype(int)
    return float(diff[y, x])


def closed_eyes(landmarks: np.ndarray) -> np.ndarray:
    """Landmarks with both upper lids lowered onto the lower lids (a blink)."""
    closed = landmarks.copy()
    for side in regions.SIDES:
        s = regions.side_indices(side)
        closed[s.eye_upper] = landmarks[s.eye_lower]
    return closed


# --------------------------------------------------------------------------
# Contract.
# --------------------------------------------------------------------------


def test_registered_under_their_settings_sections() -> None:
    for effect_cls, name, order in ((EyeshadowEffect, "eyeshadow", 40), (EyelinerEffect, "eyeliner", 50)):
        assert effect_cls in registered_effects()
        assert (effect_cls.name, effect_cls.order, effect_cls.needs_face, effect_cls.needs_hair_mask) == (
            name,
            order,
            True,
            False,
        )
        assert isinstance(getattr(MakeupSettings(), name), SETTINGS[effect_cls])


@pytest.mark.parametrize("effect_cls", EFFECTS)
@pytest.mark.parametrize(
    ("intensity", "zeroed", "use_landmarks"),
    [(1.0, {"opacity": 0.0}, True), (0.0, {}, True), (1.0, {}, False)],
    ids=["opacity-0", "intensity-0", "no-face"],
)
def test_exact_noop(frame, face_landmarks_720p, fake_ctx, effect_cls, intensity, zeroed, use_landmarks) -> None:
    values = {**STRONG[effect_cls], **zeroed}
    landmarks = face_landmarks_720p if use_landmarks else None
    out = apply(effect_cls, frame, landmarks, fake_ctx, intensity=intensity, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


def test_zero_height_eyeshadow_is_a_noop(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply(EyeshadowEffect, frame, face_landmarks_720p, fake_ctx, opacity=1.0, height=0.0, shimmer=1.0)
    assert np.array_equal(out, frame)


def _extremes(settings_cls: type) -> list[dict]:
    """All sliders at their minimum (opacity kept on), all at their maximum, and mixed."""
    floats = [f for f in fields(settings_cls) if f.metadata.get("kind") == "float"]
    low = {f.name: f.metadata["min"] for f in floats} | {"opacity": 1.0}
    high = {f.name: f.metadata["max"] for f in floats}
    mixed = {f.name: f.metadata["min" if i % 2 else "max"] for i, f in enumerate(floats)} | {"opacity": 0.6}
    return [low, high, mixed]


@pytest.mark.parametrize("effect_cls", EFFECTS)
@pytest.mark.parametrize("variant", range(3), ids=["all-min", "all-max", "mixed"])
@pytest.mark.parametrize("color", ["#000000", "#FFFFFF", "#6E3B5C"])
def test_every_setting_gives_a_valid_frame(frame, face_landmarks_720p, fake_ctx, effect_cls, variant, color) -> None:
    values = _extremes(SETTINGS[effect_cls])[variant] | {"color": color}
    out = apply(effect_cls, frame, face_landmarks_720p, fake_ctx, **values)
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("effect_cls", EFFECTS)
def test_defaults_are_visible_but_subtle(frame, face_landmarks_720p, fake_ctx, effect_cls) -> None:
    diff = change(frame, apply(effect_cls, frame, face_landmarks_720p, fake_ctx))
    assert 6 * INVISIBLE < diff.max() < 0.2


@pytest.mark.parametrize("effect_cls", EFFECTS)
def test_strength_grows_with_intensity_and_opacity(frame, face_landmarks_720p, fake_ctx, effect_cls) -> None:
    def strength(intensity: float, opacity: float) -> float:
        out = apply(effect_cls, frame, face_landmarks_720p, fake_ctx, intensity, opacity=opacity)
        return float(change(frame, out).sum())

    by_intensity = [strength(i, 0.8) for i in (0.25, 0.5, 1.0)]
    by_opacity = [strength(1.0, o) for o in (0.2, 0.5, 1.0)]
    for series in (by_intensity, by_opacity):
        assert series == sorted(series) and len(set(series)) == 3 and series[0] > 0


@pytest.mark.parametrize("effect_cls", EFFECTS)
def test_changes_stay_around_the_eyes(frame, face_landmarks_720p, fake_ctx, effect_cls) -> None:
    out = apply(effect_cls, frame, face_landmarks_720p, fake_ctx, **STRONG[effect_cls])
    eye_area = [regions.LEFT_EYE, regions.RIGHT_EYE, regions.LEFT_BROW, regions.RIGHT_BROW]
    box = regions.roi_around(face_landmarks_720p[sum(eye_area, [])], frame.shape, pad=0.15)
    outside = np.ones(frame.shape[:2], bool)
    outside[box.slices] = False
    assert np.array_equal(out[outside], frame[outside])
    # The flat gray margins of the frame are background: bit-identical.
    assert np.array_equal(out[:, :280], frame[:, :280]) and np.array_equal(out[:, 1000:], frame[:, 1000:])


@pytest.mark.parametrize("effect_cls", EFFECTS)
@pytest.mark.parametrize("softness", [0.0, 1.0])
def test_eyeballs_are_never_painted(frame, face_landmarks_720p, fake_ctx, effect_cls, softness) -> None:
    values = STRONG[effect_cls] | {"softness": softness, "color": "#000000"}
    diff = change(frame, apply(effect_cls, frame, face_landmarks_720p, fake_ctx, **values))
    for side in regions.SIDES:
        opening = regions.eye_opening_mask(face_landmarks_720p, side, full_roi(frame))
        assert diff[opening >= 1.0].max() == 0.0


# --------------------------------------------------------------------------
# Eyeshadow placement.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", regions.SIDES)
def test_shadow_sits_on_the_upper_lid(gray_frame, face_landmarks_720p, fake_ctx, side) -> None:
    lm = face_landmarks_720p
    diff = side_half(change(gray_frame, apply(EyeshadowEffect, gray_frame, lm, fake_ctx, opacity=1.0)), lm, side)
    inner, along, up, width = eye_frame(lm, side)
    s = regions.side_indices(side)
    lid_depth = float(np.dot(lm[s.brow_lower].mean(axis=0) - lm[s.eye_upper].mean(axis=0), up))
    offset = centroid(diff) - lm[s.eye_upper].mean(axis=0)
    assert 0.0 < np.dot(offset, up) < 0.6 * lid_depth  # above the lashes, below the brow
    assert 0.3 < np.dot(centroid(diff) - inner, along) / width < 1.0  # toward the outer corner
    assert diff[under_eye(lm, side, gray_frame.shape)].max() < INVISIBLE


@pytest.mark.parametrize("side", regions.SIDES)
def test_shadow_is_deepest_at_the_lash_line_and_outer_corner(gray_frame, face_landmarks_720p, fake_ctx, side) -> None:
    lm = face_landmarks_720p
    out = apply(EyeshadowEffect, gray_frame, lm, fake_ctx, opacity=1.0, height=1.0, softness=1.0)
    diff = change(gray_frame, out)
    s = regions.side_indices(side)
    lash = regions.resample_polyline(lm[s.eye_upper], 11)
    brow = regions.resample_polyline(lm[s.brow_lower], 11)

    def at(along_t: int, up_t: float) -> float:
        return sample(diff, lash[along_t] + up_t * (brow[along_t] - lash[along_t]))

    for k in (3, 5, 8):
        assert at(k, 0.15) > at(k, 0.6) > at(k, 0.95)
    assert at(8, 0.15) > at(2, 0.15)


def test_height_raises_the_shadow(gray_frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    _, _, up, _ = eye_frame(lm, "left")
    lash = lm[regions.LEFT_EYE_UPPER].mean(axis=0)

    def reach(height: float) -> tuple[float, int]:
        out = apply(EyeshadowEffect, gray_frame, lm, fake_ctx, opacity=1.0, height=height)
        diff = side_half(change(gray_frame, out), lm, "left")
        return float(np.dot(centroid(diff) - lash, up)), int((diff > 0.02).sum())

    (low_rise, low_area), (high_rise, high_area) = reach(0.3), reach(0.9)
    assert high_rise > low_rise + 0.03 * regions.face_scale(lm)
    assert high_area > 1.5 * low_area


def test_shimmer_adds_a_sheen_that_follows_the_light(frame, face_landmarks_720p, fake_ctx) -> None:
    lm = face_landmarks_720p
    s = regions.side_indices("left")
    lash = regions.resample_polyline(lm[s.eye_upper], 9)[4]
    brow = regions.resample_polyline(lm[s.brow_lower], 9)[4]
    center = lash + 0.12 * (brow - lash)  # a third of the way up the default shadow height
    patch = (slice(int(center[1]) - 1, int(center[1]) + 2), slice(int(center[0]) - 1, int(center[0]) + 2))

    def sheen(image: np.ndarray) -> float:
        matte = apply(EyeshadowEffect, image, lm, fake_ctx, shimmer=0.0)
        shiny = apply(EyeshadowEffect, image, lm, fake_ctx, shimmer=1.0)
        return float((blend.luminance(shiny)[patch] - blend.luminance(matte)[patch]).mean())

    bright, dim = sheen(frame), sheen(frame * np.float32(0.35))
    assert bright > 0.03
    assert 0.0 < dim < 0.6 * bright


# --------------------------------------------------------------------------
# Eyeliner placement.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", regions.SIDES)
def test_liner_sits_on_the_upper_lash_line(gray_frame, face_landmarks_720p, fake_ctx, side) -> None:
    lm = face_landmarks_720p
    diff = change(gray_frame, apply(EyelinerEffect, gray_frame, lm, fake_ctx, opacity=1.0, thickness=0.6))
    _, _, up, _ = eye_frame(lm, side)
    s = regions.side_indices(side)
    lash = regions.resample_polyline(lm[s.eye_upper], 11)
    on_line = [sample(diff, p + 0.6 * up) for p in lash[5:10]]
    assert min(on_line) > 0.15
    above = [sample(diff, p + 0.12 * regions.face_scale(lm) * up) for p in lash[2:9]]
    assert max(above) < INVISIBLE
    assert diff[under_eye(lm, side, gray_frame.shape)].max() < INVISIBLE  # no lower line unless asked


@pytest.mark.parametrize("side", regions.SIDES)
def test_liner_tapers_from_inner_to_outer_corner(gray_frame, face_landmarks_720p, fake_ctx, side) -> None:
    lm = face_landmarks_720p
    out = apply(EyelinerEffect, gray_frame, lm, fake_ctx, opacity=1.0, thickness=1.0, wing=0.0, softness=0.0)
    diff = side_half(change(gray_frame, out), lm, side)
    inner, along, _, width = eye_frame(lm, side)
    xs, ys = pixel_coords(diff.shape)
    u = ((xs - inner[0]) * along[0] + (ys - inner[1]) * along[1]) / width
    inked = diff > 0.1
    assert inked[(u > 0.6) & (u < 0.85)].sum() > 1.5 * inked[(u > 0.05) & (u < 0.3)].sum()


@pytest.mark.parametrize("side", regions.SIDES)
def test_wing_extends_past_the_outer_corner_and_rises(gray_frame, face_landmarks_720p, fake_ctx, side) -> None:
    lm = face_landmarks_720p
    inner, along, up, width = eye_frame(lm, side)
    outer = inner + along * width
    scale = regions.face_scale(lm)

    def tip(wing: float) -> np.ndarray:
        out = apply(EyelinerEffect, gray_frame, lm, fake_ctx, opacity=1.0, wing=wing)
        diff = side_half(change(gray_frame, out), lm, side)
        ys, xs = np.nonzero(diff > 0.05)
        points = np.stack([xs + 0.5, ys + 0.5], axis=1) - outer
        return points[np.argmax(points @ along)]  # the inked point farthest past the corner

    short, long = tip(0.0), tip(1.0)
    assert long @ along > short @ along + 0.12 * scale
    angle = math.degrees(math.atan2(long @ up, long @ along))
    assert 10.0 < angle < 35.0  # rises above the eye line


@pytest.mark.parametrize("side", regions.SIDES)
def test_lower_line_lines_the_outer_part_of_the_lower_lid(gray_frame, face_landmarks_720p, fake_ctx, side) -> None:
    lm = face_landmarks_720p
    _, _, up, _ = eye_frame(lm, side)
    lower = regions.resample_polyline(lm[regions.side_indices(side).eye_lower], 12)

    def along_lower_lid(lower_amount: float) -> list[float]:
        out = apply(EyelinerEffect, gray_frame, lm, fake_ctx, opacity=1.0, thickness=0.8, lower=lower_amount)
        diff = change(gray_frame, out)
        return [sample(diff, p - 0.5 * up) for p in lower]

    none, full = along_lower_lid(0.0), along_lower_lid(1.0)
    assert max(none[1:9]) < INVISIBLE
    assert min(full[7:11]) > 0.05  # outer third
    assert max(full[1:3]) < INVISIBLE  # inner third stays clean


def test_liner_follows_a_closed_lid(gray_frame, face_landmarks_720p, fake_ctx) -> None:
    lm = closed_eyes(face_landmarks_720p)
    diff = change(gray_frame, apply(EyelinerEffect, gray_frame, lm, fake_ctx, opacity=1.0, thickness=0.6))
    for side in regions.SIDES:
        _, _, up, _ = eye_frame(lm, side)
        lid = regions.resample_polyline(lm[regions.side_indices(side).eye_lower], 11)
        assert min(sample(diff, p + 0.6 * up) for p in lid[5:10]) > 0.15


@pytest.mark.parametrize("effect_cls", EFFECTS)
@pytest.mark.parametrize("roll_deg", [-20.0, 20.0])
def test_makeup_turns_with_a_rolled_head(gray_frame, face_landmarks_720p, fake_ctx, effect_cls, roll_deg) -> None:
    pivot = face_landmarks_720p[regions.NOSE_TIP]
    angle = math.radians(roll_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], np.float32)
    rolled = (face_landmarks_720p - pivot) @ rotation.T + pivot
    out = apply(effect_cls, gray_frame, rolled, fake_ctx, opacity=1.0)
    for side in regions.SIDES:
        _, _, up, _ = eye_frame(rolled, side)
        s = regions.side_indices(side)
        offset = centroid(side_half(change(gray_frame, out), rolled, side)) - rolled[s.eye].mean(axis=0)
        across = offset - np.dot(offset, up) * up
        assert np.dot(offset, up) > 0  # above the eye in the face's own frame
        assert np.linalg.norm(across) < 0.25 * regions.face_scale(rolled)


# --------------------------------------------------------------------------
# Robustness.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("effect_cls", EFFECTS)
def test_segmentation_masks_are_neither_needed_nor_used(frame, face_landmarks_720p, fake_ctx, effect_cls) -> None:
    without = apply(effect_cls, frame, face_landmarks_720p, fake_ctx)
    half = np.full(frame.shape[:2], 0.5, np.float32)
    masks = {name: half for name in ("hair_mask", "face_skin_mask", "body_skin_mask")}
    ctx = fake_ctx(frame, face_landmarks_720p, **masks)
    with_masks = effect_cls().apply(frame.copy(), ctx, SETTINGS[effect_cls]())
    # Equal up to float rounding: OpenCV's SIMD kernels may round the last bit
    # differently depending on buffer alignment (far below one 8-bit step).
    np.testing.assert_allclose(with_masks, without, rtol=0, atol=1e-6)


@pytest.mark.parametrize("effect_cls", EFFECTS)
def test_read_only_landmarks_are_left_untouched(frame, face_landmarks_720p, fake_ctx, effect_cls) -> None:
    landmarks = face_landmarks_720p.copy()
    landmarks.setflags(write=False)  # the pipeline shares its landmarks read-only
    apply(effect_cls, frame, landmarks, fake_ctx, **STRONG[effect_cls])
    assert np.array_equal(landmarks, face_landmarks_720p)


@pytest.mark.parametrize("effect_cls", EFFECTS)
@pytest.mark.parametrize("dx", [-600.0, -560.0, 760.0, 2000.0], ids=["one-eye-out", "edge", "other-edge", "gone"])
def test_eyes_at_and_beyond_the_frame_border(frame, face_landmarks_720p, fake_ctx, effect_cls, dx) -> None:
    moved = face_landmarks_720p + np.array([dx, 0.0], np.float32)
    out = apply(effect_cls, frame, moved, fake_ctx, **STRONG[effect_cls])
    assert out.shape == frame.shape and np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    if dx == 2000.0:
        assert np.array_equal(out, frame)


@pytest.mark.parametrize("effect_cls", EFFECTS)
def test_tiny_and_degenerate_faces_are_skipped(frame, face_landmarks_720p, fake_ctx, effect_cls) -> None:
    center = face_landmarks_720p.mean(axis=0)
    tiny = (face_landmarks_720p - center) * np.float32(0.03) + center
    collapsed = np.repeat(center[None], len(face_landmarks_720p), axis=0)
    for landmarks in (tiny, collapsed):
        out = apply(effect_cls, frame, landmarks, fake_ctx, **STRONG[effect_cls])
        assert np.array_equal(out, frame)


def test_real_pipeline_applies_eye_makeup(face_frame_720p, face_landmarks_720p, models_dir) -> None:
    from makeup_cam.pipeline import MakeupPipeline

    settings = MakeupSettings()
    pipeline = MakeupPipeline(settings, model_dir=models_dir, effects=[EyeshadowEffect(), EyelinerEffect()])
    try:
        for ts in range(0, 300, 33):  # past the fade-in
            result = pipeline.process(face_frame_720p, ts)
    finally:
        pipeline.close()
    assert result.face_found
    assert result.timings_ms.keys() >= {"eyeshadow", "eyeliner"}
    diff = np.abs(result.output.astype(np.int16) - face_frame_720p.astype(np.int16)).max(axis=2)
    eyes = regions.roi_around(face_landmarks_720p[regions.LEFT_EYE + regions.RIGHT_EYE], face_frame_720p.shape, 0.6)
    assert diff[eyes.slices].max() >= 8
    outside = np.ones(diff.shape, bool)
    outside[eyes.slices] = False
    assert diff[outside].max() == 0

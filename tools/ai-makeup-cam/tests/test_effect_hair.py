"""HairEffect on the astronaut portrait with real landmarks and a synthetic hair
mask: no-ops, locality, strength scaling, each control's direction, missing
or misleading masks, odd framings, and a real segmentation run."""

from __future__ import annotations

from dataclasses import fields

import cv2
import numpy as np
import pytest

from makeup_cam import blend, regions
from makeup_cam.config import HairSettings, MakeupSettings
from makeup_cam.effects import hair as hair_module
from makeup_cam.effects.base import FrameContext, registered_effects
from makeup_cam.effects.hair import HairEffect

SLIDERS = ("gloss", "smoothness", "color_strength", "brightness", "contrast")
OFF = {name: 0.0 for name in SLIDERS}
# Everything far from neutral, in the dye's strongest direction (darker, warmer).
STRONG = {
    "gloss": 1.0,
    "smoothness": 1.0,
    "color": "#3B2A22",
    "color_strength": 1.0,
    "brightness": 0.3,
    "contrast": 0.4,
}


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


@pytest.fixture
def hair_mask(face_landmarks_720p: np.ndarray) -> np.ndarray:
    return synthetic_hair_mask(face_landmarks_720p, (720, 1280))


def synthetic_hair_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A soft hair "cap" around the top of the face oval (the face cut out),
    close to what the segmenter finds on the portrait."""
    scale = regions.face_scale(landmarks)
    roi = regions.Roi(0, 0, shape[1], shape[0])
    oval = landmarks[regions.FACE_OVAL]
    lo, hi = oval.min(axis=0), oval.max(axis=0)
    center = np.array([(lo[0] + hi[0]) / 2, lo[1] + 0.1 * scale + (hi[1] - lo[1]) * 0.25])
    axes = ((hi[0] - lo[0]) / 2 + 0.6 * scale, center[1] - (lo[1] - 1.2 * scale))
    cap = regions.ellipse_mask(center, axes, 0.0, roi, feather_px=0.05 * scale)
    face = regions.polygon_mask(oval, roi, feather_px=0.05 * scale)
    return (cap * (1.0 - face)).astype(np.float32)


def apply_hair(
    frame: np.ndarray,
    landmarks: np.ndarray | None,
    hair_mask: np.ndarray | None,
    intensity: float = 1.0,
    face_skin_mask: np.ndarray | None = None,
    **values,
) -> np.ndarray:
    ctx = FrameContext(
        height=frame.shape[0],
        width=frame.shape[1],
        landmarks=landmarks,
        hair_mask=hair_mask,
        face_skin_mask=face_skin_mask,
        intensity=intensity,
    )
    return HairEffect().apply(frame.copy(), ctx, HairSettings(**values))


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def hair_core(hair_mask: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Pixels well inside the (synthetic) hair."""
    erode = max(3, round(0.15 * regions.face_scale(landmarks)))
    solid = (hair_mask > 0.99).astype(np.uint8)
    return cv2.erode(solid, np.ones((erode, erode), np.uint8)).astype(bool)


def far_from(mask: np.ndarray, distance_px: float) -> np.ndarray:
    """Pixels farther than ``distance_px`` from every pixel where ``mask`` > 0."""
    outside = (mask <= 0).astype(np.uint8)
    return cv2.distanceTransform(outside, cv2.DIST_L2, 5) > distance_px


def test_registered_under_its_settings_section() -> None:
    assert HairEffect in registered_effects()
    assert (HairEffect.name, HairEffect.order) == ("hair", 10)
    assert (HairEffect.needs_face, HairEffect.needs_hair_mask) == (False, True)
    assert isinstance(getattr(MakeupSettings(), HairEffect.name), HairSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "mask_kind"),
    [
        (1.0, OFF, "real"),
        (1.0, {**OFF, "edge_softness": 1.0, "color": "#FF0000"}, "real"),
        (0.0, STRONG, "real"),
        (1.0, STRONG, "none"),
        (1.0, STRONG, "empty"),
    ],
    ids=["sliders-0", "only-non-strength-settings", "intensity-0", "no-hair-mask", "empty-hair-mask"],
)
def test_exact_noop(frame, face_landmarks_720p, hair_mask, intensity, values, mask_kind) -> None:
    mask = {"real": hair_mask, "none": None, "empty": np.zeros_like(hair_mask)}[mask_kind]
    out = apply_hair(frame, face_landmarks_720p, mask, intensity=intensity, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


SINGLE_CONTROLS = [
    {"gloss": 1.0},
    {"smoothness": 1.0},
    {"color_strength": 1.0, "color": "#8E2F2A"},
    {"brightness": 0.5},
    {"brightness": -0.5},
    {"contrast": 0.5},
    {"contrast": -0.5},
]


@pytest.mark.parametrize("values", [*SINGLE_CONTROLS, STRONG], ids=lambda v: "-".join(f"{k}={v[k]}" for k in v))
@pytest.mark.parametrize("edge_softness", [0.0, 1.0])
def test_every_control_gives_a_valid_frame(frame, face_landmarks_720p, hair_mask, values, edge_softness) -> None:
    out = apply_hair(frame, face_landmarks_720p, hair_mask, **{**OFF, **values, "edge_softness": edge_softness})
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(frame, out)[hair_core(hair_mask, face_landmarks_720p)].mean() > 0.002


def test_changes_stay_on_the_hair(frame, face_landmarks_720p, hair_mask) -> None:
    out = apply_hair(frame, face_landmarks_720p, hair_mask, **STRONG, edge_softness=1.0)
    diff = change(frame, out)
    scale = regions.face_scale(face_landmarks_720p)
    assert diff[far_from(hair_mask, 0.5 * scale)].max() == 0.0
    assert np.array_equal(out[:60, :60], frame[:60, :60])  # background corners
    assert np.array_equal(out[-60:, -60:], frame[-60:, -60:])
    # The eyes, brows, nose and mouth are protected even where the mask strays.
    inner = regions.polygon_mask(face_landmarks_720p[regions.FACE_OVAL], regions.Roi(0, 0, 1280, 720))
    features = regions.expand_polygon(face_landmarks_720p[regions.FACE_OVAL], -0.6 * scale)
    features = regions.polygon_mask(features, regions.Roi(0, 0, 1280, 720)) > 0.99
    assert inner.sum() > features.sum() > 1000
    assert diff[features].max() == 0.0


def test_inner_face_is_protected_from_a_stray_mask(frame, face_landmarks_720p, hair_mask) -> None:
    """A mask lagging behind the head (or confusing dark brows for hair) covers
    the face; the landmarks keep the features untouched."""
    roi = regions.Roi(0, 0, 1280, 720)
    stray = np.maximum(hair_mask, regions.polygon_mask(face_landmarks_720p[regions.FACE_OVAL], roi))
    out = apply_hair(frame, face_landmarks_720p, stray, **STRONG)
    for side in regions.SIDES:
        eye = regions.eye_opening_mask(face_landmarks_720p, side, roi) > 0.5
        brow = regions.brow_mask(face_landmarks_720p, side, roi) > 0.5
        assert change(frame, out)[eye | brow].max() == 0.0
    mouth = regions.lips_mask(face_landmarks_720p, roi) > 0.5
    assert change(frame, out)[mouth].max() == 0.0


def test_face_skin_confidence_is_never_treated_as_hair(frame, face_landmarks_720p, hair_mask) -> None:
    roi = regions.Roi(0, 0, 1280, 720)
    face = regions.polygon_mask(face_landmarks_720p[regions.FACE_OVAL], roi, feather_px=2.0)
    bleeding = np.maximum(hair_mask, 0.6 * face)  # hair confidence bleeding over the forehead
    out = apply_hair(frame, face_landmarks_720p, bleeding, face_skin_mask=face, **STRONG)
    assert change(frame, out)[face >= 1.0].max() == 0.0
    assert change(frame, out)[face > 0.9].max() < 0.02


def test_background_false_positives_far_from_the_face_are_ignored(frame, face_landmarks_720p, hair_mask) -> None:
    blob = hair_mask.copy()
    blob[500:640, 1050:1200] = 0.9  # a dark object in the background, seen as hair
    out = apply_hair(frame, face_landmarks_720p, blob, **STRONG)
    assert change(frame, out)[480:660, 1030:1220].max() == 0.0
    assert change(frame, out)[hair_core(hair_mask, face_landmarks_720p)].mean() > 0.02


def test_works_without_a_face(frame, face_landmarks_720p, hair_mask) -> None:
    """The hair stays made up while the face is not detected (e.g. turned away)."""
    out = apply_hair(frame, None, hair_mask, **STRONG)
    diff = change(frame, out)
    scale = regions.face_scale(face_landmarks_720p)
    assert diff[hair_core(hair_mask, face_landmarks_720p)].mean() > 0.02
    assert diff[far_from(hair_mask, 0.5 * scale)].max() == 0.0


@pytest.mark.parametrize("values", SINGLE_CONTROLS, ids=lambda v: "-".join(f"{k}={v[k]}" for k in v))
def test_strength_grows_with_intensity(frame, face_landmarks_720p, hair_mask, values) -> None:
    core = hair_core(hair_mask, face_landmarks_720p)

    def strength(intensity: float) -> float:
        out = apply_hair(frame, face_landmarks_720p, hair_mask, intensity=intensity, **{**OFF, **values})
        return float(change(frame, out)[core].mean())

    levels = [strength(i) for i in (0.25, 0.5, 1.0)]
    assert levels[0] > 0.0
    assert levels[0] < levels[1] < levels[2]


def hair_luminance(image: np.ndarray, core: np.ndarray) -> np.ndarray:
    return blend.luminance(image)[core]


def test_brightness_and_contrast_go_the_right_way(frame, face_landmarks_720p, hair_mask) -> None:
    core = hair_core(hair_mask, face_landmarks_720p)
    original = hair_luminance(frame, core)

    def toned(**values) -> np.ndarray:
        return hair_luminance(apply_hair(frame, face_landmarks_720p, hair_mask, **{**OFF, **values}), core)

    assert toned(brightness=0.5).mean() > original.mean() + 0.03
    assert toned(brightness=-0.5).mean() < original.mean() - 0.03
    assert toned(contrast=0.5).std() > original.std() * 1.1
    assert toned(contrast=-0.5).std() < original.std() * 0.9
    assert abs(toned(contrast=0.5).mean() - original.mean()) < 0.03  # pivot at the hair's mean


def test_brightening_keeps_the_hue(frame, face_landmarks_720p, hair_mask) -> None:
    """Lifted shadows keep the hair's hue instead of turning orange."""
    core = hair_core(hair_mask, face_landmarks_720p)
    before = blend.bgr_to_lab(frame)[core].mean(axis=0)
    brightened = apply_hair(frame, face_landmarks_720p, hair_mask, **{**OFF, "brightness": 0.5})
    after = blend.bgr_to_lab(brightened)[core].mean(axis=0)
    hue_before = np.degrees(np.arctan2(before[2], before[1]))
    hue_after = np.degrees(np.arctan2(after[2], after[1]))
    assert after[0] > before[0] + 3.0
    assert abs(hue_after - hue_before) < 6.0


def test_dye_tones_light_hair_down_to_the_chosen_color(frame, face_landmarks_720p, hair_mask) -> None:
    core = hair_core(hair_mask, face_landmarks_720p)
    target = blend.bgr_to_lab(np.float32([[[0x22, 0x2A, 0x3B]]]) / 255.0)[0, 0]  # "#3B2A22"
    before = blend.bgr_to_lab(frame)[core].mean(axis=0)
    dyed = apply_hair(frame, face_landmarks_720p, hair_mask, **{**OFF, "color": "#3B2A22", "color_strength": 1.0})
    after = blend.bgr_to_lab(dyed)[core].mean(axis=0)
    assert after[0] < before[0] - 15.0  # much darker
    assert abs(after[0] - target[0]) < abs(before[0] - target[0]) / 2
    assert np.linalg.norm(after[1:] - target[1:]) < np.linalg.norm(before[1:] - target[1:])


def test_dye_only_tints_dark_hair(frame, face_landmarks_720p, hair_mask) -> None:
    """Without bleach a vivid red cannot cover black hair: it shows as a tint."""
    core = hair_core(hair_mask, face_landmarks_720p)
    m = hair_mask[..., None]
    dark = frame * (1.0 - m) + frame * np.float32([0.25, 0.22, 0.2]) * m
    red = {**OFF, "color": "#C0302A", "color_strength": 1.0}

    def chroma_gain(image: np.ndarray) -> float:
        before = blend.bgr_to_lab(image)[core][:, 1:].mean(axis=0)
        after = blend.bgr_to_lab(apply_hair(image, face_landmarks_720p, hair_mask, **red))[core][:, 1:].mean(axis=0)
        return float(np.linalg.norm(after - before))

    assert 0.5 < chroma_gain(dark) < 0.5 * chroma_gain(frame)


def test_gloss_lifts_the_highlights_more_than_the_rest(frame, face_landmarks_720p, hair_mask) -> None:
    core = hair_core(hair_mask, face_landmarks_720p)
    before = hair_luminance(frame, core)
    after = hair_luminance(apply_hair(frame, face_landmarks_720p, hair_mask, **{**OFF, "gloss": 1.0}), core)
    gain = after - before
    bright = before >= np.percentile(before, 80)
    dark = before <= np.percentile(before, 30)
    assert gain.min() >= -1e-6  # gloss only adds light
    assert gain[bright].mean() > 3.0 * gain[dark].mean()
    assert np.percentile(after, 98) > np.percentile(before, 98) + 0.01


def test_smoothness_reduces_texture_inside_the_hair(frame, face_landmarks_720p, hair_mask) -> None:
    core = hair_core(hair_mask, face_landmarks_720p)

    def texture(image: np.ndarray) -> float:
        luma = blend.luminance(image)
        return float(np.std((luma - cv2.GaussianBlur(luma, (0, 0), 2.0))[core]))

    smoothed = apply_hair(frame, face_landmarks_720p, hair_mask, **{**OFF, "smoothness": 1.0})
    assert texture(smoothed) < 0.85 * texture(frame)
    # The silhouette against the background is not blurred.
    edge = (hair_mask > 0.02) & (hair_mask < 0.3)
    assert change(frame, smoothed)[edge].max() < 0.02


def test_repeated_frames_give_the_same_result(frame, face_landmarks_720p, hair_mask) -> None:
    """Temporal smoothing of the statistics converges on a still image, and reset() clears it."""
    effect = HairEffect()
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p, hair_mask=hair_mask)
    settings = HairSettings(**STRONG)
    first = effect.apply(frame.copy(), ctx, settings)
    second = effect.apply(frame.copy(), ctx, settings)
    assert np.allclose(first, second, atol=1e-5)
    effect.apply(np.clip(frame * 1.5, 0.0, 1.0), ctx, settings)  # different statistics
    effect.reset()
    assert np.allclose(effect.apply(frame.copy(), ctx, settings), first, atol=1e-5)
    # A new mask (the next segmentation run) replaces what was found in the old one.
    no_hair = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p, hair_mask=np.zeros_like(hair_mask))
    assert np.array_equal(effect.apply(frame.copy(), no_hair, settings), frame)


def test_close_up_face_is_processed_on_reduced_copies(frame, face_landmarks_720p, hair_mask) -> None:
    """A face close to the camera (large hair area) still gives a clean, local result."""
    zoom = 2.5
    center = face_landmarks_720p[regions.FACE_OVAL].mean(axis=0)
    shift = np.float32([640.0, 420.0]) - zoom * center
    matrix = np.float32([[zoom, 0, shift[0]], [0, zoom, shift[1]]])
    big = cv2.warpAffine(frame, matrix, (1280, 720), flags=cv2.INTER_LINEAR)
    big_mask = cv2.warpAffine(hair_mask, matrix, (1280, 720), flags=cv2.INTER_LINEAR)
    landmarks = (face_landmarks_720p * zoom + shift).astype(np.float32)
    out = apply_hair(big, landmarks, big_mask, **STRONG)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(big, out)[hair_core(big_mask, landmarks)].mean() > 0.02
    assert change(big, out)[far_from(big_mask, 0.5 * regions.face_scale(landmarks))].max() == 0.0


def test_copy_resolution_does_not_flip_at_a_threshold() -> None:
    """A head hovering where the working resolution would change keeps it,
    so the look does not flicker between two resolutions."""
    assert hair_module._grid_factor(2.6, None) == 3
    assert hair_module._grid_factor(2.6, 2) == 2
    assert hair_module._grid_factor(2.7, 2) == 3
    assert hair_module._grid_factor(0.2, None) == 1


def test_hair_cut_off_by_the_frame_border(frame, face_landmarks_720p, hair_mask) -> None:
    shift = np.float32([0.0, -110.0])  # the top of the head leaves the frame
    moved = np.roll(frame, -110, axis=0)
    moved_mask = np.roll(hair_mask, -110, axis=0)
    moved_mask[-110:] = 0.0
    out = apply_hair(moved, face_landmarks_720p + shift, moved_mask, **STRONG)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(moved, out)[:20].max() > 0.0  # the hair at the top edge is made up


def test_settings_cover_every_slider() -> None:
    names = {f.name for f in fields(HairSettings)}
    assert set(SLIDERS) | {"enabled", "color", "edge_softness"} == names


def test_real_segmentation_mask(face_frame_720p, face_landmarks_720p, models_dir) -> None:
    from makeup_cam.segmentation import Segmenter

    model = models_dir / "selfie_multiclass_256x256.tflite"
    if not model.is_file():
        pytest.skip("segmentation model not downloaded")
    segmenter = Segmenter(model, interval=1, background=False)
    try:
        masks = segmenter.update(face_frame_720p, 33, 1)
    finally:
        segmenter.close()
    frame = face_frame_720p.astype(np.float32) / np.float32(255.0)
    out = apply_hair(frame, face_landmarks_720p, masks.hair, face_skin_mask=masks.face_skin, **STRONG)
    diff = change(frame, out)
    real_hair = masks.hair > 0.9
    assert real_hair.sum() > 5000
    assert diff[real_hair].mean() > 0.02
    assert diff[far_from(masks.hair > 0.2, 0.5 * regions.face_scale(face_landmarks_720p))].max() == 0.0
    assert np.array_equal(out[:60, :60], frame[:60, :60])

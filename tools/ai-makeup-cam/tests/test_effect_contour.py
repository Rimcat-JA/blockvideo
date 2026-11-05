"""ContourEffect and TearbagEffect on the astronaut portrait (real landmarks)
and on flat skin-colored frames (where the change shows the masks directly):
no-ops, locality, protected features, geometry (face size, roll, yaw, blinks),
where each slider paints and how soft and how strong it is at the size of a
webcam face, segmentation masks and the hairline, color on light / medium /
dark skin, real motion (tracker lag in a nod, pixels changing in a blink),
temporal behavior, cost and a run through the pipeline."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from makeup_cam import regions
from makeup_cam.config import ContourSettings, MakeupSettings, TearbagSettings, hex_to_bgr
from makeup_cam.effects import contour as contour_module
from makeup_cam.effects.base import registered_effects
from makeup_cam.effects.contour import ContourEffect, TearbagEffect
from makeup_cam.effects.skin import SkinEffect
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.segmentation import SegMasks
from makeup_cam.tracking import OneEuroFilter, face_scale_px, smoothing_params

# Every contour area at full strength.
AREAS = {
    "jawline": 1.0,
    "cheek_hollow": 1.0,
    "nose_shadow": 1.0,
    "forehead": 1.0,
    "nose_bridge": 1.0,
    "cheekbone": 1.0,
    "chin": 1.0,
    "cupids_bow": 1.0,
}
STRONG = {"shading": 1.0, "highlight": 1.0, "softness": 1.0, **AREAS}
SHADE_ONLY = {"shading": 1.0, "highlight": 0.0, **AREAS}
LIGHT_ONLY = {"shading": 0.0, "highlight": 1.0, **AREAS}
BAG_STRONG = {"strength": 1.0, "width": 1.0, "shadow": 1.0, "softness": 1.0}
# Skin tones, BGR.
LIGHT_SKIN = (0.74, 0.86, 0.98)
MEDIUM_SKIN = (0.45, 0.6, 0.8)
DARK_SKIN = (0.17, 0.28, 0.42)
SKINS = {"light": LIGHT_SKIN, "medium": MEDIUM_SKIN, "dark": DARK_SKIN}
INVISIBLE = 1.0 / 255.0  # below one 8-bit output step


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


@pytest.fixture
def large_face(face_landmarks_720p: np.ndarray) -> np.ndarray:
    """The same landmarks at the size of a face in a webcam interview (face scale about 128 px; the
    fixture's own 58 px is a small face, where the coarse grids hardly coarsen anything)."""
    nose = face_landmarks_720p[regions.NOSE_TIP]
    return ((face_landmarks_720p - nose) * 2.2 + np.float32([640, 370])).astype(np.float32)


def flat(color: tuple[float, float, float] = MEDIUM_SKIN, shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
    image = np.empty((*shape, 3), np.float32)
    image[...] = color
    return image


def apply_contour(frame, landmarks, fake_ctx, intensity=1.0, face_skin_mask=None, hair_mask=None, **values):
    ctx = fake_ctx(frame, landmarks, intensity=intensity, face_skin_mask=face_skin_mask, hair_mask=hair_mask)
    return ContourEffect().apply(frame.copy(), ctx, ContourSettings(**values))


def apply_bag(frame, landmarks, fake_ctx, intensity=1.0, effect=None, **values):
    ctx = fake_ctx(frame, landmarks, intensity=intensity)
    return (effect or TearbagEffect()).apply(frame.copy(), ctx, TearbagSettings(**values))


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def centroid(weights: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[: weights.shape[0], : weights.shape[1]]
    total = weights.sum()
    return np.array([(xs * weights).sum() / total, (ys * weights).sum() / total])


def affine(landmarks: np.ndarray, matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    return ((landmarks - center) @ matrix.T + center).astype(np.float32)


def rotation(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def full_skin_mask(frame: np.ndarray) -> np.ndarray:
    return np.ones(frame.shape[:2], np.float32)


def lab_lightness(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[..., 0]


def only_area(area: str, **values) -> dict:
    """Settings with one contour area on, both tones at full strength."""
    return {**dict.fromkeys(AREAS, 0.0), "shading": 1.0, "highlight": 1.0, area: 1.0, **values}


def around(image: np.ndarray, point: np.ndarray, radius: int = 2) -> float:
    """Mean of ``image`` in a small box around ``point``."""
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    return float(image[y - radius : y + radius + 1, x - radius : x + radius + 1].mean())


def section(image: np.ndarray, start: np.ndarray, end: np.ndarray, samples: int = 200) -> tuple[np.ndarray, float]:
    """``image`` sampled along a line, and the length of the line."""
    t = np.linspace(0.0, 1.0, samples)[:, None]
    points = np.asarray(start, np.float64) * (1.0 - t) + np.asarray(end, np.float64) * t
    return image[points[:, 1].astype(int), points[:, 0].astype(int)], float(np.linalg.norm(np.subtract(end, start)))


def steepest_slope(diff: np.ndarray, scale: float) -> float:
    """The steepest slope of a (slightly blurred) change map, in shares of its peak per face scale."""
    gy, gx = np.gradient(cv2.GaussianBlur(diff, (0, 0), 1.0))
    return float(np.hypot(gx, gy).max() / diff.max() * scale)


# --------------------------------------------------------------------------
# Contract: registration, no-ops, valid output, locality.
# --------------------------------------------------------------------------


def test_registered_under_their_settings_sections() -> None:
    assert ContourEffect in registered_effects() and TearbagEffect in registered_effects()
    assert (ContourEffect.name, ContourEffect.order, ContourEffect.needs_face) == ("contour", 25, True)
    assert (TearbagEffect.name, TearbagEffect.order, TearbagEffect.needs_face) == ("tearbag", 45, True)
    settings = MakeupSettings()
    assert isinstance(settings.contour, ContourSettings)
    assert isinstance(settings.tearbag, TearbagSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "use_landmarks"),
    [
        (1.0, {"shading": 0.0, "highlight": 0.0, **AREAS}, True),
        (1.0, {"shading": 1.0, "highlight": 1.0, **dict.fromkeys(AREAS, 0.0)}, True),
        (0.0, STRONG, True),
        (1.0, STRONG, False),
    ],
    ids=["strengths-0", "areas-0", "intensity-0", "no-face"],
)
def test_contour_exact_noop(frame, face_landmarks_720p, fake_ctx, intensity, values, use_landmarks) -> None:
    landmarks = face_landmarks_720p if use_landmarks else None
    mask = full_skin_mask(frame)
    out = apply_contour(frame, landmarks, fake_ctx, intensity=intensity, face_skin_mask=mask, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


@pytest.mark.parametrize(
    ("intensity", "values", "use_landmarks"),
    [
        (1.0, {**BAG_STRONG, "strength": 0.0}, True),
        (1.0, {**BAG_STRONG, "width": 0.0}, True),
        (0.0, BAG_STRONG, True),
        (1.0, BAG_STRONG, False),
    ],
    ids=["strength-0", "width-0", "intensity-0", "no-face"],
)
def test_tearbag_exact_noop(frame, face_landmarks_720p, fake_ctx, intensity, values, use_landmarks) -> None:
    landmarks = face_landmarks_720p if use_landmarks else None
    out = apply_bag(frame, landmarks, fake_ctx, intensity=intensity, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


@pytest.mark.parametrize(
    "off",
    [{"shading": 0.0}, {"highlight": 0.0}, {"forehead": 0.0}, {"softness": 0.0}, {"nose_shadow": 0.0, "jawline": 0.0}],
    ids=lambda o: "+".join(o),
)
def test_contour_output_is_valid_for_slider_combinations(frame, face_landmarks_720p, fake_ctx, off) -> None:
    out = apply_contour(frame, face_landmarks_720p, fake_ctx, face_skin_mask=full_skin_mask(frame), **{**STRONG, **off})
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(frame, out).max() > 0.02


def test_every_contour_slider_does_something_on_its_own(fake_ctx, face_landmarks_720p) -> None:
    image = flat()
    mask = np.ones(image.shape[:2], np.float32)
    mask[: int(face_landmarks_720p[10][1]) + 8] = 0.0  # a hairline for the forehead area
    off = dict.fromkeys(AREAS, 0.0)
    for area in AREAS:
        values = {**off, area: 1.0}
        out = apply_contour(image, face_landmarks_720p, fake_ctx, face_skin_mask=mask, shading=1.0, highlight=1.0, **values)
        assert change(image, out).max() > 0.02, area


def test_changes_stay_on_the_face(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_contour(frame, face_landmarks_720p, fake_ctx, **STRONG)
    bag = apply_bag(out, face_landmarks_720p, fake_ctx, **BAG_STRONG)
    diff = change(frame, bag)
    scale = regions.face_scale(face_landmarks_720p)
    near = regions.roi_around(face_landmarks_720p[regions.FACE_OVAL], frame.shape, pad=0.0, margin_px=0.1 * scale)
    outside = np.ones(diff.shape, bool)
    outside[near.slices] = False
    assert diff[outside].max() == 0.0
    assert np.array_equal(bag[:50, :50], frame[:50, :50])  # background corner
    assert diff.max() > 0.03


def test_without_mask_nothing_passes_the_face_outline(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    out = apply_contour(image, face_landmarks_720p, fake_ctx, **STRONG)
    margin = 0.05 * regions.face_scale(face_landmarks_720p)
    oval = regions.face_oval_mask(face_landmarks_720p, full_roi(image), expand_px=margin)
    assert change(image, out)[oval < 0.01].max() < INVISIBLE


# --------------------------------------------------------------------------
# Protected features.
# --------------------------------------------------------------------------


def feature_masks(frame: np.ndarray, landmarks: np.ndarray) -> dict[str, np.ndarray]:
    roi = full_roi(frame)
    nostrils = np.zeros(frame.shape[:2], bool)
    for ids in (contour_module._RIGHT_NOSTRIL, contour_module._LEFT_NOSTRIL):
        hull = cv2.convexHull(landmarks[ids].astype(np.float32)).reshape(-1, 2)
        nostrils |= regions.polygon_mask(hull, roi) > 0.5
    return {
        "eyes": (regions.eye_opening_mask(landmarks, "left", roi) + regions.eye_opening_mask(landmarks, "right", roi)) > 0.5,
        "brows": (regions.brow_mask(landmarks, "left", roi) + regions.brow_mask(landmarks, "right", roi)) > 0.5,
        "lips": regions.polygon_mask(landmarks[regions.LIPS_OUTER], roi) > 0.5,
        "nostrils": nostrils,
    }


@pytest.mark.parametrize("use_flat", [False, True], ids=["portrait", "flat"])
def test_contour_never_paints_eyes_brows_lips_or_nostrils(frame, face_landmarks_720p, fake_ctx, use_flat) -> None:
    image = flat() if use_flat else frame
    out = apply_contour(image, face_landmarks_720p, fake_ctx, face_skin_mask=full_skin_mask(image), **STRONG)
    diff = change(image, out)
    for name, mask in feature_masks(image, face_landmarks_720p).items():
        assert mask.sum() > 20, name
        assert diff[mask].max() == 0.0, name


def test_contour_keeps_features_clean_on_a_large_face(frame, face_landmarks_720p, fake_ctx) -> None:
    landmarks = affine(face_landmarks_720p, np.eye(2) * 2.2, face_landmarks_720p[regions.NOSE_TIP])
    landmarks += np.float32([640, 360]) - landmarks[regions.NOSE_TIP]
    image = flat()
    out = apply_contour(image, landmarks, fake_ctx, face_skin_mask=full_skin_mask(image), **STRONG)
    diff = change(image, out)
    for name, mask in feature_masks(image, landmarks).items():
        assert diff[mask].max() == 0.0, name


def test_tearbag_keeps_the_eye_clean_on_a_large_face(frame, large_face, fake_ctx) -> None:
    """The masks are built on a coarse grid there; the upsampling must not reach the eye."""
    image = flat()
    for factor in (1.0, 2.0):
        landmarks = affine(large_face, np.eye(2) * factor, np.float32([640, 330]))
        diff = change(image, apply_bag(image, landmarks, fake_ctx, **BAG_STRONG))
        assert diff.max() > 0.02
        for side in regions.SIDES:
            eye = regions.eye_opening_mask(landmarks, side, full_roi(image)) > 0.5
            assert eye.sum() > 50
            assert diff[eye].max() < INVISIBLE


def test_tearbag_never_paints_the_eyeball(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_bag(frame, face_landmarks_720p, fake_ctx, **BAG_STRONG)
    roi = full_roi(frame)
    for side in regions.SIDES:
        eye = regions.eye_opening_mask(face_landmarks_720p, side, roi) > 0.5
        assert eye.sum() > 50
        assert change(frame, out)[eye].max() == 0.0
    assert change(frame, out).max() > 0.02


# --------------------------------------------------------------------------
# Geometry.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("apply", "values"), [(apply_contour, STRONG), (apply_bag, BAG_STRONG)], ids=["contour", "tearbag"])
def test_scales_with_face_size(face_landmarks_720p, fake_ctx, apply, values) -> None:
    image = flat()
    center = np.float32([640, 360])
    small = affine(face_landmarks_720p, np.eye(2), face_landmarks_720p[regions.NOSE_TIP])
    small += center - small[regions.NOSE_TIP]
    big = affine(small, np.eye(2) * 2.0, center)
    painted = {}
    for name, landmarks in (("small", small), ("big", big)):
        diff = change(image, apply(image, landmarks, fake_ctx, **values))
        assert diff.max() > 0.02
        painted[name] = (float((diff > 0.25 * diff.max()).sum()), float(diff.max()))
    assert 3.0 < painted["big"][0] / painted["small"][0] < 5.2
    assert painted["big"][1] == pytest.approx(painted["small"][1], rel=0.35)


def test_contour_follows_head_roll(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    center = face_landmarks_720p[regions.NOSE_TIP]
    base = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, shading=1.0, highlight=1.0))
    angle = 10.0
    rolled = affine(face_landmarks_720p, rotation(angle), center)
    out = change(image, apply_contour(image, rolled, fake_ctx, shading=1.0, highlight=1.0))
    # cv2 rotates counter-clockwise for positive angles in image coordinates.
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -angle, 1.0)
    expected = cv2.warpAffine(base, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    box = regions.roi_around(rolled[regions.FACE_OVAL], image.shape, pad=0.05).slices
    correlation = np.corrcoef(out[box].ravel(), expected[box].ravel())[0, 1]
    assert correlation > 0.97


def test_tearbag_follows_head_roll(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    center = face_landmarks_720p[regions.NOSE_TIP]
    rolled = affine(face_landmarks_720p, rotation(10.0), center)
    for landmarks in (face_landmarks_720p, rolled):
        diff = change(image, apply_bag(image, landmarks, fake_ctx, strength=1.0, shadow=0.0))
        split = int(round(float(landmarks[regions.NOSE_TIP][0])))
        left, right = diff.copy(), diff.copy()
        left[:, split:] = 0
        right[:, :split] = 0
        scale = regions.face_scale(landmarks)
        down = contour_module._face_down(landmarks)
        for weights, side in ((left, "right"), (right, "left")):
            lid = landmarks[regions.side_indices(side).eye_lower].astype(np.float32).mean(axis=0)
            offset = (centroid(weights) - lid) @ down / scale
            assert 0.0 < offset < 0.1  # just below the lower lash line, whatever the roll


def yawed(landmarks: np.ndarray, squeeze: float) -> np.ndarray:
    """A head turned so the image-right half of the face narrows toward the nose."""
    out = landmarks.copy()
    mid = float(landmarks[regions.NOSE_TIP][0])
    far = out[:, 0] > mid
    out[far, 0] = mid + (out[far, 0] - mid) * squeeze
    return out


def test_contour_follows_head_yaw(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    turned = yawed(face_landmarks_720p, 0.6)
    diff = change(image, apply_contour(image, turned, fake_ctx, shading=1.0, highlight=1.0))
    margin = 0.05 * regions.face_scale(turned)
    oval = regions.face_oval_mask(turned, full_roi(image), expand_px=margin)
    assert diff[oval < 0.01].max() < INVISIBLE  # stays inside the narrowed outline
    mid = int(round(float(turned[regions.NOSE_TIP][0])))
    near, far = (diff[:, :mid] > 0.02).sum(), (diff[:, mid:] > 0.02).sum()
    assert 0.3 * near < far < 0.85 * near  # the far side's strokes narrow with it


def test_tearbag_follows_head_yaw(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    turned = yawed(face_landmarks_720p, 0.6)
    diff = change(image, apply_bag(image, turned, fake_ctx, **BAG_STRONG))
    roi = full_roi(image)
    assert diff[regions.eye_opening_mask(turned, "left", roi) > 0.5].max() == 0.0
    mid = int(round(float(turned[regions.NOSE_TIP][0])))
    left_eye_x = turned[regions.LEFT_EYE, 0]
    far = diff[:, mid:].copy()
    xs = np.nonzero(far.max(axis=0) > 0.02)[0] + mid
    assert left_eye_x.min() - 6 < xs.min() and xs.max() < left_eye_x.max() + 6


def test_mirror_table_pairs_are_mirror_images(face_landmarks_720p) -> None:
    """Every pair lies at about the same height and on both sides of the nose."""
    landmarks = face_landmarks_720p
    scale = regions.face_scale(landmarks)
    across = regions.eye_center(landmarks, "left") - regions.eye_center(landmarks, "right")
    across /= np.linalg.norm(across)
    down = contour_module._face_down(landmarks)
    mid = landmarks[regions.NOSE_BRIDGE[0]]
    for right, left in contour_module._MIRROR_PAIRS.items():
        a, b = (landmarks[right] - mid), (landmarks[left] - mid)
        assert a @ across < 0.01 * scale < b @ across or abs(a @ across) < 0.05 * scale, (right, left)
        assert abs((a - b) @ down) < 0.12 * scale, (right, left)
        assert abs((a + b) @ across) < 0.06 * scale, (right, left)  # as far from the nose on both sides


def test_strokes_reference_only_known_mirrored_landmarks() -> None:
    midline = {168, 8, 9, 151, 10, 6, 197, 195, 5, 4, 1, 199, 0, 164, 2}
    for stroke in contour_module._STROKES:
        for a, b, *_ in stroke.vertices:
            if not stroke.mirrored:
                continue
            for index in (a, b):
                assert index in contour_module._MIRROR or index in midline, (stroke.area, index)


# --------------------------------------------------------------------------
# Where each slider paints, how soft and how strong (face at webcam size).
# --------------------------------------------------------------------------

# Slider -> landmarks whose mean must lie in its stroke, and places it must leave alone.
PLACES: dict[str, tuple[list[list[int]], list[list[int]]]] = {
    "jawline": ([[58, 138], [172, 135], [288, 367], [397, 364]], [[6], [151], [50], [280], [152], [199], [61], [291]]),
    "cheek_hollow": ([[147, 187], [376, 411]], [[6], [151], [152], [172, 135], [397, 364], [61], [291]]),
    "nose_shadow": ([[122, 245], [351, 465], [196, 188], [419, 412]], [[1], [61], [58], [151], [152], [50], [280]]),
    "nose_bridge": ([[6], [197], [195], [9], [151]], [[50], [58], [61], [152], [2]]),
    "cheekbone": ([[117, 111], [346, 340], [143], [372]], [[58], [172], [1], [152], [61], [6], [147, 187]]),
    "chin": ([[199, 208], [199, 428], [199]], [[0], [1], [58], [397], [61], [291]]),
    "cupids_bow": ([[37, 167], [267, 393], [164]], [[199], [152], [1], [58], [50], [6]]),
}


@pytest.mark.parametrize("area", PLACES)
def test_each_slider_paints_its_own_place_and_nothing_else(large_face, fake_ctx, area) -> None:
    image = flat()
    out = apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area(area))
    diff = change(image, out)
    strong, absent = PLACES[area]
    for anchors in strong:
        assert around(diff, large_face[anchors].mean(axis=0)) > 0.3 * diff.max(), (area, anchors)
    for anchors in absent:
        assert around(diff, large_face[anchors].mean(axis=0)) < 0.2 * diff.max(), (area, anchors)
    assert ((out - image).min() < 0.0) == (area in ("jawline", "cheek_hollow", "nose_shadow"))  # shade or light


def test_jaw_shadow_is_darkest_at_the_edge_of_the_face_and_fades_inward(large_face, fake_ctx) -> None:
    image = flat()
    scale = regions.face_scale(large_face)
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area("jawline")))
    for edge in (58, 172, 288, 397):
        profile, length = section(diff, large_face[edge], large_face[regions.NOSE_TIP])
        at = lambda distance: float(profile[int(distance * scale / length * (len(profile) - 1))])  # noqa: E731
        assert int(np.argmax(profile)) * length / len(profile) < 0.16 * scale, edge
        assert at(0.45) < 0.2 * float(profile.max()), edge


def test_cheek_shadow_ends_well_before_the_mouth_corner(large_face, fake_ctx) -> None:
    """A shadow beside the mouth reads as dirt."""
    image = flat()
    scale = regions.face_scale(large_face)
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area("cheek_hollow")))
    ys, xs = np.mgrid[: diff.shape[0], : diff.shape[1]]
    for corner in (regions.MOUTH_RIGHT_CORNER, regions.MOUTH_LEFT_CORNER):
        distance = np.hypot(xs + 0.5 - large_face[corner][0], ys + 0.5 - large_face[corner][1]) / scale
        assert diff[distance < 0.1].max() < 0.15 * diff.max()
        assert diff[distance < 0.2].max() < 0.3 * diff.max()


def test_nose_shadow_lies_on_the_side_walls_not_on_the_bridge(large_face, fake_ctx) -> None:
    image = flat()
    scale = regions.face_scale(large_face)
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area("nose_shadow")))
    for index in (6, 197):
        x, y = np.round(large_face[index]).astype(int)
        row = diff[y - 1 : y + 2].mean(axis=0)
        span = int(0.3 * scale)
        left, right = row[x - span : x], row[x + 1 : x + span]
        assert row[x] < 0.4 * min(left.max(), right.max()), index
        assert x - span + int(np.argmax(left)) < x - 0.06 * scale, index  # the peaks are away from the center
        assert x + 1 + int(np.argmax(right)) > x + 0.06 * scale, index


@pytest.mark.parametrize(
    ("area", "limit"),
    [("jawline", 11.0), ("cheek_hollow", 12.0), ("nose_shadow", 19.5), ("nose_bridge", 20.5), ("cheekbone", 20.0), ("chin", 8.5), ("cupids_bow", 34.0)],
)
def test_no_stroke_has_a_hard_edge(large_face, fake_ctx, area, limit) -> None:
    """The slope of a stroke is bounded: a stripe or a crease shows up as a steep edge."""
    image = flat()
    scale = regions.face_scale(large_face)
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area(area)))
    assert steepest_slope(diff, scale) < limit


@pytest.mark.parametrize(("area", "spread"), [("jawline", 3.05), ("cheek_hollow", 3.8)])
def test_broad_shadows_have_a_long_tail(large_face, fake_ctx, area, spread) -> None:
    """Blended powder: the shadow is still there far from its stroke, so it never reads as a band."""
    image = flat()
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area(area)))
    assert (diff > 0.05 * diff.max()).sum() > spread * (diff > 0.5 * diff.max()).sum()


@pytest.mark.parametrize(
    ("skin", "shade_range", "light_range"),
    [(LIGHT_SKIN, (-8.5, -4.5), (0.3, 1.8)), (MEDIUM_SKIN, (-7.5, -4.0), (3.0, 6.0)), (DARK_SKIN, (-4.5, -2.0), (3.0, 6.0))],
    ids=SKINS.keys(),
)
def test_default_strength_is_pinned(large_face, fake_ctx, skin, shade_range, light_range) -> None:
    """At the defaults the shape shows but stays in the range an interviewer takes for a real face."""
    image = flat(skin)
    out = apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image))
    delta = lab_lightness(out) - lab_lightness(image)
    assert shade_range[0] < float(delta.min()) < shade_range[1]
    assert light_range[0] < float(delta.max()) < light_range[1]


@pytest.mark.parametrize(
    ("skin", "darkest", "lightest"),
    [(LIGHT_SKIN, -23.0, 1.8), (MEDIUM_SKIN, -20.0, 8.0), (DARK_SKIN, -12.0, 10.0)],
    ids=SKINS.keys(),
)
def test_full_strength_stays_makeup(large_face, fake_ctx, skin, darkest, lightest) -> None:
    """Every slider at 1: heavy, but with limits (layered powder saturates)."""
    image = flat(skin)
    out = apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **STRONG)
    delta = lab_lightness(out) - lab_lightness(image)
    assert darkest < float(delta.min()) < 0.6 * darkest
    assert float(delta.max()) < lightest


def test_layered_powder_saturates() -> None:
    weights = np.float32([0.0, 0.05, 0.5, 1.0])
    saturated = contour_module._saturate(weights)
    assert saturated[0] == 0.0
    assert saturated[1] > 0.95 * weights[1]  # light layers keep their value
    assert saturated[3] < 0.75  # a full layer does not read as painted on


def test_shading_and_highlight_colors_reach_the_result(large_face, fake_ctx) -> None:
    image = flat()
    mask = full_skin_mask(image)

    def peak_color(**values) -> np.ndarray:
        out = apply_contour(image, large_face, fake_ctx, face_skin_mask=mask, **values)
        y, x = np.unravel_index(np.argmax(change(image, out)), out.shape[:2])
        return (out - image)[y, x]

    shade = {**only_area("jawline"), "highlight": 0.0}
    warm, deep = peak_color(**shade, shading_color="#C08060"), peak_color(**shade, shading_color="#604030")
    assert deep.sum() < warm.sum() - 0.05  # a darker color gives a deeper shadow
    light = {**only_area("nose_bridge"), "shading": 0.0}
    pink, pale = peak_color(**light, highlight_color="#FFB0B0"), peak_color(**light, highlight_color="#FFFFFF")
    assert pink[0] < pale[0] - 0.005  # a pink highlight lifts blue less
    assert pink[2] == pytest.approx(pale[2], abs=0.015)  # and red about the same


def test_shadow_is_deeper_on_more_saturated_skin() -> None:
    """Skin more saturated than the reference deepens along its own chroma; its green and blue
    drop more than the same relative shadow on a paler skin."""
    pale, deep = (0.68, 0.76, 0.9), DARK_SKIN
    pale_loss = 1.0 - tone(pale, shade=1.0) / np.float32(pale)
    deep_loss = 1.0 - tone(deep, shade=1.0) / np.float32(deep)
    assert deep_loss[0] > pale_loss[0] + 0.1
    assert deep_loss[1] > pale_loss[1] + 0.02


def test_shading_color_limits() -> None:
    """A black shading color cannot take the skin below the floor, a neon one cannot tint it far."""
    skin = MEDIUM_SKIN
    black = tone(skin, shade=1.0, shading_color="#000000")
    assert float(np.dot(contour_module._LUMA_BGR, black)) > 0.2 * float(np.dot(contour_module._LUMA_BGR, np.float32(skin)))
    neon = tone(skin, shade=1.0, shading_color="#FF2000") / np.float32(skin)
    plain = tone(skin, shade=1.0, shading_color="#A0745F") / np.float32(skin)
    assert (neon / neon[1]).max() < 1.8 * (plain / plain[1]).max()


def test_highlight_uses_only_part_of_the_headroom() -> None:
    """Near the top of the range the lift is limited by what is left, softly, far from clipping."""
    skin = (0.7, 0.8, 0.9)
    lit = tone(skin, light=1.0)
    assert lit[2] - skin[2] < 0.6 * (1.0 - skin[2])
    assert lit[2] > skin[2] + 0.02


def test_pale_sheen_on_skin_without_headroom() -> None:
    """Red is clipped, so nothing can lift it: a pearl sheen closes the gap to the other channels a little."""
    skin = (0.62, 0.84, 1.0)
    lit = tone(skin, light=1.0)
    assert lit[0] > skin[0] + 0.005
    assert lit[0] < skin[0] + 0.05


def test_far_side_strokes_narrow_on_a_turned_head(large_face, fake_ctx) -> None:
    image = flat()
    assert contour_module._cheek_squeeze(large_face, "left", regions.face_scale(large_face)) > 0.85
    turned = yawed(large_face, 0.6)
    scale = regions.face_scale(turned)
    assert contour_module._cheek_squeeze(turned, "left", scale) < 0.8
    assert contour_module._cheek_squeeze(turned, "right", scale) > 0.85
    diff = change(image, apply_contour(image, turned, fake_ctx, face_skin_mask=full_skin_mask(image), **only_area("cheek_hollow")))
    mid = int(round(float(turned[regions.NOSE_TIP][0])))
    near, far = (diff[:, :mid] > 0.3 * diff.max()).sum(), (diff[:, mid:] > 0.3 * diff.max()).sum()
    assert far < 0.6 * near


# --------------------------------------------------------------------------
# Segmentation masks and the hairline.
# --------------------------------------------------------------------------


def test_contour_is_restricted_to_skin_when_the_mask_is_present(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    mask = np.ones(image.shape[:2], np.float32)
    split = int(round(float(face_landmarks_720p[regions.NOSE_TIP][0])))
    mask[:, split + 25 :] = 0.0  # e.g. a curtain of hair over the image-right cheek
    free = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, **STRONG))
    masked = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, face_skin_mask=mask, **STRONG))
    assert free[:, split + 40 :].max() > 0.05
    assert masked[:, split + 40 :].max() < INVISIBLE
    assert masked[:, : split - 20].max() > 0.05


@pytest.mark.parametrize("with_hair", [False, True], ids=["no-hair-mask", "hair-mask"])
def test_a_skin_mask_that_misses_the_face_turns_the_contour_off(frame, face_landmarks_720p, fake_ctx, with_hair) -> None:
    """The mask and the tracker disagree: no guessing, whatever the hair mask says."""
    empty = np.zeros(frame.shape[:2], np.float32)
    out = apply_contour(frame, face_landmarks_720p, fake_ctx, face_skin_mask=empty, hair_mask=empty if with_hair else None, **STRONG)
    assert np.array_equal(out, frame)


@pytest.mark.parametrize("with_hair", [False, True], ids=["no-hair-mask", "hair-mask"])
def test_falling_skin_coverage_fades_the_contour_without_a_jump(large_face, fake_ctx, with_hair) -> None:
    """As the mask loses the face the effect only ever gets weaker, step by step."""
    image = flat()
    skin = full_skin_mask(image)
    hair = np.zeros(image.shape[:2], np.float32) if with_hair else None
    outs = [
        apply_contour(image, large_face, fake_ctx, face_skin_mask=skin * np.float32(c), hair_mask=hair)
        for c in np.arange(0.8, 0.199, -0.02)
    ]
    painted = [float(change(image, out).sum()) for out in outs]
    assert painted[0] > 0.0 and painted[-1] == 0.0
    assert all(b <= a * 1.001 for a, b in zip(painted, painted[1:], strict=False))  # never bolder on the way down
    steps = [float(change(a, b).max()) for a, b in zip(outs, outs[1:], strict=False)]
    assert max(steps) < 4.0 * INVISIBLE


def skin_down_to(frame: np.ndarray, landmarks: np.ndarray, coverage: float) -> np.ndarray:
    """A mask of skin from the top of the mesh down to the row where it covers ``coverage`` of the face outline."""
    scale = regions.face_scale(landmarks)
    size = (frame.shape[1], frame.shape[0])
    inset = contour_module._filled(size, [regions.expand_polygon(landmarks[regions.FACE_OVAL], -contour_module._OVAL_INSET * scale)])
    covered = np.cumsum(inset.sum(axis=1)) / inset.sum()
    top = int(landmarks[10][1])
    bottom = int(np.searchsorted(covered, coverage + float(covered[top])))
    mask = np.zeros(frame.shape[:2], np.float32)
    mask[top:bottom] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), 2.0)


def test_a_small_patch_of_skin_mask_turns_the_contour_off(large_face, fake_ctx) -> None:
    """Confident skin on a quarter of the face is not a face: nothing is painted, not even on that quarter."""
    image = flat()
    assert np.array_equal(apply_contour(image, large_face, fake_ctx, face_skin_mask=skin_down_to(image, large_face, 0.25), **STRONG), image)


@pytest.mark.parametrize(
    ("values", "landmark"),
    [
        ({"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}, 10),
        ({"shading": 0.0, "highlight": 1.0, **dict.fromkeys(AREAS, 0.0), "nose_bridge": 1.0}, 151),
    ],
    ids=["hairline-shadow", "strokes"],
)
def test_partial_skin_coverage_weakens_everything_where_the_mask_is_sure(large_face, fake_ctx, values, landmark) -> None:
    """Where the skin is the same, less of it on the rest of the face still means less effect."""
    image = flat()
    whole = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=skin_down_to(image, large_face, 0.99), **values))
    partial = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=skin_down_to(image, large_face, 0.42), **values))
    box = regions.roi_around(large_face[[landmark]], image.shape, pad=0.0, margin_px=0.25 * regions.face_scale(large_face)).slices
    assert 0.0 < float(partial[box].max()) < 0.5 * float(whole[box].max())


def test_unsure_skin_is_not_a_hairline(large_face, fake_ctx) -> None:
    """A forehead the model is merely less sure about (a highlight, dim light) gets no hairline shadow."""
    image = flat()
    unsure = np.full(image.shape[:2], 0.65, np.float32)
    only_forehead = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}
    assert np.array_equal(apply_contour(image, large_face, fake_ctx, face_skin_mask=unsure, **only_forehead), image)


def test_hair_mask_is_the_fallback_for_a_missing_skin_mask(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    hair = np.zeros(image.shape[:2], np.float32)
    split = int(round(float(face_landmarks_720p[regions.NOSE_TIP][0])))
    hair[:, split + 25 :] = 1.0
    diff = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, hair_mask=hair, **STRONG))
    assert diff[:, split + 40 :].max() < INVISIBLE
    assert diff[:, : split - 20].max() > 0.05


def test_hair_mask_alone_keeps_the_contour_inside_the_outline(large_face, fake_ctx) -> None:
    """It says where hair is, not where the face ends: nothing may pass the jaw or the temples."""
    image = flat()
    no_hair = np.zeros(image.shape[:2], np.float32)
    diff = change(image, apply_contour(image, large_face, fake_ctx, hair_mask=no_hair, **STRONG))
    margin = 0.05 * regions.face_scale(large_face)
    oval = regions.face_oval_mask(large_face, full_roi(image), expand_px=margin)
    assert diff.max() > 0.05
    assert diff[oval < 0.01].max() < INVISIBLE
    # Soft hair confidence counts as hair: a 0.85 is not skin.
    hair = no_hair.copy()
    split = int(round(float(large_face[regions.NOSE_TIP][0])))
    hair[:, split:] = 0.85
    diff = change(image, apply_contour(image, large_face, fake_ctx, hair_mask=hair, **STRONG))
    assert diff[:, split + 25 :].max() < INVISIBLE
    assert diff[:, : split - 25].max() > 0.05


def test_with_a_mask_the_contour_stays_near_the_outline(large_face, fake_ctx) -> None:
    """A mask finds the real edge of the face near the outline, not far beyond it."""
    image = flat()
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=full_skin_mask(image), **STRONG))
    margin = 0.2 * regions.face_scale(large_face)
    oval = regions.face_oval_mask(large_face, full_roi(image), expand_px=margin)
    assert diff[oval < 0.01].max() < INVISIBLE


def test_mask_edge_is_soft(large_face, fake_ctx) -> None:
    """Hair across the cheek: the effect fades out toward it instead of ending in a line."""
    image = flat()
    split = int(round(float(large_face[regions.NOSE_TIP][0]))) + 40
    mask = np.ones(image.shape[:2], np.float32)
    mask[:, split:] = 0.0
    scale = regions.face_scale(large_face)
    values = {**only_area("cheekbone"), "shading": 0.0, "cheek_hollow": 1.0, "chin": 1.0}
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=mask, **values))
    y = int(large_face[117][1])
    row = diff[y - 2 : y + 3].mean(axis=0)[split - int(0.3 * scale) : split + 5]
    assert np.abs(np.diff(row)).max() * scale / max(float(row.max()), 1e-6) < 15.0


def test_hairline_needs_the_skin_mask_not_the_hair_mask(large_face, fake_ctx) -> None:
    """The hair mask alone cannot tell forehead from background: no shadow beside the head."""
    image = flat()
    hair = np.zeros(image.shape[:2], np.float32)
    hair[: int(large_face[10][1])] = 1.0
    only_forehead = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}
    assert np.array_equal(apply_contour(image, large_face, fake_ctx, hair_mask=hair, **only_forehead), image)


def hairline_mask(frame: np.ndarray, y: int) -> np.ndarray:
    """Skin below row ``y``, hair above it (soft edge, as from the segmentation model)."""
    mask = np.zeros(frame.shape[:2], np.float32)
    mask[y:] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), 2.0)


def test_hairline_shadow_needs_a_mask(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    only_forehead = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}
    assert np.array_equal(apply_contour(image, face_landmarks_720p, fake_ctx, **only_forehead), image)
    y = int(face_landmarks_720p[10][1] - 0.2 * regions.face_scale(face_landmarks_720p))
    out = apply_contour(image, face_landmarks_720p, fake_ctx, face_skin_mask=hairline_mask(image, y), **only_forehead)
    assert change(image, out).max() > 0.05


@pytest.mark.parametrize("above_mesh", [True, False], ids=["high-forehead", "bangs"])
def test_hairline_shadow_hugs_the_real_hairline(face_landmarks_720p, fake_ctx, above_mesh) -> None:
    """The shadow follows the mask, whether the hair starts above the end of the
    mesh (a high forehead) or below it (bangs), and never lies on the hair."""
    image = flat()
    top = float(face_landmarks_720p[10][1])
    scale = regions.face_scale(face_landmarks_720p)
    y = int(top - 0.35 * scale) if above_mesh else int(top + 0.25 * scale)
    only_forehead = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}
    out = apply_contour(image, face_landmarks_720p, fake_ctx, face_skin_mask=hairline_mask(image, y), **only_forehead)
    diff = change(image, out)
    x = int(face_landmarks_720p[9][0])
    column = diff[:, x - 3 : x + 4].mean(axis=1)
    reach = int(0.5 * scale)
    assert column[: y - 12].max() < INVISIBLE  # on the hair
    assert column[y + 4 : y + 16].max() > 3.0 * column[y + reach]  # fades into the skin
    assert column[y + 4 : y + 16].max() > 0.03


def test_hairline_shadow_stays_on_the_forehead(large_face, fake_ctx) -> None:
    """Skin that ends elsewhere (jaw, neck, a hand on the cheek) gets no hairline shadow."""
    image = flat()
    scale = regions.face_scale(large_face)
    top, chin = float(large_face[10][1]), float(large_face[152][1])
    mask = np.zeros(image.shape[:2], np.float32)
    # Skin everywhere inside a box: its bottom and sides are "not skin" too.
    mask[int(top - 0.3 * scale) : int(chin), int(large_face[234][0]) - 10 : int(large_face[454][0]) + 10] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    only_forehead = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=mask, **only_forehead))
    assert diff[: int(top + 0.2 * scale)].max() > 0.05
    below_the_eyes = int(large_face[regions.NOSE_TIP][1])
    assert diff[below_the_eyes:].max() < INVISIBLE


def test_hairline_shadow_fades_out_along_the_hairline(large_face, fake_ctx) -> None:
    """Toward the temples the shadow fades over a long way: it does not end in an edge."""
    image = flat()
    scale = regions.face_scale(large_face)
    y = int(large_face[10][1] - 0.15 * scale)
    only_forehead = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0, "softness": 0.0}
    diff = change(image, apply_contour(image, large_face, fake_ctx, face_skin_mask=hairline_mask(image, y), **only_forehead))
    row = diff[y + 6 : y + 16].mean(axis=0)
    assert np.abs(np.diff(row)).max() * scale / float(row.max()) < 10.0


def test_hairline_shadow_has_no_top_edge_at_the_end_of_the_mesh(face_landmarks_720p, fake_ctx) -> None:
    """The face mesh ends on the forehead; a high forehead must not show a band ending there."""
    image = flat()
    top = float(face_landmarks_720p[10][1])
    scale = regions.face_scale(face_landmarks_720p)
    y = int(top - 0.45 * scale)
    values = {"shading": 1.0, "highlight": 0.0, **dict.fromkeys(AREAS, 0.0), "forehead": 1.0}
    out = apply_contour(image, face_landmarks_720p, fake_ctx, face_skin_mask=hairline_mask(image, y), **values)
    x = int(face_landmarks_720p[9][0])
    column = change(image, out)[:, x - 3 : x + 4].mean(axis=1)
    steps = np.abs(np.diff(column[y + 3 : int(top) + 30]))
    assert steps.max() < 2.5 * INVISIBLE


# --------------------------------------------------------------------------
# Color.
# --------------------------------------------------------------------------


def tone(color, shade=0.0, light=0.0, shading_color="#A0745F", highlight_color="#FFF4EA") -> np.ndarray:
    """The color after ``_tone_gain`` with these weights (a 1x1 image)."""
    local = np.float32(color).reshape(1, 1, 3)
    gain = contour_module._tone_gain(
        local,
        np.float32([[shade]]),
        np.float32([[light]]),
        contour_module._shading_filter(hex_to_bgr(shading_color)),
        contour_module._Light.of(hex_to_bgr(highlight_color), 0.5, 0.6, contour_module._CONTOUR_PEARL),
    )
    return (local * (1.0 + gain))[0, 0]


def test_tone_gain_is_exactly_zero_at_zero_weight() -> None:
    local = np.random.default_rng(0).random((8, 8, 3), np.float32) * 0.9 + 0.05
    zeros = np.zeros((8, 8), np.float32)
    filter_ = contour_module._shading_filter(hex_to_bgr("#A0745F"))
    light = contour_module._Light.of(hex_to_bgr("#FFF4EA"), 0.5, 0.6, 0.06)
    assert not contour_module._tone_gain(local, zeros, zeros, filter_, light).any()
    assert not contour_module._tone_gain(local, zeros, None, filter_, light).any()
    assert not contour_module._tone_gain(local, None, zeros, filter_, light).any()


def test_full_shading_turns_the_reference_skin_into_the_shading_color() -> None:
    color = hex_to_bgr("#A0745F")
    assert tone(contour_module._REFERENCE_SKIN, shade=1.0, shading_color="#A0745F") == pytest.approx(color, abs=0.02)


@pytest.mark.parametrize("skin", SKINS.values(), ids=SKINS.keys())
def test_shadow_is_the_skins_own_darker_tone(skin) -> None:
    """Darker, at least as saturated, and warm: never a grey or ashy veil."""
    shaded = tone(skin, shade=0.8)
    b0, g0, r0 = skin
    b1, g1, r1 = shaded
    assert b1 < b0 and g1 < g0 and r1 < r0
    assert (r1 - b1) / r1 >= (r0 - b0) / r0  # saturation does not drop
    assert r1 / g1 >= r0 / g0 and g1 / b1 >= g0 / b0 - 1e-3  # warmer, never cooler


def test_shadow_on_blown_out_skin_stays_warm() -> None:
    """A clipped red channel lost its real value: lowering it like the others reads as grey."""
    blown = (0.62, 0.84, 1.0)
    b, g, r = tone(blown, shade=0.8)
    assert r > 0.85
    assert r / g > 1.1 * blown[2] / blown[1]  # redder than the skin, not olive or grey
    assert g < blown[1] - 0.1 and b < blown[0] - 0.1  # it still reads as a shadow


def test_cool_shading_color_never_cools_the_skin() -> None:
    for skin in SKINS.values():
        b, g, r = tone(skin, shade=1.0, shading_color="#7080A0")
        assert r / b >= skin[2] / skin[0] - 1e-3


@pytest.mark.parametrize("skin", [MEDIUM_SKIN, DARK_SKIN], ids=["medium", "dark"])
def test_highlight_lifts_skin_with_headroom_keeping_its_hue(skin) -> None:
    lit = tone(skin, light=0.8)
    ratio = lit / np.float32(skin)
    assert ratio.min() > 1.05
    assert ratio.max() - ratio.min() < 0.06  # all channels by about the same factor: no chalky veil
    assert float(lit.max()) < 1.0


@pytest.mark.parametrize("skin", [LIGHT_SKIN, (0.8, 0.92, 1.0), (0.9, 0.95, 0.99)], ids=["light", "blown-red", "pale"])
def test_highlight_cannot_flatten_light_skin_to_white(skin) -> None:
    for weight in (0.5, 1.0):
        lit = tone(skin, light=weight)
        assert float(lit.max()) <= 1.0
        gap_before = max(skin) - min(skin)
        gap_after = float(lit.max() - lit.min())
        assert gap_after > 0.85 * gap_before  # the channels stay apart: skin, not white
        assert float(np.abs(lit - np.float32(skin)).max()) < 0.04


def test_highlight_on_a_light_face_is_gentle_and_never_clips(face_landmarks_720p, fake_ctx) -> None:
    image = flat(LIGHT_SKIN)
    out = apply_contour(image, face_landmarks_720p, fake_ctx, **LIGHT_ONLY)
    assert out.max() <= 1.0
    assert (out[..., 2] > 0.999).mean() == 0.0
    channel_gap = out[..., 2] - out[..., 0]
    assert channel_gap.min() > 0.8 * (LIGHT_SKIN[2] - LIGHT_SKIN[0])


def test_highlight_keeps_texture(frame, face_landmarks_720p, fake_ctx) -> None:
    """A gain on the skin multiplies its detail instead of painting over it."""
    noise = np.random.default_rng(1).normal(0.0, 0.02, frame.shape[:2]).astype(np.float32)
    base = flat()
    textured = np.clip(base * (1.0 + noise[..., None]), 0.0, 1.0)
    clean = apply_contour(base, face_landmarks_720p, fake_ctx, **LIGHT_ONLY)
    with_noise = apply_contour(textured, face_landmarks_720p, fake_ctx, **LIGHT_ONLY)
    nose = regions.roi_around(face_landmarks_720p[regions.NOSE_BRIDGE[1:4]], frame.shape, pad=0.0, margin_px=3.0).slices
    assert (with_noise[nose] / clean[nose]).std() >= 0.9 * (textured[nose] / base[nose]).std()


@pytest.mark.parametrize("tone_name", ["highlight", "shading"])
def test_nose_stroke_is_broad_without_a_hard_center_line(face_landmarks_720p, fake_ctx, tone_name) -> None:
    image = flat()
    scale = regions.face_scale(face_landmarks_720p)
    if tone_name == "highlight":
        values = {**dict.fromkeys(AREAS, 0.0), "shading": 0.0, "highlight": 1.0, "nose_bridge": 1.0}
    else:
        values = {**dict.fromkeys(AREAS, 0.0), "shading": 1.0, "highlight": 0.0, "nose_shadow": 1.0}
    diff = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, **values))
    x, y = np.round(face_landmarks_720p[197]).astype(int)
    row = diff[y - 2 : y + 3].mean(axis=0)
    peak = float(row[x - int(0.3 * scale) : x + int(0.3 * scale)].max())
    assert peak > 0.03
    # Gentle everywhere: a soft stroke changes by a small share of its peak per pixel.
    steepest = float(np.abs(np.diff(row)).max())
    relative = steepest / peak * scale  # share of the peak per face-scale unit
    assert relative < 25.0
    if tone_name == "highlight":
        wide = np.nonzero(row[x - int(0.4 * scale) : x + int(0.4 * scale)] > 0.5 * peak)[0]
        assert wide.size > 0.12 * scale


def test_nose_shadow_leaves_the_top_of_the_bridge_alone(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    values = {**dict.fromkeys(AREAS, 0.0), "shading": 1.0, "highlight": 0.0, "nose_shadow": 1.0}
    diff = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, **values))
    x, y = np.round(face_landmarks_720p[197]).astype(int)
    scale = regions.face_scale(face_landmarks_720p)
    span = int(0.2 * scale)
    assert diff[y, x] < 0.35 * max(diff[y, x - span : x].max(), diff[y, x + 1 : x + span].max())


def test_contour_sculpts_rather_than_paints(frame, face_landmarks_720p, fake_ctx) -> None:
    """Default strengths visibly shape the face; everything at full strength stays makeup."""
    landmarks = face_landmarks_720p
    jaw = np.round((landmarks[172] + landmarks[135]) / 2).astype(int)
    default = apply_contour(frame, landmarks, fake_ctx)
    shade = 1.0 - (default[jaw[1], jaw[0]] / frame[jaw[1], jaw[0]])
    assert 0.03 < float(shade.mean()) < 0.2
    heavy = apply_contour(frame, landmarks, fake_ctx, **STRONG)
    ratio = (heavy + 1e-3) / (frame + 1e-3)
    skin = regions.face_oval_mask(landmarks, full_roi(frame)) > 0.5
    assert np.percentile(ratio[skin], 0.5) > 0.5
    assert np.percentile(ratio[skin], 99.5) < 1.5


def test_softness_makes_the_contour_smoother(face_landmarks_720p, fake_ctx) -> None:
    image = flat()

    def steepest(softness: float) -> float:
        diff = change(image, apply_contour(image, face_landmarks_720p, fake_ctx, shading=1.0, highlight=1.0, softness=softness))
        gy, gx = np.gradient(cv2.GaussianBlur(diff, (0, 0), 1.0))
        return float(np.percentile(np.hypot(gx, gy), 99.5))

    assert steepest(1.0) < steepest(0.0)


@pytest.mark.parametrize(("apply", "field"), [(apply_contour, "shading"), (apply_contour, "highlight"), (apply_bag, "strength")])
def test_strength_grows_with_intensity_and_slider(frame, face_landmarks_720p, fake_ctx, apply, field) -> None:
    def amount(intensity: float, value: float) -> float:
        out = apply(frame, face_landmarks_720p, fake_ctx, intensity=intensity, **{field: value})
        return float(change(frame, out).sum())

    by_intensity = [amount(i, 0.8) for i in (0.25, 0.5, 1.0)]
    by_slider = [amount(1.0, v) for v in (0.2, 0.5, 1.0)]
    assert by_intensity == sorted(by_intensity) and by_intensity[0] > 0.0
    assert by_slider == sorted(by_slider) and by_slider[0] > 0.0


# --------------------------------------------------------------------------
# Tear bag.
# --------------------------------------------------------------------------


def test_tearbag_is_a_lit_band_under_the_lashes_with_a_shadow_below(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    landmarks = face_landmarks_720p
    scale = regions.face_scale(landmarks)
    out = apply_bag(image, landmarks, fake_ctx, strength=1.0, width=1.0, shadow=1.0)
    signed = (out - image)[..., 2]
    for side in regions.SIDES:
        s = regions.side_indices(side)
        x, y = landmarks[s.eye_lower[4]]
        column = signed[:, int(round(x))]
        y = int(round(y))
        lit = column[y : y + int(0.07 * scale) + 3]
        dark = column[y + int(0.07 * scale) : y + int(0.16 * scale) + 3]
        assert lit.max() > 0.02
        assert dark.min() < -0.005
        assert int(np.argmax(column[y : y + int(0.2 * scale)])) < int(np.argmin(column[y : y + int(0.2 * scale)]))


def test_tearbag_width_slider_reaches_toward_the_outer_corner(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    landmarks = face_landmarks_720p
    s = regions.side_indices("left")
    inner, outer = landmarks[s.eye_lower[0]], landmarks[s.eye_lower[-1]]
    along = (outer - inner) / np.linalg.norm(outer - inner)
    ys, xs = np.mgrid[: image.shape[0], : image.shape[1]]

    def reach(width: float) -> float:
        diff = change(image, apply_bag(image, landmarks, fake_ctx, strength=1.0, shadow=0.0, width=width))
        right = xs > float(landmarks[regions.NOSE_TIP][0])
        sel = right & (diff > 0.25 * diff.max())
        return float((((xs[sel] - inner[0]) * along[0] + (ys[sel] - inner[1]) * along[1]).max()) / np.linalg.norm(outer - inner))

    assert reach(0.2) < reach(0.5) < reach(1.0) <= 1.15
    assert reach(0.5) > 0.5  # the default covers the bag's middle, not only the inner corner


def lid_coordinates(landmarks: np.ndarray, side: regions.Side, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: position along this lid (0 inner corner, 1 outer corner) and depth below it (face scales)."""
    s = regions.side_indices(side)
    inner, outer = landmarks[s.eye_lower[0]], landmarks[s.eye_lower[-1]]
    length = float(np.linalg.norm(outer - inner))
    along = (outer - inner) / length
    down = contour_module._face_down(landmarks)
    ys, xs = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    u = ((xs + 0.5 - inner[0]) * along[0] + (ys + 0.5 - inner[1]) * along[1]) / length
    lid_y = landmarks[s.eye_lower].astype(np.float32).mean(axis=0)
    depth = ((xs + 0.5 - lid_y[0]) * down[0] + (ys + 0.5 - lid_y[1]) * down[1]) / regions.face_scale(landmarks)
    return u, depth


def test_tearbag_is_brightest_under_the_iris_and_fades_toward_both_corners(large_face, fake_ctx) -> None:
    """At the default width: not a short highlight next to the inner corner."""
    image = flat()
    diff = change(image, apply_bag(image, large_face, fake_ctx, shadow=0.0))
    for side in regions.SIDES:
        u, depth = lid_coordinates(large_face, side, image.shape[:2])
        area = (depth > -0.1) & (depth < 0.2) & (u > -0.2) & (u < 1.4)
        centroid_u = float((u[area] * diff[area]).sum() / diff[area].sum())
        assert 0.42 < centroid_u < 0.58, side
        lit = u[area & (diff > 0.25 * diff.max())]
        assert lit.min() < 0.2 and 0.8 < lit.max() < 1.05, side

        def level(position: float) -> float:
            return float(diff[area & (np.abs(u - position) < 0.03)].max() / diff.max())

        assert level(0.5) > 0.95
        assert level(0.1) < 0.45  # the inner corner stays quiet
        assert level(0.9) < 0.55


def test_tearbag_band_and_shadow_lie_where_they_should(large_face, fake_ctx) -> None:
    """The light right under the lashes, the shadow along the lower edge of the bag."""
    image = flat()
    signed = (apply_bag(image, large_face, fake_ctx, strength=1.0, width=1.0, shadow=1.0) - image)[..., 2]
    scale = regions.face_scale(large_face)
    down = contour_module._face_down(large_face)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        middle = large_face[s.eye_lower[3:6]].mean(axis=0)
        profile, length = section(signed, middle, middle + down * np.float32(0.3 * scale))
        depth = np.linspace(0.0, length / scale, len(profile))
        assert 0.03 < depth[int(np.argmax(profile))] < 0.075, side
        assert 0.075 < depth[int(np.argmin(profile))] < 0.13, side


def test_tearbag_shadow_starts_later_than_the_light(large_face, fake_ctx) -> None:
    """A shadow under the inner corner reads as a dark circle."""
    image = flat()
    settings = {"strength": 1.0, "width": 1.0, "softness": 0.6}
    without = apply_bag(image, large_face, fake_ctx, shadow=0.0, **settings)
    shadow = change(without, apply_bag(image, large_face, fake_ctx, shadow=1.0, **settings))
    for side in regions.SIDES:
        u, depth = lid_coordinates(large_face, side, image.shape[:2])
        below = (depth > 0.0) & (depth < 0.2)
        assert float(shadow[below & (np.abs(u - 0.5) < 0.03)].max()) > 10.0 * INVISIBLE, side
        assert float(shadow[below & (u > -0.1) & (u < 0.12)].max()) < INVISIBLE, side


def test_tearbag_softness_slider_and_no_hard_edge(large_face, fake_ctx) -> None:
    image = flat()
    scale = regions.face_scale(large_face)
    down = contour_module._face_down(large_face)
    middle = large_face[regions.side_indices("left").eye_lower[3:6]].mean(axis=0)

    def fade_length(softness: float) -> float:
        """How far below its peak the band takes to fade to a tenth, in face scales."""
        diff = change(image, apply_bag(image, large_face, fake_ctx, softness=softness, shadow=0.0))
        profile, length = section(diff, middle, middle + down * np.float32(0.3 * scale))
        peak = int(np.argmax(profile))
        return float(np.argmax(profile[peak:] < 0.1 * profile[peak])) / len(profile) * length / scale

    assert fade_length(0.0) < 0.8 * fade_length(0.6) < 0.8 * 0.9 * fade_length(1.0)
    diff = change(image, apply_bag(image, large_face, fake_ctx, shadow=0.0))
    assert steepest_slope(diff, scale) < 36.0


def test_tearbag_width_near_zero_fades_away(large_face, fake_ctx) -> None:
    """No pop when the slider leaves 0."""
    image = flat()

    def peak(width: float) -> float:
        return float(change(image, apply_bag(image, large_face, fake_ctx, width=width)).max())

    assert 0.0 < peak(0.01) < 0.15 * peak(0.5)
    assert peak(0.05) < peak(0.1) < peak(0.2)


@pytest.mark.parametrize(
    ("skin", "lit", "dark", "lit_max", "dark_max"),
    [
        (LIGHT_SKIN, (0.5, 2.0), (-1.5, -0.1), 2.0, -9.0),
        (MEDIUM_SKIN, (4.5, 8.5), (-1.5, -0.1), 11.0, -6.0),
        (DARK_SKIN, (4.0, 8.0), (-1.2, -0.05), 14.0, -3.5),
    ],
    ids=SKINS.keys(),
)
def test_tearbag_strength_is_pinned(large_face, fake_ctx, skin, lit, dark, lit_max, dark_max) -> None:
    """The defaults show on every skin without becoming a crescent; at full strength it stays bounded."""
    image = flat(skin)
    delta = lab_lightness(apply_bag(image, large_face, fake_ctx)) - lab_lightness(image)
    assert lit[0] < float(delta.max()) < lit[1]
    assert dark[0] < float(delta.min()) < dark[1]
    delta = lab_lightness(apply_bag(image, large_face, fake_ctx, strength=1.0, width=1.0, shadow=1.0)) - lab_lightness(image)
    assert float(delta.max()) < lit_max
    assert float(delta.min()) > dark_max


def test_a_full_smile_only_makes_the_bag_a_little_fuller(large_face, fake_ctx) -> None:
    image = flat()
    scale = regions.face_scale(large_face)
    resting = large_face.copy()
    center = (resting[regions.MOUTH_LEFT_CORNER] + resting[regions.MOUTH_RIGHT_CORNER]) / 2
    half = resting[regions.MOUTH_LEFT_CORNER] - resting[regions.MOUTH_RIGHT_CORNER]
    half = half / np.linalg.norm(half) * np.float32(0.45 * scale)
    resting[regions.MOUTH_LEFT_CORNER], resting[regions.MOUTH_RIGHT_CORNER] = center + half, center - half
    smiling = wider_mouth(resting, 1.5)
    calm = change(image, apply_bag(image, resting, fake_ctx, shadow=0.0))
    full = change(image, apply_bag(image, smiling, fake_ctx, shadow=0.0))
    assert 1.05 < float((full > 0.25 * calm.max()).sum()) / float((calm > 0.25 * calm.max()).sum()) < 1.6
    assert 1.02 < float(full.max()) / float(calm.max()) < 1.4
    u, depth = lid_coordinates(smiling, "left", image.shape[:2])
    below = depth[(full > 0.5 * full.max()) & (u > 0.3) & (u < 0.7)]
    assert below.max() < 0.14  # still right under the lashes


def blink(landmarks: np.ndarray, closed: float) -> np.ndarray:
    """``closed`` = 1: the upper lash lines rest on the lower ones."""
    out = landmarks.copy()
    for side in regions.SIDES:
        s = regions.side_indices(side)
        out[s.eye_upper] = landmarks[s.eye_upper] + closed * (landmarks[s.eye_lower] - landmarks[s.eye_upper])
    return out


def test_tearbag_stays_put_through_a_blink(frame, face_landmarks_720p, fake_ctx) -> None:
    """The lower lid hardly moves in a blink: the bag stays where it was, no pop."""
    opened = change(frame, apply_bag(frame, face_landmarks_720p, fake_ctx, strength=0.6, shadow=0.0))
    shut = change(frame, apply_bag(frame, blink(face_landmarks_720p, 1.0), fake_ctx, strength=0.6, shadow=0.0))
    area = opened > 0.25 * opened.max()
    assert np.abs(shut - opened)[area].mean() < 0.15 * opened[area].mean()


def test_tearbag_follows_the_lower_lid(face_landmarks_720p, fake_ctx) -> None:
    """When the lower lash line rises (a smile, a squint) the bag rises with it."""
    image = flat()
    landmarks = face_landmarks_720p
    raised = landmarks.copy()
    down = contour_module._face_down(landmarks)
    lift = 0.05 * regions.face_scale(landmarks)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        profile = np.sin(np.linspace(0.0, math.pi, len(s.eye_lower)))[:, None].astype(np.float32)
        raised[s.eye_lower] = landmarks[s.eye_lower] - down * np.float32(lift) * profile
    before = centroid(change(image, apply_bag(image, landmarks, fake_ctx, strength=1.0, shadow=0.0)))
    after = centroid(change(image, apply_bag(image, raised, fake_ctx, strength=1.0, shadow=0.0)))
    assert (before - after) @ down > 0.3 * lift


def test_narrowed_eyes_soften_the_bag_shadow(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    narrowed = blink(face_landmarks_720p, 0.75)
    values = {"strength": 1.0, "width": 1.0, "shadow": 1.0}

    def darkest(landmarks: np.ndarray) -> float:
        return float((apply_bag(image, landmarks, fake_ctx, **values) - image)[..., 2].min())

    assert darkest(narrowed) > 0.6 * darkest(face_landmarks_720p)  # less negative


@pytest.mark.parametrize("skin", SKINS.values(), ids=SKINS.keys())
def test_tearbag_color_is_relative_to_the_local_skin(face_landmarks_720p, fake_ctx, skin) -> None:
    """The bag is that skin lit up: never a whitish crescent on dark skin, never clipped on light skin."""
    image = flat(skin)
    out = apply_bag(image, face_landmarks_720p, fake_ctx, strength=1.0, width=1.0, shadow=0.0)
    peak = np.unravel_index(np.argmax(change(image, out)), out.shape[:2])
    lit = out[peak]
    ratio = lit / np.float32(skin)
    assert ratio.max() - ratio.min() < 0.1  # hue kept
    assert float(lit.max()) < 1.0
    assert float(ratio.mean()) < 1.45
    assert (lit[2] - lit[0]) / lit[2] > 0.85 * (skin[2] - skin[0]) / skin[2]  # as saturated as the skin


def test_tearbag_uses_the_skin_under_it_not_a_fixed_color(face_landmarks_720p, fake_ctx) -> None:
    lifts = {}
    for name, skin in SKINS.items():
        image = flat(skin)
        out = apply_bag(image, face_landmarks_720p, fake_ctx, strength=1.0, shadow=0.0)
        peak = np.unravel_index(np.argmax(change(image, out)), out.shape[:2])
        lifts[name] = out[peak] / np.float32(skin)
    assert lifts["light"].max() < 1.06  # no headroom left
    assert lifts["light"].max() < lifts["medium"].min()
    assert lifts["medium"].max() < lifts["dark"].min()  # plenty of headroom, still a relative lift


def wider_mouth(landmarks: np.ndarray, factor: float = 1.25) -> np.ndarray:
    out = landmarks.copy()
    center = (landmarks[regions.MOUTH_LEFT_CORNER] + landmarks[regions.MOUTH_RIGHT_CORNER]) / 2
    for index in (regions.MOUTH_LEFT_CORNER, regions.MOUTH_RIGHT_CORNER):
        out[index] = center + (landmarks[index] - center) * factor
    return out


def test_tearbag_fills_out_smoothly_when_smiling(frame, face_landmarks_720p, fake_ctx) -> None:
    resting = face_landmarks_720p.copy()
    center = (resting[regions.MOUTH_LEFT_CORNER] + resting[regions.MOUTH_RIGHT_CORNER]) / 2
    scale = regions.face_scale(resting)
    half = (resting[regions.MOUTH_LEFT_CORNER] - resting[regions.MOUTH_RIGHT_CORNER])
    half = half / np.linalg.norm(half) * np.float32(0.45 * scale)  # mouth width 0.9 face scales: no smile
    resting[regions.MOUTH_LEFT_CORNER] = center + half
    resting[regions.MOUTH_RIGHT_CORNER] = center - half
    smiling = wider_mouth(resting, 1.3)
    effect = TearbagEffect()
    amounts = [float(change(frame, apply_bag(frame, resting, fake_ctx, effect=effect)).sum())]
    for _ in range(12):
        amounts.append(float(change(frame, apply_bag(frame, smiling, fake_ctx, effect=effect)).sum()))
    assert amounts[-1] > 1.1 * amounts[0]
    assert amounts == sorted(amounts)  # grows over frames, no jump
    assert max(np.diff(amounts)) < 0.5 * (amounts[-1] - amounts[0])
    # reset() forgets the smile.
    effect.reset()
    reset = apply_bag(frame, resting, fake_ctx, effect=effect)
    fresh = apply_bag(frame, resting, fake_ctx)
    assert np.array_equal(reset, fresh)


# --------------------------------------------------------------------------
# Real motion: the tracker's lag and the pixels changing under the bag.
# --------------------------------------------------------------------------

SCLERA = (0.82, 0.84, 0.86)
IRIS = (0.1, 0.12, 0.16)
LASH = (0.08, 0.08, 0.1)


def eye_scene(landmarks: np.ndarray, skin=LIGHT_SKIN, shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """Flat skin with white eyes and dark irises exactly where ``landmarks`` say the eyes are."""
    image = flat(skin, shape)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        cv2.fillPoly(image, [np.round(landmarks[s.eye]).astype(np.int32)], SCLERA, cv2.LINE_AA)
        iris_radius = int(round(0.11 * regions.face_scale(landmarks)))
        center = np.round(landmarks[s.iris_center]).astype(int)
        cv2.circle(image, (int(center[0]), int(center[1])), iris_radius, IRIS, -1, cv2.LINE_AA)
    return image


def true_eye(landmarks: np.ndarray, shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """The inside of the real eye opening (one pixel in from its outline)."""
    roi = regions.Roi(0, 0, shape[1], shape[0])
    mask = np.zeros(shape, bool)
    for side in regions.SIDES:
        mask |= regions.eye_opening_mask(landmarks, side, roi) > 0.9
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def nod(landmarks: np.ndarray, depth: float, frames: int) -> list[np.ndarray]:
    """The real landmarks through a nod down and back up, ``depth`` face scales, in ``frames`` frames each way."""
    rest = [landmarks] * 8
    offset = depth * regions.face_scale(landmarks)
    ramp = np.linspace(0.0, 1.0, frames + 1)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    down = [(landmarks + np.float32([0.0, offset * t])).astype(np.float32) for t in ramp]
    up = [(landmarks + np.float32([0.0, offset * t])).astype(np.float32) for t in ramp[::-1]]
    return rest + down + up + rest


def tracked(script: list[np.ndarray], smoothing: float = 0.5) -> list[np.ndarray]:
    """The same landmarks after the tracker's own One Euro filter at 30 fps."""
    params = smoothing_params(smoothing)
    assert params is not None
    smoother = OneEuroFilter(params.min_cutoff, params.beta / face_scale_px(script[0]), params.d_cutoff)
    return [smoother(points.astype(np.float64), k / 30.0).astype(np.float32) for k, points in enumerate(script)]


def eye_paint_in_a_nod(large_face: np.ndarray, fake_ctx, values: dict, smoothing: float = 0.5, depth: float = 0.6, frames: int = 7) -> tuple[float, float]:
    """(brightest lift on the real eye in 8-bit levels, largest lag behind the real lower lid in face scales)."""
    script = nod(large_face, depth, frames)
    effect = TearbagEffect()
    worst, lag = 0.0, 0.0
    scale = regions.face_scale(large_face)
    s = regions.side_indices("left")
    for real, smoothed in zip(script, tracked(script, smoothing), strict=True):
        image = eye_scene(real, MEDIUM_SKIN)
        out = effect.apply(image.copy(), fake_ctx(image, smoothed), TearbagSettings(**values))
        worst = max(worst, float((out - image)[true_eye(real)].max()) * 255.0)
        lag = max(lag, float((real[s.eye_lower] - smoothed[s.eye_lower])[:, 1].mean()) / scale)
    return worst, lag


@pytest.mark.parametrize("values", [{}, BAG_STRONG], ids=["default", "max"])
def test_bag_never_lights_the_real_eye_in_a_nod(large_face, fake_ctx, values) -> None:
    """The smoothed landmarks trail the real eye: a band hanging off them would sit on its white."""
    worst, lag = eye_paint_in_a_nod(large_face, fake_ctx, values)
    assert lag > 0.04  # the tracker really is behind in this nod, nearly as in the worst real nods
    assert worst < 4.0


@pytest.mark.parametrize("values", [{}, BAG_STRONG], ids=["default", "max"])
def test_bag_stays_off_the_real_eye_at_the_heaviest_tracking_smoothing(large_face, fake_ctx, values) -> None:
    """The face-tracking slider at its top, in a deep fast nod: the lag is twice the usual one."""
    worst, lag = eye_paint_in_a_nod(large_face, fake_ctx, values, smoothing=1.0, depth=1.0)
    assert lag > 0.08
    assert worst < 4.0


def test_the_nod_test_can_fail(large_face, fake_ctx, monkeypatch) -> None:
    """Without the lag compensation the same nod lights the white of the eye."""
    monkeypatch.setattr(contour_module, "_LID_LAG_PER_STEP", 0.0)
    worst, _ = eye_paint_in_a_nod(large_face, fake_ctx, {})
    assert worst > 10.0


def test_bag_stays_under_the_real_lid_in_a_nod(large_face, fake_ctx) -> None:
    """It moves with the real lid, so the bag neither dims nor swims on the skin in a nod."""
    script = nod(large_face, 0.6, 7)
    effect = TearbagEffect()
    scale = regions.face_scale(large_face)
    down = contour_module._face_down(large_face)
    s = regions.side_indices("left")
    peaks = []
    for real, smoothed in zip(script, tracked(script), strict=True):
        image = flat(LIGHT_SKIN)
        signed = effect.apply(image.copy(), fake_ctx(image, smoothed), TearbagSettings(shadow=0.0)) - image
        middle = real[s.eye_lower[3:6]].mean(axis=0)
        profile, length = section(signed[..., 1], middle - down * np.float32(0.1 * scale), middle + down * np.float32(0.2 * scale))
        peaks.append((float(profile.max()), float(np.argmax(profile)) / len(profile) * length / scale - 0.1))
    level = np.array([p[0] for p in peaks])
    assert level.min() > 0.75 * level.max()
    # Moving down the band keeps its place under the real lid; moving up it may fall behind, on the skin.
    moving_down = range(9, 16)
    assert all(0.02 < peaks[k][1] < 0.11 for k in moving_down)


def test_lid_lag_estimate() -> None:
    scale = 100.0
    lag = contour_module._lid_lag
    assert lag(0.0, scale) == 0.0
    assert lag(-5.0, scale) == 0.0  # moving up: nothing to do
    assert lag(0.1, scale) == 0.0  # jitter
    assert 1.0 < lag(1.0, scale) < 2.5
    assert lag(50.0, scale) == pytest.approx(contour_module._LID_LAG_MAX * scale)
    assert lag(1.0, 200.0) < lag(1.0, 100.0)


def test_jitter_does_not_move_the_bag_along_the_lid_lag(large_face, fake_ctx) -> None:
    """Tracking noise stays in the dead zone: a still face with noise looks like a still face."""
    image = flat()
    rng = np.random.default_rng(5)
    effect = TearbagEffect()
    still = apply_bag(image, large_face, fake_ctx)
    noisy = [large_face + rng.normal(0.0, 0.08, large_face.shape).astype(np.float32) for _ in range(6)]
    for landmarks in noisy:
        out = effect.apply(image.copy(), fake_ctx(image, landmarks), TearbagSettings())
        assert float(np.percentile(change(still, out), 99.9)) < 2.0 * INVISIBLE


def test_a_jump_is_not_motion(large_face, fake_ctx) -> None:
    """A face found again somewhere else starts afresh, without a band far ahead of the lid."""
    image = flat()
    moved = large_face + np.float32([0.0, 0.5 * regions.face_scale(large_face)])
    effect = TearbagEffect()
    apply_bag(image, large_face, fake_ctx, effect=effect)
    assert np.array_equal(apply_bag(image, moved, fake_ctx, effect=effect), apply_bag(image, moved, fake_ctx))


def blink_scene(landmarks: np.ndarray, closed: float, skin=LIGHT_SKIN) -> np.ndarray:
    """What the camera sees in a blink: the lid over the eye, and the lashes reaching down over the bag."""
    open_eyes = eye_scene(landmarks, skin)
    if closed <= 0.0:
        return open_eyes
    scale = regions.face_scale(landmarks)
    down = contour_module._face_down(landmarks)
    image = open_eyes
    for side in regions.SIDES:
        s = regions.side_indices(side)
        lid = np.float32(skin) * np.float32(0.92)
        cv2.fillPoly(image, [np.round(landmarks[s.eye]).astype(np.int32)], tuple(float(c) for c in lid), cv2.LINE_AA)
        length = (0.02 + 0.03 * closed) * scale
        for point in regions.resample_polyline(landmarks[s.eye_lower], 14):
            tip = point + down * np.float32(length)
            cv2.line(image, tuple(np.round(point).astype(int).tolist()), tuple(np.round(tip).astype(int).tolist()), LASH, 2, cv2.LINE_AA)
        cv2.polylines(image, [np.round(landmarks[s.eye_lower]).astype(np.int32)], False, LASH, 3, cv2.LINE_AA)
    return image


def bag_gain_through_a_blink(large_face: np.ndarray, fake_ctx, values: dict) -> np.ndarray:
    """Per frame: relative lift of a patch of plain skin on the lower part of the bag, both eyes."""
    closing = [0.0] * 8 + [0.35, 0.8, 1.0, 1.0, 0.7, 0.3] + [0.0] * 6
    scale = regions.face_scale(large_face)
    down = contour_module._face_down(large_face)
    effect = TearbagEffect()
    gains = []
    for closed in closing:
        landmarks = blink(large_face, closed)
        image = blink_scene(landmarks, closed)
        out = effect.apply(image.copy(), fake_ctx(image, landmarks), TearbagSettings(**values))
        row = []
        for side in regions.SIDES:
            s = regions.side_indices(side)
            # Below the longest lashes, still inside the bag.
            point = landmarks[s.eye_lower[3:6]].mean(axis=0) + down * np.float32(0.065 * scale)
            x, y = int(point[0]), int(point[1])
            row.append(float((out[y - 1 : y + 2, x - 3 : x + 4] / image[y - 1 : y + 2, x - 3 : x + 4]).mean()))
        gains.append(row)
    return np.array(gains)


@pytest.mark.parametrize(("values", "limit"), [({"shadow": 0.0}, 0.006), ({**BAG_STRONG, "shadow": 0.0}, 0.012)], ids=["default", "max"])
def test_a_blink_does_not_pulse_the_bag(large_face, fake_ctx, values, limit) -> None:
    """The lashes coming down must not count as darker skin (which would mean more lift)."""
    gains = bag_gain_through_a_blink(large_face, fake_ctx, values)
    assert float(gains.max() - 1.0) > 0.02
    assert float((gains.max(axis=0) - gains.min(axis=0)).max()) < limit


def test_skin_measure_ignores_lashes_and_eye_white() -> None:
    skin = np.float32(MEDIUM_SKIN)
    patch = np.empty((40, 60, 3), np.float32)
    patch[...] = skin
    weight = np.ones((40, 60), np.float32)
    assert contour_module._measure_skin(patch, weight) == pytest.approx(skin, abs=1e-3)
    patch[:12] = 0.05  # lashes over 30 %
    patch[-6:] = (0.9, 0.9, 0.9)  # some eye white
    assert contour_module._measure_skin(patch, weight) == pytest.approx(skin, abs=1e-3)
    assert contour_module._measure_skin(patch, np.zeros((40, 60), np.float32)) is None


def test_bag_color_follows_the_light_over_several_frames_and_reset_forgets_it(large_face, fake_ctx) -> None:
    """Per-frame statistics are smoothed, with their state dropped by reset()."""
    bright, dim = flat(MEDIUM_SKIN), flat(tuple(0.6 * c for c in MEDIUM_SKIN))

    def lift(image: np.ndarray, effect: TearbagEffect) -> float:
        out = effect.apply(image.copy(), fake_ctx(image, large_face), TearbagSettings(shadow=0.0))
        return float((out / image).max())

    settled = TearbagEffect()
    fresh_bright, fresh_dim = lift(bright, TearbagEffect()), lift(dim, TearbagEffect())
    assert fresh_dim > fresh_bright + 0.01  # more headroom on dim skin
    assert lift(bright, settled) == pytest.approx(fresh_bright)
    first = lift(dim, settled)
    assert first < fresh_bright + 0.25 * (fresh_dim - fresh_bright)  # no jump on the first frame
    for _ in range(40):
        last = lift(dim, settled)
    assert last == pytest.approx(fresh_dim, abs=0.002)  # but it does get there
    settled.reset()
    assert lift(bright, settled) == pytest.approx(fresh_bright)


def test_a_blink_hardly_touches_the_shadow_but_narrowed_eyes_do(large_face, fake_ctx) -> None:
    image = flat()
    effect = TearbagEffect()
    values = {"strength": 1.0, "width": 1.0, "shadow": 1.0}

    def darkest(landmarks: np.ndarray, target: TearbagEffect) -> float:
        return float((target.apply(image.copy(), fake_ctx(image, landmarks), TearbagSettings(**values)) - image)[..., 2].min())

    opened = darkest(large_face, effect)
    shut = blink(large_face, 1.0)
    during = [darkest(shut, effect) for _ in range(4)]
    assert min(during) < 0.0
    assert max(during) < 0.7 * opened  # at most 30 % weaker after a long blink
    for _ in range(70):
        narrowed = darkest(shut, effect)
    assert narrowed == pytest.approx(darkest(shut, TearbagEffect()), rel=0.02)  # eyes that stay narrowed
    assert narrowed > 0.5 * opened


def test_turning_the_bag_off_forgets_what_it_remembered(large_face, fake_ctx) -> None:
    image = flat()
    effect = TearbagEffect()
    apply_bag(image, large_face, fake_ctx, effect=effect)
    assert effect._eyes
    assert np.array_equal(apply_bag(image, large_face, fake_ctx, effect=effect, strength=0.0), image)
    assert not effect._eyes and effect._smile is None


# --------------------------------------------------------------------------
# Temporal behavior and robustness.
# --------------------------------------------------------------------------


def test_same_input_gives_the_same_output(frame, face_landmarks_720p, fake_ctx) -> None:
    contour, bag = ContourEffect(), TearbagEffect()
    results = []
    for _ in range(3):
        ctx = fake_ctx(frame, face_landmarks_720p, face_skin_mask=full_skin_mask(frame))
        out = contour.apply(frame.copy(), ctx, ContourSettings())
        results.append(bag.apply(out, ctx, TearbagSettings()))
    assert np.array_equal(results[0], results[1]) and np.array_equal(results[1], results[2])
    contour.reset()
    bag.reset()
    ctx = fake_ctx(frame, face_landmarks_720p, face_skin_mask=full_skin_mask(frame))
    again = bag.apply(contour.apply(frame.copy(), ctx, ContourSettings()), ctx, TearbagSettings())
    assert np.array_equal(again, results[0])


def test_small_landmark_jitter_barely_changes_the_contour(frame, face_landmarks_720p, fake_ctx) -> None:
    """Sub-pixel motion must not make the soft masks jump (grid rounding, stroke steps)."""
    mask = full_skin_mask(frame)
    rng = np.random.default_rng(3)
    previous = None
    means, peaks = [], []
    for k in range(12):
        landmarks = face_landmarks_720p + np.float32([k * 0.13, k * 0.07]) + rng.normal(0.0, 0.1, face_landmarks_720p.shape).astype(np.float32)
        out = apply_contour(frame, landmarks, fake_ctx, face_skin_mask=mask)
        if previous is not None:
            diff = change(previous, out)
            means.append(float(diff.mean()))
            peaks.append(float(diff.max()))
        previous = out
    assert max(means) < 0.25 * INVISIBLE
    assert np.median(peaks) < 5.0 * INVISIBLE


def test_tearbag_is_steady_under_small_landmark_jitter(frame, face_landmarks_720p, fake_ctx) -> None:
    rng = np.random.default_rng(4)
    effect = TearbagEffect()
    outs = [
        apply_bag(frame, face_landmarks_720p + rng.normal(0.0, 0.1, face_landmarks_720p.shape).astype(np.float32), fake_ctx, effect=effect)
        for _ in range(8)
    ]
    diffs = [change(a, b) for a, b in zip(outs, outs[1:], strict=False)]
    assert max(float(d.mean()) for d in diffs) < 0.1 * INVISIBLE
    assert np.median([float(d.max()) for d in diffs]) < 8.0 * INVISIBLE


def grid_of(landmarks: np.ndarray, image: np.ndarray) -> tuple[int, int]:
    """The size of the coarse grid ContourEffect uses for ``landmarks`` (no forehead zone)."""
    scale = regions.face_scale(landmarks)
    roi = regions.roi_around(landmarks[regions.FACE_OVAL], image.shape, pad=0.0, margin_px=contour_module._ROI_MARGIN * scale)
    return contour_module._Grid.over(roi, scale).size


def test_grid_size_changes_do_not_jump(large_face, fake_ctx) -> None:
    """The ROI (and with it the grid size) changes by a pixel now and then: the result must not."""
    image = flat()
    factor = 1.0
    grown = large_face
    # The grid follows the face, so only a change of its proportions changes the grid's size.
    while grid_of(grown, image) == grid_of(large_face, image):
        factor += 0.0005
        assert factor < 1.03  # a small change of the face, and the grid did change
        grown = affine(large_face, np.diag([1.0, factor]), large_face[regions.NOSE_TIP])
    a = apply_contour(image, large_face, fake_ctx, shading=1.0, highlight=1.0)
    b = apply_contour(image, grown, fake_ctx, shading=1.0, highlight=1.0)
    assert np.percentile(change(a, b), 99.9) < 3.0 * INVISIBLE


@pytest.mark.parametrize(("apply", "limit"), [(apply_contour, 3.0), (apply_bag, 2.5)], ids=["contour", "tearbag"])
def test_slow_drift_of_a_large_face_is_smooth(large_face, fake_ctx, apply, limit) -> None:
    """Sub-pixel drift of the whole face: no pops from rounding of the ROI or the grid, and no
    rasterized edge that flips a run of pixels at once."""
    image = flat()
    steps = []
    previous = None
    for k in range(24):
        landmarks = (large_face + np.float32([0.17 * k, 0.11 * k])).astype(np.float32)
        out = apply(image, landmarks, fake_ctx)
        if previous is not None:
            steps.append(float(change(previous, out).max()))
        previous = out
    assert max(steps) < limit * INVISIBLE


def test_sensor_noise_does_not_reach_the_gain(large_face, fake_ctx) -> None:
    """The color factors come from a smooth version of the frame: pixel noise must not modulate them."""
    rng = np.random.default_rng(6)
    image = flat()
    noisy = np.clip(image + rng.normal(0.0, 0.02, image.shape).astype(np.float32), 0.0, 1.0)
    mask = full_skin_mask(image)
    clean_out = apply_contour(image, large_face, fake_ctx, face_skin_mask=mask)
    noisy_out = apply_contour(noisy, large_face, fake_ctx, face_skin_mask=mask)
    ratio = (noisy_out / noisy - clean_out / image)[..., 1]
    box = regions.roi_around(large_face[regions.FACE_OVAL], image.shape, pad=0.0).slices
    assert float(ratio[box].std()) < 0.0012


def test_the_grid_is_a_smooth_version_of_the_frame(large_face) -> None:
    """Bilinear samples alone keep two thirds of the pixel noise."""
    noise = np.random.default_rng(1).normal(0.5, 0.02, (500, 800)).astype(np.float32)
    grid = contour_module._Grid.over(regions.Roi(200, 100, 1000, 600), regions.face_scale(large_face))
    assert grid.coarse
    assert float(grid.downsample(noise).std()) < 0.4 * 0.02


def test_the_lip_mask_does_not_grow_with_the_face(face_landmarks_720p, fake_ctx, monkeypatch) -> None:
    """Protecting the lips must not bring back a full-resolution mask for a face close to the camera."""
    sizes = []
    original = contour_module._solid_shapes

    def spy(size, *args, **kwargs):
        sizes.append(size)
        return original(size, *args, **kwargs)

    monkeypatch.setattr(contour_module, "_solid_shapes", spy)
    image = flat()
    for scale in (141.0, 424.0):
        factor = np.float32(scale / regions.face_scale(face_landmarks_720p))
        landmarks = ((face_landmarks_720p - face_landmarks_720p[regions.NOSE_TIP]) * factor + np.float32([640, 330])).astype(np.float32)
        apply_contour(image, landmarks, fake_ctx, face_skin_mask=full_skin_mask(image))
    # Each run: the feature guard on the coarse grid, then the lip and nostril mask.
    assert len(sizes) == 4
    assert sizes[3][0] * sizes[3][1] < 1.3 * sizes[1][0] * sizes[1][1]


def test_bag_is_stable_on_a_noisy_frame(large_face, fake_ctx) -> None:
    rng = np.random.default_rng(7)
    image = flat()
    effect = TearbagEffect()
    lifts = []
    for _ in range(8):
        noisy = np.clip(image + rng.normal(0.0, 0.02, image.shape).astype(np.float32), 0.0, 1.0)
        out = effect.apply(noisy.copy(), fake_ctx(noisy, large_face), TearbagSettings(shadow=0.0))
        lifts.append(cv2.GaussianBlur(out / noisy, (0, 0), 2.0))
    assert float(np.max(np.std(np.stack(lifts), axis=0))) < 0.002


# Budgets in ms, minimum over a few runs (robust on a loaded machine; the nominal
# figure at 1280x720 on the development machine is about half of each).
CONTOUR_BUDGET_MS = 8.0
BAG_BUDGET_MS = 5.0


def fastest_ms(effect, image: np.ndarray, ctx, settings, runs: int = 12) -> float:
    best = math.inf
    for _ in range(runs):
        copy = image.copy()
        start = time.perf_counter()
        effect.apply(copy, ctx, settings)
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def face_of_size(landmarks: np.ndarray, target_scale: float) -> np.ndarray:
    factor = target_scale / regions.face_scale(landmarks)
    return ((landmarks - landmarks[regions.NOSE_TIP]) * np.float32(factor) + np.float32([640, 330])).astype(np.float32)


def test_cost_at_webcam_size_is_within_budget(face_landmarks_720p, fake_ctx) -> None:
    image = flat()
    landmarks = face_of_size(face_landmarks_720p, 141.0)
    ctx = fake_ctx(image, landmarks, face_skin_mask=full_skin_mask(image))
    assert fastest_ms(ContourEffect(), image, ctx, ContourSettings(**STRONG)) < CONTOUR_BUDGET_MS
    assert fastest_ms(TearbagEffect(), image, ctx, TearbagSettings(**BAG_STRONG)) < BAG_BUDGET_MS


def test_bag_cost_hardly_grows_with_the_face(face_landmarks_720p, fake_ctx) -> None:
    """Sitting close to a laptop must not take the bag out of its budget."""
    image = flat()
    settings = TearbagSettings()
    times = []
    for scale in (141.0, 424.0):
        landmarks = face_of_size(face_landmarks_720p, scale)
        times.append(fastest_ms(TearbagEffect(), image, fake_ctx(image, landmarks), settings))
    assert times[1] < 3.0 * times[0]
    assert times[1] < 1.5 * BAG_BUDGET_MS


# --------------------------------------------------------------------------
# More temporal behavior and robustness.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("apply", "values"), [(apply_contour, STRONG), (apply_bag, BAG_STRONG)], ids=["contour", "tearbag"])
def test_face_at_and_beyond_the_frame_border(frame, face_landmarks_720p, fake_ctx, apply, values) -> None:
    bottom = float(face_landmarks_720p[:, 1].max())
    partly = face_landmarks_720p + np.float32([0.0, frame.shape[0] - bottom + 40.0])
    out = apply(frame, partly, fake_ctx, **values)
    assert out.min() >= 0.0 and out.max() <= 1.0
    left = float(face_landmarks_720p[:, 0].min())
    sideways = face_landmarks_720p + np.float32([-left - 60.0, 0.0])
    out = apply(frame, sideways, fake_ctx, **values)
    assert out.min() >= 0.0 and out.max() <= 1.0
    gone = face_landmarks_720p + np.float32([0.0, 3000.0])
    assert np.array_equal(apply(frame, gone, fake_ctx, **values), frame)


@pytest.mark.parametrize(("apply", "values"), [(apply_contour, STRONG), (apply_bag, BAG_STRONG)], ids=["contour", "tearbag"])
def test_tiny_or_broken_faces_are_left_alone(frame, face_landmarks_720p, fake_ctx, apply, values) -> None:
    center = face_landmarks_720p[regions.NOSE_TIP]
    tiny = affine(face_landmarks_720p, np.eye(2) * 0.05, center)
    assert np.array_equal(apply(frame, tiny, fake_ctx, **values), frame)
    nan = np.full_like(face_landmarks_720p, np.nan)
    assert np.array_equal(apply(frame, nan, fake_ctx, **values), frame)


def test_read_only_landmarks_and_masks_are_not_modified(frame, face_landmarks_720p, fake_ctx) -> None:
    landmarks = face_landmarks_720p.copy()
    landmarks.setflags(write=False)
    mask = full_skin_mask(frame)
    mask.setflags(write=False)
    ctx = fake_ctx(frame, landmarks, face_skin_mask=mask)
    ctx.landmarks = landmarks
    image = frame.copy()
    image = ContourEffect().apply(image, ctx, ContourSettings(**STRONG))
    image = TearbagEffect().apply(image, ctx, TearbagSettings(**BAG_STRONG))
    assert change(frame, image).max() > 0.02


def test_works_after_the_skin_effect_and_keeps_skin_detail(frame, face_landmarks_720p, fake_ctx) -> None:
    ctx = fake_ctx(frame, face_landmarks_720p)
    settings = MakeupSettings()
    retouched = SkinEffect().apply(frame.copy(), ctx, settings.skin)
    out = ContourEffect().apply(retouched.copy(), ctx, settings.contour)
    out = TearbagEffect().apply(out, ctx, settings.tearbag)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Fine detail survives: high-pass energy on the cheek stays about the same.
    cheek = regions.roi_around(face_landmarks_720p[regions.RIGHT_CHEEK_CENTER], frame.shape, pad=0.0, margin_px=8.0).slices

    def detail(image: np.ndarray) -> float:
        return float((image - cv2.GaussianBlur(image, (0, 0), 2.0))[cheek].std())

    assert detail(out) > 0.8 * detail(retouched)


# --------------------------------------------------------------------------
# In the pipeline.
# --------------------------------------------------------------------------


class _Tracker:
    def __init__(self, script: list[np.ndarray | None]) -> None:
        self.script = script

    def detect(self, frame: np.ndarray, ts_ms: int):
        landmarks = self.script.pop(0) if self.script else None
        return (None, None) if landmarks is None else (landmarks.copy(), np.zeros(len(landmarks), np.float32))

    def set_smoothing(self, smoothing: float) -> None:
        pass

    def close(self) -> None:
        pass


class _Segmenter:
    def __init__(self, face_skin: np.ndarray | None = None) -> None:
        self.face_skin = face_skin

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        return SegMasks(face_skin=self.face_skin)

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_runs_in_the_pipeline_fades_and_resets(face_frame_720p, face_landmarks_720p) -> None:
    settings = MakeupSettings()
    settings.general.mirror_preview = False
    settings.skin.enabled = True
    script = [face_landmarks_720p] * 8 + [None] * 10
    mask = np.ones(face_frame_720p.shape[:2], np.float32)
    contour, bag = ContourEffect(), TearbagEffect()
    pipeline = MakeupPipeline(settings, tracker=_Tracker(script), segmenter=_Segmenter(mask), effects=[bag, SkinEffect(), contour])
    try:
        results = [pipeline.process(face_frame_720p, timestamp_ms=1000 + 33 * i) for i in range(18)]
    finally:
        pipeline.close()
    assert all(r.face_found for r in results[:8])
    assert {"contour", "tearbag", "skin"} <= set(results[7].timings_ms)
    # Effects run in their declared order whatever the constructor order.
    assert list(results[7].timings_ms).index("skin") < list(results[7].timings_ms).index("contour") < list(results[7].timings_ms).index("tearbag")
    made_up = np.abs(results[7].output.astype(np.int16) - face_frame_720p).max(axis=2)
    assert made_up.max() > 5
    assert np.array_equal(results[7].output[:50, :50], face_frame_720p[:50, :50])
    # The fade-in grows smoothly and the fade-out ends at the untouched frame.
    change_by_frame = [float(np.abs(r.output.astype(np.int16) - face_frame_720p).mean()) for r in results]
    assert change_by_frame[0] < change_by_frame[2] < change_by_frame[5]
    assert np.array_equal(results[-1].output, face_frame_720p)
    assert bag._smile is None  # reset when the face was lost

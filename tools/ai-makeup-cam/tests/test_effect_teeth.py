"""TeethEffect on the astronaut portrait (real landmarks): no-ops, locality,
whitening / yellow reduction / evenness behavior, what is never touched
(lips, tongue, an open mouth without teeth, a hand in front of it), temporal
state, close-ups, frame borders and a run through the pipeline."""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from makeup_cam import blend, regions
from makeup_cam.config import MakeupSettings, TeethSettings
from makeup_cam.effects import teeth as teeth_module
from makeup_cam.effects.base import registered_effects
from makeup_cam.effects.teeth import TeethEffect
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.segmentation import SegMasks

STRONG = {"whitening": 1.0, "yellow_reduction": 1.0, "evenness": 1.0, "feather": 1.0}
OFF = {"whitening": 0.0, "yellow_reduction": 0.0, "evenness": 0.0}


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


@pytest.fixture
def dim(frame: np.ndarray) -> np.ndarray:
    """The portrait in a dimmer, warmer light: yellower, darker teeth that
    whitening can lift (the original teeth are already near white)."""
    return np.clip(frame * np.float32([0.62, 0.8, 0.86]), 0.0, 1.0)


def apply_teeth(
    frame: np.ndarray,
    landmarks: np.ndarray | None,
    fake_ctx,
    intensity: float = 1.0,
    effect: TeethEffect | None = None,
    **values,
) -> np.ndarray:
    ctx = fake_ctx(frame, landmarks, intensity=intensity)
    return (effect or TeethEffect()).apply(frame.copy(), ctx, TeethSettings(**values))


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def distance_to_opening(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """True (Euclidean) distance in px of every pixel to the mouth opening.

    Not ``regions.expand_polygon``: that moves each vertex radially from the
    centroid, so on a flat mouth the flat edges barely move along their normals
    (measured on the portrait, for a 6.19 px expansion: 0.13-0.67 px on the
    upper edge). Its "outside" starts right next to the lip line.
    """
    outside = regions.polygon_mask(landmarks[regions.LIPS_INNER], full_roi(frame)) == 0.0
    return cv2.distanceTransform(outside.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def support_px(landmarks: np.ndarray, **values) -> float:
    """Where the effect promises to stop (3 feather sigmas), plus 2 px for two
    different rasterizations. A literal on purpose: the promise, not the constant."""
    sigma = teeth_module._feather(TeethSettings(**values)) * regions.face_scale(landmarks)
    return 3.0 * sigma + 2.0


def opening_mask(frame: np.ndarray, landmarks: np.ndarray, inset_px: float = 1.0) -> np.ndarray:
    """Boolean mask of pixels well inside the mouth opening."""
    inner = regions.expand_polygon(landmarks[regions.LIPS_INNER], -inset_px)
    return regions.polygon_mask(inner, full_roi(frame)) > 0.99


def teeth_mask(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """The brighter part of the mouth opening (the portrait shows her teeth)."""
    inside = opening_mask(frame, landmarks)
    lightness = blend.bgr_to_lab(frame)[..., 0]
    return inside & (lightness > np.percentile(lightness[inside], 40))


def lab_of(image: np.ndarray) -> np.ndarray:
    return blend.bgr_to_lab(image)


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def bgr(lightness: float, a: float, b: float) -> np.ndarray:
    """One Lab color as float32 BGR."""
    return blend.lab_to_bgr(np.float32([[[lightness, a, b]]]))[0, 0]


def zoomed(frame: np.ndarray, landmarks: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """The frame and landmarks magnified around the mouth (a face close to the camera)."""
    center = landmarks[regions.LIPS_OUTER].mean(axis=0)
    shift = np.float32([frame.shape[1] / 2, frame.shape[0] / 2]) - zoom * center
    matrix = np.float32([[zoom, 0, shift[0]], [0, zoom, shift[1]]])
    big = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    return big, (landmarks * zoom + shift).astype(np.float32)


def closed_mouth(landmarks: np.ndarray) -> np.ndarray:
    closed = landmarks.copy()
    seam = (closed[regions.LIPS_UPPER_INNER] + closed[regions.LIPS_LOWER_INNER]) / 2
    closed[regions.LIPS_UPPER_INNER] = seam
    closed[regions.LIPS_LOWER_INNER] = seam
    return closed


def painted_mouth(image: np.ndarray, landmarks: np.ndarray, painter) -> tuple[np.ndarray, np.ndarray]:
    """``image`` with the mouth opening repainted by ``painter(xs, ys) -> (N, 3) BGR``;
    returns the image and the painted mask."""
    out = image.copy()
    region = opening_mask(image, landmarks, inset_px=0.0) | (
        regions.polygon_mask(landmarks[regions.LIPS_INNER], full_roi(image)) > 0.2
    )
    ys, xs = np.nonzero(region)
    out[ys, xs] = painter(xs, ys)
    return out, region


# --------------------------------------------------------------------------
# Registration and no-ops
# --------------------------------------------------------------------------


def test_registered_under_its_settings_section() -> None:
    assert TeethEffect in registered_effects()
    assert (TeethEffect.name, TeethEffect.order, TeethEffect.needs_face) == ("teeth", 75, True)
    assert TeethEffect.order > 70  # after the lipstick
    assert isinstance(getattr(MakeupSettings(), TeethEffect.name), TeethSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "landmarks_kind"),
    [
        (1.0, OFF, "face"),
        (0.0, STRONG, "face"),
        (1.0, STRONG, "none"),
        (1.0, STRONG, "closed"),
    ],
    ids=["sliders-0", "intensity-0", "no-face", "closed-mouth"],
)
def test_exact_noop(dim, face_landmarks_720p, fake_ctx, intensity, values, landmarks_kind) -> None:
    landmarks = {"face": face_landmarks_720p, "none": None, "closed": closed_mouth(face_landmarks_720p)}[landmarks_kind]
    effect = TeethEffect()
    out = apply_teeth(dim, landmarks, fake_ctx, intensity=intensity, effect=effect, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, dim)
    assert effect.stats is None  # skipped before any measurement


def test_feather_alone_does_nothing(dim, face_landmarks_720p, fake_ctx) -> None:
    out = apply_teeth(dim, face_landmarks_720p, fake_ctx, **OFF, feather=1.0)
    assert np.array_equal(out, dim)


def test_closed_mouth_and_first_slit_skip_all_image_work(dim, face_landmarks_720p, fake_ctx, monkeypatch) -> None:
    def fail(image: np.ndarray) -> np.ndarray:
        raise AssertionError("image work on a closed mouth")

    monkeypatch.setattr(teeth_module.blend, "bgr_to_lab", fail)
    closed = closed_mouth(face_landmarks_720p)
    effect = TeethEffect()
    assert np.array_equal(apply_teeth(dim, closed, fake_ctx, effect=effect), dim)
    scale = regions.face_scale(face_landmarks_720p)
    # A slit too thin to measure, before any reference exists.
    openness = 0.02 * scale / teeth_module._opening(face_landmarks_720p)
    slit = (closed + openness * (face_landmarks_720p - closed)).astype(np.float32)
    assert teeth_module._OPEN_RAMP[0] < teeth_module._opening(slit) / scale < teeth_module._CONFIDENCE_OPENING[0]
    assert np.array_equal(apply_teeth(dim, slit, fake_ctx, effect=effect), dim)
    assert effect.stats is None


@pytest.mark.parametrize("light", ["original", "dim"])
def test_output_is_a_valid_frame(frame, dim, face_landmarks_720p, fake_ctx, light) -> None:
    image = frame if light == "original" else dim
    out = apply_teeth(image, face_landmarks_720p, fake_ctx, **STRONG)
    assert out.dtype == np.float32 and out.shape == image.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(image, out)[teeth_mask(image, face_landmarks_720p)].mean() > 0.005


# --------------------------------------------------------------------------
# Where it works
# --------------------------------------------------------------------------


def test_changes_stay_in_the_mouth_opening(dim, face_landmarks_720p, fake_ctx) -> None:
    out = apply_teeth(dim, face_landmarks_720p, fake_ctx, **STRONG)
    diff = change(dim, out)
    outside = distance_to_opening(dim, face_landmarks_720p) > support_px(face_landmarks_720p, **STRONG)
    assert diff[outside].max() == 0.0
    assert np.array_equal(out[:50, :50], dim[:50, :50])  # background corner
    # The lips (away from the teeth edge) keep their color.
    lips = regions.lips_mask(face_landmarks_720p, full_roi(dim), outline=-0.05) > 0.99
    lips &= regions.polygon_mask(regions.expand_polygon(face_landmarks_720p[regions.LIPS_INNER], 3.0), full_roi(dim)) == 0
    assert lips.sum() > 200
    assert diff[lips].max() < 0.01
    assert diff[teeth_mask(dim, face_landmarks_720p)].mean() > 0.02


@pytest.mark.parametrize("feather", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("zoom", [1.0, 2.0, 3.0, 5.0])
def test_nothing_changes_beyond_the_support(dim, face_landmarks_720p, fake_ctx, zoom, feather) -> None:
    """At every face size, including the reduced-resolution path (zoom 3 and 5),
    whose upsampling must not widen the support."""
    big, landmarks = zoomed(dim, face_landmarks_720p, zoom)
    values = {**STRONG, "feather": feather}
    diff = change(big, apply_teeth(big, landmarks, fake_ctx, **values))
    distance = distance_to_opening(big, landmarks)
    reach = support_px(landmarks, **values)
    assert diff[distance > reach].max() == 0.0
    assert diff[distance <= reach].max() > 0.01  # the effect did run


@pytest.mark.parametrize("zoom", [1.0, 3.0])
def test_support_holds_for_a_strongly_curved_mouth(dim, face_landmarks_720p, fake_ctx, zoom) -> None:
    """A big smile makes the opening concave: the feather's tails add up inside
    the curve, where a threshold on the blurred mask alone would leak."""
    big, landmarks = zoomed(dim, face_landmarks_720p, zoom)
    curved = landmarks.copy()
    lips = sorted(set(regions.LIPS_INNER + regions.LIPS_OUTER))
    center = curved[lips, 0].mean()
    half = (curved[lips, 0].max() - curved[lips, 0].min()) / 2
    curved[lips, 1] -= 0.5 * half * ((curved[lips, 0] - center) / half) ** 2
    diff = change(big, apply_teeth(big, curved, fake_ctx, **STRONG))
    assert diff[distance_to_opening(big, curved) > support_px(curved, **STRONG)].max() == 0.0


def test_support_window_cuts_the_feather_to_exactly_zero_at_its_reach() -> None:
    """A flat polygon: compact support, monotonic falloff, and inside the reach
    (up to the window's last sigma) the same mask as the plain Gaussian feather."""
    roi = regions.Roi(0, 0, 240, 120)
    polygon = np.float32([[40, 50], [200, 50], [200, 70], [40, 70]])
    sigma = 2.5
    window = teeth_module._support_window(polygon, roi, sigma)
    feather = regions.polygon_mask(polygon, roi, sigma)
    reach = teeth_module._support_reach(sigma)
    outside = regions.polygon_mask(polygon, roi) == 0.0
    distance = cv2.distanceTransform(outside.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Some slack: the window and this test rasterize the polygon differently (up to a pixel diagonal).
    assert window[distance >= reach + 1.5].max() == 0.0
    assert feather[distance >= reach + 1.5].max() > 0.0  # the Gaussian alone does not get there
    assert window[distance < reach - sigma - 1.5].min() == 1.0
    falloff = (feather * window)[50:0:-1, 120]  # upward from the polygon's top edge
    assert np.all(np.diff(falloff) <= 0.0)
    assert falloff[-1] == 0.0


def test_tongue_and_lipstick_colors_are_left_alone(dim, face_landmarks_720p, fake_ctx) -> None:
    """Upper half of the opening: yellowish teeth; lower half: a pink tongue."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    center_y = float(landmarks[regions.LIPS_INNER][:, 1].mean())
    teeth, tongue = bgr(72.0, 1.0, 24.0), bgr(58.0, 18.0, 12.0)
    painted, region = painted_mouth(big, landmarks, lambda xs, ys: np.where((ys < center_y)[:, None], teeth, tongue))
    out = apply_teeth(painted, landmarks, fake_ctx, **STRONG)
    diff = change(painted, out)
    ys, xs = np.nonzero(region)
    inner = opening_mask(painted, landmarks, inset_px=2.0)
    yy = np.arange(painted.shape[0])[:, None]
    tongue_core = inner & (yy > center_y + 3)
    teeth_core = inner & (yy < center_y - 3)
    assert tongue_core.sum() > 100 and teeth_core.sum() > 100
    assert diff[tongue_core].max() < 0.004
    assert diff[teeth_core].mean() > 0.03


def test_pink_that_is_not_very_red_is_left_alone(dim, face_landmarks_720p, fake_ctx) -> None:
    """A low a* passes the a* test, so only the hue test (a* - 0.6 b*) can tell
    a grayish-pink tongue from yellowish teeth."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    center_y = float(landmarks[regions.LIPS_INNER][:, 1].mean())
    teeth, pink = bgr(74.0, 1.0, 24.0), bgr(62.0, 4.5, 2.0)
    painted, _ = painted_mouth(big, landmarks, lambda xs, ys: np.where((ys < center_y)[:, None], teeth, pink))
    effect = TeethEffect()
    diff = change(painted, apply_teeth(painted, landmarks, fake_ctx, effect=effect, **STRONG))
    assert 4.5 < effect.stats.red_threshold()  # the a* test alone lets this pink through
    inner = opening_mask(painted, landmarks, inset_px=2.0)
    yy = np.arange(painted.shape[0])[:, None]
    assert diff[inner & (yy > center_y + 6)].max() < 0.006
    assert diff[inner & (yy < center_y - 3)].mean() > 0.03


def test_a_dark_interior_that_is_not_red_is_left_alone(dim, face_landmarks_720p, fake_ctx) -> None:
    """Far darker than the skin around the mouth is the shadowed inside of the
    mouth, however little red it is: the presence check's lightness test."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    painted, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(30.0, 0.0, 14.0), (len(xs), 1)))
    effect = TeethEffect()
    assert np.array_equal(apply_teeth(painted, landmarks, fake_ctx, effect=effect, **STRONG), painted)
    assert effect.stats.teeth_a < effect.stats.skin[1] - 4.5  # the red test alone would say "teeth"


def test_open_mouth_without_teeth_is_left_alone(dim, face_landmarks_720p, fake_ctx) -> None:
    """Only the tongue and the dark mouth cavity: the presence check stops everything."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    center_y = float(landmarks[regions.LIPS_INNER][:, 1].mean())
    cavity, tongue = bgr(14.0, 6.0, 4.0), bgr(55.0, 19.0, 13.0)
    painted, _ = painted_mouth(big, landmarks, lambda xs, ys: np.where((ys < center_y)[:, None], cavity, tongue))
    effect = TeethEffect()
    for _ in range(4):
        out = apply_teeth(painted, landmarks, fake_ctx, effect=effect, **STRONG)
        assert np.array_equal(out, painted)


# --------------------------------------------------------------------------
# What it does
# --------------------------------------------------------------------------


def test_whitening_lifts_the_teeth_below_a_ceiling(dim, face_landmarks_720p, fake_ctx) -> None:
    teeth = teeth_mask(dim, face_landmarks_720p)
    before = lab_of(dim)[..., 0]
    after = lab_of(apply_teeth(dim, face_landmarks_720p, fake_ctx, whitening=1.0, yellow_reduction=0.0, evenness=0.0))[..., 0]
    lift = after - before
    assert lift[teeth].mean() > 4.0
    assert lift[teeth].min() > -0.05  # never darkens
    assert after[teeth].max() <= max(float(before[teeth].max()), teeth_module._L_CAP) + 0.05
    # Texture survives: the teeth's lightness variation is not flattened.
    assert after[teeth].std() > 0.9 * before[teeth].std()


def test_whitening_leaves_already_white_teeth_alone(frame, face_landmarks_720p, fake_ctx) -> None:
    teeth = teeth_mask(frame, face_landmarks_720p)
    assert lab_of(frame)[..., 0][teeth].mean() > 85  # the portrait's own teeth are bright
    out = apply_teeth(frame, face_landmarks_720p, fake_ctx, whitening=1.0, yellow_reduction=0.0, evenness=0.0)
    assert np.abs(lab_of(out)[..., 0] - lab_of(frame)[..., 0])[teeth].mean() < 0.3


def test_yellow_reduction_moves_toward_a_warm_neutral(dim, face_landmarks_720p, fake_ctx) -> None:
    teeth = teeth_mask(dim, face_landmarks_720p)
    effect = TeethEffect()
    out = apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, whitening=0.0, yellow_reduction=1.0, evenness=0.0)
    before, after = lab_of(dim)[teeth], lab_of(out)[teeth]
    neutral_a, neutral_b = effect.stats.neutral()
    assert neutral_b >= teeth_module._NEUTRAL_FLOOR[1] and neutral_a >= teeth_module._NEUTRAL_FLOOR[0]
    assert (after[:, 2] - before[:, 2]).mean() < -5.0
    # Never past the neutral (no blue or gray teeth), never more yellow.
    assert np.all(after[:, 2] >= np.minimum(before[:, 2], neutral_b) - 0.1)
    assert np.all(after[:, 2] <= before[:, 2] + 0.1)
    assert np.abs(after[:, 0] - before[:, 0]).mean() < 0.5  # lightness is whitening's job


def whitening_lift(teeth_l: float, amount: float, lightness: float | None = None) -> float:
    """The L lift of a tooth pixel at ``lightness`` (default: the reference itself)."""
    value = np.float32([[teeth_l if lightness is None else lightness]])
    before = value.copy()
    teeth_module._whiten(value, before, np.float32([[1.0]]), teeth_l, amount)
    return float(value[0, 0] - before[0, 0])


def test_whitening_does_not_amplify_reference_jitter() -> None:
    """The lift depends gently on the teeth reference, also near the limit: a
    hard ceiling here moved the lift by about 1 L per unit of reference, which
    showed as flicker on bright teeth while talking."""
    references = np.arange(60.0, 96.0, 0.5)
    lifts = np.array([whitening_lift(float(r), 0.3, lightness=78.0) for r in references])
    assert np.abs(np.diff(lifts) / 0.5).max() < 0.45
    assert lifts[-1] == 0.0  # already white teeth stay as they are
    assert np.all(np.diff(lifts) <= 1e-6)


@pytest.mark.parametrize("amount", [0.3, 0.6, 1.0])
def test_whitening_keeps_the_reference_below_the_limit(amount) -> None:
    for reference in np.arange(40.0, 96.0, 1.0):
        assert reference + whitening_lift(float(reference), amount) <= max(float(reference), 89.5)


def test_yellow_reduction_follows_a_warm_light() -> None:
    """In warm light the neutral stays warm (full reduction on such footage reads gray)."""
    warm = teeth_module._Stats(80.0, 2.0, 32.0, (60.0, 14.0, 34.0))
    neutral_light = teeth_module._Stats(80.0, 2.0, 32.0, teeth_module._REFERENCE_SKIN)
    assert neutral_light.neutral() == pytest.approx(teeth_module._NEUTRAL_AB)
    assert warm.neutral()[1] > neutral_light.neutral()[1] + 8.0


def test_yellow_reduction_fades_out_on_dim_teeth(dim, face_landmarks_720p, fake_ctx) -> None:
    """A dark room (the whole frame dims, so the teeth stay "present"): dim teeth
    with less yellow read as gray, so the reduction backs off."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    painted, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(78.0, 1.0, 26.0), (len(xs), 1)))
    inner = opening_mask(big, landmarks, inset_px=3.0)

    def share_removed(light: float) -> float:
        image = painted * np.float32(light)
        effect = TeethEffect()
        out = apply_teeth(image, landmarks, fake_ctx, effect=effect, whitening=0.0, yellow_reduction=1.0, evenness=0.0)
        assert effect._presence > 0.5
        before, after = lab_of(image)[inner], lab_of(out)[inner]
        removed = (before[:, 2] - after[:, 2]).mean() / (before[:, 2].mean() - effect.stats.neutral()[1])
        return float(removed / effect._presence)

    assert share_removed(1.0) > 0.55
    assert share_removed(0.45) < 0.5 * share_removed(1.0)


def test_strength_grows_with_sliders_and_intensity(dim, face_landmarks_720p, fake_ctx) -> None:
    teeth = teeth_mask(dim, face_landmarks_720p)
    base = lab_of(dim)[teeth]

    def shift(intensity: float, **values) -> tuple[float, float]:
        out = lab_of(apply_teeth(dim, face_landmarks_720p, fake_ctx, intensity=intensity, **values))[teeth]
        return float((out[:, 0] - base[:, 0]).mean()), float((base[:, 2] - out[:, 2]).mean())

    whitening = [shift(1.0, whitening=w, yellow_reduction=0.0, evenness=0.0)[0] for w in (0.25, 0.5, 1.0)]
    yellow = [shift(1.0, whitening=0.0, yellow_reduction=y, evenness=0.0)[1] for y in (0.25, 0.5, 1.0)]
    by_intensity = [shift(i, **STRONG) for i in (0.25, 0.5, 1.0)]
    assert whitening == sorted(whitening) and whitening[0] > 0.5 and len(set(whitening)) == 3
    assert yellow == sorted(yellow) and yellow[0] > 0.5 and len(set(yellow)) == 3
    assert [s[0] for s in by_intensity] == sorted(s[0] for s in by_intensity)
    assert [s[1] for s in by_intensity] == sorted(s[1] for s in by_intensity)


def striped_teeth(image: np.ndarray, landmarks: np.ndarray, gap_lab, teeth_labs, period: int = 12, gap: int = 2):
    """Repaint the mouth opening as a row of teeth ``period`` px wide separated
    by ``gap`` px dark gaps; ``teeth_labs`` alternate from tooth to tooth."""
    x0 = int(landmarks[regions.LIPS_INNER][:, 0].min())
    colors = [bgr(*lab) for lab in teeth_labs]
    gap_color = bgr(*gap_lab)

    def painter(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        position = (xs - x0) % period
        tooth = ((xs - x0) // period) % len(colors)
        out = np.stack([colors[i] for i in tooth])
        out[position < gap] = gap_color
        return out

    painted, _ = painted_mouth(image, landmarks, painter)
    inner = opening_mask(image, landmarks, inset_px=3.0)
    columns = (np.arange(image.shape[1]) - x0) % period
    gaps = inner & (columns < gap)[None, :]
    teeth = inner & ((columns >= gap + 2) & (columns < period - 2))[None, :]
    return painted, gaps, teeth


def test_evenness_lifts_gaps_but_keeps_the_separations(dim, face_landmarks_720p, fake_ctx) -> None:
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    painted, gaps, teeth = striped_teeth(big, landmarks, (38.0, 2.0, 10.0), [(74.0, 1.0, 22.0)])
    assert gaps.sum() > 20 and teeth.sum() > 100
    out = apply_teeth(painted, landmarks, fake_ctx, whitening=0.0, yellow_reduction=0.0, evenness=1.0)
    before, after = lab_of(painted)[..., 0], lab_of(out)[..., 0]
    depth = before[teeth].mean() - before[gaps].mean()
    lift = after[gaps].mean() - before[gaps].mean()
    assert 0.1 * depth < lift <= teeth_module._GAP_LIFT * depth + 1.0
    # The separations stay clearly visible and the teeth are not darkened.
    assert after[teeth].mean() - after[gaps].mean() > 0.5 * depth
    assert after[teeth].mean() >= before[teeth].mean() - 0.05


@pytest.mark.parametrize("gap_lightness", [58.0, 64.0, 68.0])
def test_shallow_gaps_are_not_lifted_twice(dim, face_landmarks_720p, fake_ctx, gap_lightness) -> None:
    """Blurred webcam gaps are shallow: the blotch smoothing also reaches them,
    and on top of the gap lift it merged the teeth at high evenness."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    painted, gaps, teeth = striped_teeth(big, landmarks, (gap_lightness, 2.0, 18.0), [(74.0, 1.0, 22.0)])
    out = apply_teeth(painted, landmarks, fake_ctx, whitening=0.0, yellow_reduction=0.0, evenness=1.0)
    before, after = lab_of(painted)[..., 0], lab_of(out)[..., 0]
    depth = before[teeth].mean() - before[gaps].mean()
    lift = after[gaps].mean() - before[gaps].mean()
    assert 0.1 * depth < lift <= teeth_module._GAP_LIFT * depth + 0.3
    assert after[teeth].mean() - after[gaps].mean() >= (1.0 - teeth_module._GAP_LIFT) * depth - 0.3


@pytest.mark.parametrize("teeth_lightness", [74.0, 84.0, 90.0])
@pytest.mark.parametrize("depth", [6.0, 12.0, 30.0])
def test_all_sliders_at_maximum_keep_the_separations(dim, face_landmarks_720p, fake_ctx, teeth_lightness, depth) -> None:
    """Whitening's soft shoulder compresses bright lightness: applied per pixel it
    lifted a gap more than the teeth beside it, and on top of the evenness gap
    lift that inverted shallow gaps. The promise holds for the sum of the two."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    teeth_lab = (teeth_lightness, 1.0, 22.0)
    painted, gaps, teeth = striped_teeth(big, landmarks, (teeth_lightness - depth, 1.0, 22.0), [teeth_lab])
    out = apply_teeth(painted, landmarks, fake_ctx, **STRONG)
    before, after = lab_of(painted)[..., 0], lab_of(out)[..., 0]
    measured = before[teeth].mean() - before[gaps].mean()
    assert after[teeth].mean() - after[gaps].mean() >= 0.6 * measured


def test_whitening_alone_does_not_narrow_a_gap(dim, face_landmarks_720p, fake_ctx) -> None:
    """Gaps inherit the lift of the tooth surface around them instead of having
    their own tone curve, so the depth stays, also where the shoulder compresses."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    painted, gaps, teeth = striped_teeth(big, landmarks, (58.0, 1.0, 22.0), [(80.0, 1.0, 22.0)])
    out = apply_teeth(painted, landmarks, fake_ctx, whitening=1.0, yellow_reduction=0.0, evenness=0.0)
    before, after = lab_of(painted)[..., 0], lab_of(out)[..., 0]
    assert after[teeth].mean() > before[teeth].mean() + 0.5  # it did whiten
    assert after[teeth].mean() - after[gaps].mean() >= before[teeth].mean() - before[gaps].mean() - 0.5


def test_the_shoulder_does_not_close_a_gap_between_bright_teeth() -> None:
    """Bright teeth above a dimmer reference, where the tone curve is at its flattest.
    A curve per pixel lifts a gap at L 78 by about 11 and the teeth at L 88 by about 4."""
    lightness = np.full((40, 80), 88.0, np.float32)
    lightness[:, 10::12] = 78.0
    lightness[:, 11::12] = 78.0
    lab = np.dstack([lightness, np.full_like(lightness, 1.0), np.full_like(lightness, 20.0)])
    stats = teeth_module._Stats(72.0, 1.0, 20.0, None)
    levels = teeth_module._Levels(whitening=1.0, yellow=0.0, evenness=0.0)
    out = teeth_module._make_up(lab, lab[..., 1:], stats, levels, 70.0, None)[..., 0]
    lift = out - lightness
    assert lift[20, 40] > 2.0  # it did whiten
    assert out[20, 40] - out[20, 34] >= 10.0 - 0.5


def test_a_row_below_a_lip_shadow_is_not_a_gap(dim, face_landmarks_720p, fake_ctx) -> None:
    """A dark band as wide as the mouth has bright teeth above or below it but
    not beside it: it is a shadow and the gap lift must not fill it in."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    inner = regions.polygon_mask(landmarks[regions.LIPS_INNER], full_roi(big)) > 0.99
    rows = np.flatnonzero(inner.any(axis=1))
    middle = (rows[0] + rows[-1]) // 2
    teeth, shadow = bgr(76.0, 1.0, 22.0), bgr(62.0, 1.0, 22.0)
    painted, _ = painted_mouth(big, landmarks, lambda xs, ys: np.where((np.abs(ys - middle) <= 1)[:, None], shadow, teeth))
    out = apply_teeth(painted, landmarks, fake_ctx, whitening=0.0, yellow_reduction=0.0, evenness=1.0)
    lift = lab_of(out)[..., 0] - lab_of(painted)[..., 0]
    band = inner.copy()
    band[: middle - 1] = False
    band[middle + 2 :] = False
    assert band.sum() > 30
    assert lift[band].mean() < 0.2 * 14.0  # the blotch smoothing only; a gap would get 0.35 of its 14 L


def test_evenness_evens_the_color_between_teeth(dim, face_landmarks_720p, fake_ctx) -> None:
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    yellower, whiter = (74.0, 2.0, 30.0), (76.0, 0.0, 16.0)
    painted, _, teeth = striped_teeth(big, landmarks, (40.0, 2.0, 10.0), [yellower, whiter])
    out = apply_teeth(painted, landmarks, fake_ctx, whitening=0.0, yellow_reduction=0.0, evenness=1.0)
    before, after = lab_of(painted)[teeth], lab_of(out)[teeth]
    assert after[:, 2].std() < 0.75 * before[:, 2].std()
    assert abs(after[:, 2].mean() - before[:, 2].mean()) < 2.0  # evens out, does not bleach


# --------------------------------------------------------------------------
# Temporal behavior
# --------------------------------------------------------------------------


def test_reset_forgets_everything(dim, face_landmarks_720p, fake_ctx) -> None:
    """Measured on another scene than before the reset: whatever survives (the
    rate-limited presence, the skin's running mean) would show in the output."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    tongue, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(55.0, 19.0, 13.0), (len(xs), 1)))
    other_light = np.clip(tongue * np.float32([1.15, 0.95, 0.8]), 0.0, 1.0)
    effect = TeethEffect()
    for _ in range(6):
        apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    assert effect._presence == pytest.approx(1.0, abs=1e-6)
    effect.reset()
    assert vars(effect) == vars(TeethEffect())
    fresh = TeethEffect()
    assert np.array_equal(
        apply_teeth(other_light, landmarks, fake_ctx, effect=effect, **STRONG),
        apply_teeth(other_light, landmarks, fake_ctx, effect=fresh, **STRONG),
    )
    assert effect.stats == fresh.stats


def test_a_change_of_light_moves_the_skin_reference_gradually(dim, face_landmarks_720p, fake_ctx) -> None:
    effect = TeethEffect()
    apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)
    before = effect.stats.skin
    darker = dim * np.float32(0.7)
    apply_teeth(darker, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)
    after = effect.stats.skin
    target = TeethEffect()
    apply_teeth(darker, face_landmarks_720p, fake_ctx, effect=target, **STRONG)
    moved = (before[0] - after[0]) / (before[0] - target.stats.skin[0])
    assert 0.1 < moved < 0.5
    for _ in range(25):
        apply_teeth(darker, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)
    assert effect.stats.skin[0] == pytest.approx(target.stats.skin[0], abs=0.1)


def test_state_settles_and_reset_clears_it(dim, face_landmarks_720p, fake_ctx) -> None:
    effect = TeethEffect()
    outputs = [apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, **STRONG) for _ in range(6)]
    assert effect.stats is not None
    assert np.abs(outputs[-1] - outputs[-2]).max() < 1e-4  # a steady scene gives a steady result
    effect.reset()
    assert effect.stats is None
    fresh = apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)
    assert np.abs(fresh - outputs[0]).max() < 1e-6  # same as a brand-new effect


def test_camera_noise_does_not_make_the_teeth_flicker(dim, face_landmarks_720p, fake_ctx) -> None:
    rng = np.random.default_rng(7)
    teeth = teeth_mask(dim, face_landmarks_720p)
    effect = TeethEffect()
    lifts = []
    for _ in range(12):
        noisy = np.clip(dim + rng.normal(0.0, 0.01, dim.shape).astype(np.float32), 0.0, 1.0)
        out = apply_teeth(noisy, face_landmarks_720p, fake_ctx, effect=effect)
        lifts.append(float((lab_of(out)[..., 0] - lab_of(noisy)[..., 0])[teeth].mean()))
    assert min(lifts[2:]) > 0.5
    assert np.std(lifts[2:]) < 0.15 * np.mean(lifts[2:])


def test_a_stray_frame_does_not_switch_the_effect_off(dim, face_landmarks_720p, fake_ctx) -> None:
    """A frame where the mouth shows only something red (a hand, the tongue)
    barely moves the references, and the presence moves at most one step."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    red, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(50.0, 30.0, 18.0), (len(xs), 1)))
    teeth = teeth_mask(big, landmarks)
    effect = TeethEffect()
    for _ in range(8):
        steady = apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    references = effect.stats
    apply_teeth(red, landmarks, fake_ctx, effect=effect, **STRONG)
    assert abs(effect.stats.teeth_a - references.teeth_a) < 2.0
    assert effect._presence >= 1.0 - teeth_module._PRESENCE_STEP - 1e-6
    after = apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    assert change(big, after)[teeth].mean() > 0.9 * change(big, steady)[teeth].mean()


def test_teeth_appearing_fade_in_over_a_few_frames(dim, face_landmarks_720p, fake_ctx) -> None:
    """Tongue-only frames first (no teeth), then teeth: the presence ramps up
    in bounded steps instead of popping on."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    tongue, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(55.0, 19.0, 13.0), (len(xs), 1)))
    effect = TeethEffect()
    for _ in range(3):
        apply_teeth(tongue, landmarks, fake_ctx, effect=effect, **STRONG)
    assert effect._presence == pytest.approx(0.0, abs=1e-6)
    levels = []
    for _ in range(8):
        apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
        levels.append(effect._presence)
    steps = np.diff([0.0, *levels])
    assert steps.max() <= teeth_module._PRESENCE_STEP + 1e-6
    assert levels[-1] == pytest.approx(1.0, abs=1e-6)


def test_references_measured_on_the_tongue_are_replaced_at_once(dim, face_landmarks_720p, fake_ctx) -> None:
    """Thresholds from a tongue-only (or lips-only) start must not stay wrong
    while the teeth fade in: that would let gums pass as teeth."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    tongue, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(55.0, 19.0, 13.0), (len(xs), 1)))
    teeth, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(76.0, 1.0, 24.0), (len(xs), 1)))
    settled = TeethEffect()
    apply_teeth(teeth, landmarks, fake_ctx, effect=settled, **STRONG)
    effect = TeethEffect()
    for _ in range(6):
        apply_teeth(tongue, landmarks, fake_ctx, effect=effect, **STRONG)
    assert effect.stats.teeth_a > settled.stats.teeth_a + 10.0
    apply_teeth(teeth, landmarks, fake_ctx, effect=effect, **STRONG)
    assert abs(effect.stats.teeth_a - settled.stats.teeth_a) < 4.0
    apply_teeth(teeth, landmarks, fake_ctx, effect=effect, **STRONG)
    assert abs(effect.stats.teeth_a - settled.stats.teeth_a) < 1.5
    assert abs(effect.stats.teeth_l - settled.stats.teeth_l) < 2.0


def test_statistics_stay_clear_of_a_flat_lip_line(dim, face_landmarks_720p, fake_ctx) -> None:
    """The interior is a true inset: a radial one barely moves the flat edges,
    so a thin mouth would be measured on its lip line."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    scale = regions.face_scale(landmarks)
    work = teeth_module._Work(big, landmarks, full_roi(big), scale, False)
    interior = teeth_module._interior(work)
    distance_to_lip = cv2.distanceTransform(
        (regions.polygon_mask(landmarks[regions.LIPS_INNER], full_roi(big)) > 0.5).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    assert interior.sum() > 100
    assert distance_to_lip[interior].min() > teeth_module._STATS_INSET * scale


def test_effect_fades_in_with_the_mouth_opening(dim, face_landmarks_720p, fake_ctx) -> None:
    """Partly opened lips: smaller mouth openings get a weaker effect."""
    teeth = teeth_mask(dim, face_landmarks_720p)
    closed = closed_mouth(face_landmarks_720p)
    strengths = []
    effect = TeethEffect()
    apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)  # measure the teeth once
    for openness in (0.1, 0.25, 1.0):
        landmarks = closed + openness * (face_landmarks_720p - closed)
        out = apply_teeth(dim, landmarks.astype(np.float32), fake_ctx, effect=effect, **STRONG)
        strengths.append(float(change(dim, out)[teeth].mean()))
    assert strengths == sorted(strengths)
    assert strengths[0] < 0.6 * strengths[-1]


def test_effect_fades_in_gradually_above_the_slit(dim, face_landmarks_720p, fake_ctx, monkeypatch) -> None:
    """The opening's own ramp, apart from the shrinking area: a slit gets a small
    share of the change a wide-open gate would give at the very same pixels."""
    closed = closed_mouth(face_landmarks_720p)
    scale = regions.face_scale(face_landmarks_720p)
    slit = (closed + 0.02 * scale / teeth_module._opening(face_landmarks_720p) * (face_landmarks_720p - closed)).astype(np.float32)
    assert teeth_module._opening(slit) / scale == pytest.approx(0.02, abs=1e-3)
    effect = TeethEffect()
    apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)
    state = copy.deepcopy(effect)
    ramped = change(dim, apply_teeth(dim, slit, fake_ctx, effect=effect, **STRONG))
    monkeypatch.setattr(teeth_module, "_OPEN_RAMP", (0.0121, 0.0122))
    gate_open = change(dim, apply_teeth(dim, slit, fake_ctx, effect=state, **STRONG))
    assert gate_open.max() > 0.005
    assert 0.03 < ramped.max() / gate_open.max() < 0.3


# --------------------------------------------------------------------------
# Something skin-colored in front of the mouth
# --------------------------------------------------------------------------


def skin_of(image: np.ndarray, landmarks: np.ndarray) -> tuple[float, float, float]:
    measured = teeth_module._skin_reference(image, landmarks, regions.face_scale(landmarks))
    assert measured is not None
    return measured


def hand_over_mouth(image: np.ndarray, landmarks: np.ndarray, color: np.ndarray, cover: float = 1.0) -> np.ndarray:
    """A skin-colored ellipse rising from below over the mouth; at ``cover`` 1 it hides
    the opening and the lips around it. The tracker still reports an open mouth."""
    scale = regions.face_scale(landmarks)
    center = landmarks[regions.LIPS_OUTER].mean(axis=0)
    y = center[1] + (1.0 - cover) * 1.5 * scale
    mask = np.zeros(image.shape[:2], np.float32)
    cv2.ellipse(mask, (int(center[0]), int(y)), (int(0.95 * scale), int(0.75 * scale)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]
    return image * (1.0 - mask) + color * mask


def test_a_hand_in_front_of_the_mouth_is_left_alone(dim, face_landmarks_720p, fake_ctx) -> None:
    """The dim fixture's warm light is the hard case: its skin passes the hue test."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    skin = skin_of(big, landmarks)
    effect = TeethEffect()
    for _ in range(6):
        apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    references = effect.stats
    hand = hand_over_mouth(big, landmarks, bgr(*skin))
    for _ in range(3):
        assert np.array_equal(apply_teeth(hand, landmarks, fake_ctx, effect=effect, **STRONG), hand)
    # The references did not follow the hand, so the teeth come back as they were.
    assert abs(effect.stats.teeth_a - references.teeth_a) < 1e-6
    assert abs(effect.stats.teeth_b - references.teeth_b) < 1e-6
    assert abs(effect.stats.teeth_l - references.teeth_l) < 1e-6
    teeth = teeth_mask(big, landmarks)
    back = change(big, apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG))
    fresh = TeethEffect()
    for _ in range(6):
        settled = apply_teeth(big, landmarks, fake_ctx, effect=fresh, **STRONG)
    assert back[teeth].mean() > 0.8 * change(big, settled)[teeth].mean()


def test_a_textured_hand_is_recognized_by_its_color(dim, face_landmarks_720p, fake_ctx) -> None:
    """Shading and finger creases give a hand as much lightness structure as a
    mouth has; its chroma still is the skin's."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    skin = skin_of(big, landmarks)
    effect = TeethEffect()
    for _ in range(6):
        apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    hand = hand_over_mouth(big, landmarks, bgr(*skin))
    lab = lab_of(hand)
    rows = np.arange(hand.shape[0], dtype=np.float32)
    center_y = landmarks[regions.LIPS_INNER][:, 1].mean()
    creases = np.abs((rows - center_y) % 16.0 - 8.0) < 1.0
    lab[..., 0] += ((center_y - rows) * 0.25 - 18.0 * creases)[:, None]
    textured = blend.lab_to_bgr(np.clip(lab[..., 0], 0.0, 100.0)[..., None].repeat(3, axis=2) * np.float32([1, 0, 0]) + lab * np.float32([0, 1, 1]))
    inner = opening_mask(hand, landmarks, inset_px=3.0)
    low, high = np.percentile(lab_of(textured)[..., 0][inner], [10, 90])
    assert high - low > 8.0
    assert np.array_equal(apply_teeth(textured, landmarks, fake_ctx, effect=effect, **STRONG), textured)


def test_a_hand_arriving_stays_out_while_the_teeth_are_made_up(dim, face_landmarks_720p, fake_ctx) -> None:
    """Partly covered: the ring of skin samples is still mostly the face, and the
    teeth still are the brightest pixels. Only the teeth change."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    skin = skin_of(big, landmarks)
    effect = TeethEffect()
    for _ in range(6):
        apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    center_x = float(landmarks[regions.LIPS_INNER][:, 0].mean())
    color = bgr(0.85 * effect.stats.teeth_l, skin[1], skin[2])  # bright enough to pass as a tooth, but not the brightest
    half, _ = painted_mouth(big, landmarks, lambda xs, ys: np.where((xs > center_x)[:, None], color, big[ys, xs]))
    diff = change(half, apply_teeth(half, landmarks, fake_ctx, effect=effect, **STRONG))
    inner = opening_mask(half, landmarks, inset_px=3.0)
    xx = np.arange(half.shape[1])[None, :]
    covered = inner & (xx > center_x + 6)
    teeth = teeth_mask(big, landmarks) & (xx < center_x - 2)
    assert covered.sum() > 100 and teeth.sum() > 100
    assert diff[teeth].mean() > 0.01
    assert diff[covered].mean() < 0.15 * diff[teeth].mean()


def test_a_hand_of_another_color_than_the_face_stays_out_at_once(dim, face_landmarks_720p, fake_ctx) -> None:
    """A hand is not the cheek's color. Where it covers most of the lit skin
    samples, this frame's skin color knows it before the smoothed one does."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    skin = skin_of(big, landmarks)
    effect = TeethEffect()
    for _ in range(6):
        apply_teeth(big, landmarks, fake_ctx, effect=effect, **STRONG)
    center_x = float(landmarks[regions.LIPS_INNER][:, 0].mean())
    color = bgr(skin[0] + 4.0, skin[1] + 1.0, skin[2] - 7.0)
    hand = big.copy()
    hand[:, int(center_x) :] = color
    assert np.hypot(*np.subtract(skin_of(hand, landmarks)[1:], (skin[1] + 1.0, skin[2] - 7.0))) < 1.0
    diff = change(hand, apply_teeth(hand, landmarks, fake_ctx, effect=effect, **STRONG))
    inner = opening_mask(hand, landmarks, inset_px=3.0)
    xx = np.arange(hand.shape[1])[None, :]
    covered = inner & (xx > center_x + 6)
    teeth = teeth_mask(big, landmarks) & (xx < center_x - 2)
    assert covered.sum() > 100 and teeth.sum() > 100
    assert diff[teeth].mean() > 0.01
    assert diff[covered].mean() < 0.1 * diff[teeth].mean()


def test_teeth_of_the_same_lightness_as_the_skin_are_not_mistaken_for_it(dim, face_landmarks_720p, fake_ctx) -> None:
    """The cue is color only: ordinary teeth, however flat and however close to
    the skin's lightness, stay made up."""
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    skin = skin_of(big, landmarks)
    painted, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(skin[0], 1.0, 24.0), (len(xs), 1)))
    inner = opening_mask(big, landmarks, inset_px=3.0)
    out = apply_teeth(painted, landmarks, fake_ctx, **STRONG)
    assert change(painted, out)[inner].mean() > 0.02


# --------------------------------------------------------------------------
# Geometry: close-ups, borders, feather
# --------------------------------------------------------------------------


def test_close_up_faces_are_processed_at_reduced_resolution(dim, face_landmarks_720p, fake_ctx, monkeypatch) -> None:
    big, landmarks = zoomed(dim, face_landmarks_720p, 3.0)
    assert regions.face_scale(landmarks) > 1.1 * teeth_module._MAX_WORK_SCALE
    fast = apply_teeth(big, landmarks, fake_ctx)
    monkeypatch.setattr(teeth_module, "_MAX_WORK_SCALE", 1e9)
    exact = apply_teeth(big, landmarks, fake_ctx)
    teeth = teeth_mask(big, landmarks)
    assert fast.min() >= 0.0 and fast.max() <= 1.0
    assert change(big, fast)[teeth].mean() > 0.01
    assert np.abs(fast - exact).max(axis=2)[teeth].mean() < 0.006
    # Both paths stop at the same distance, far inside the old 0.1 x scale margin.
    outside = distance_to_opening(big, landmarks) > support_px(landmarks)
    assert change(big, fast)[outside].max() == 0.0
    assert change(big, exact)[outside].max() == 0.0


def test_mouth_at_and_beyond_the_frame_border(dim, face_landmarks_720p, fake_ctx) -> None:
    mouth_bottom = face_landmarks_720p[regions.LIPS_INNER][:, 1].max()
    partly = face_landmarks_720p + np.float32([0.0, dim.shape[0] - mouth_bottom + 3.0])
    out = apply_teeth(dim, partly, fake_ctx, **STRONG)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert change(dim, out)[: dim.shape[0] - 60].max() == 0.0
    gone = face_landmarks_720p + np.float32([0.0, 2000.0])
    assert np.array_equal(apply_teeth(dim, gone, fake_ctx, **STRONG), dim)
    broken = face_landmarks_720p.copy()
    broken[regions.LIPS_INNER] = np.nan
    assert np.array_equal(apply_teeth(dim, broken, fake_ctx, **STRONG), dim)


@pytest.mark.parametrize("corner", [78, 308])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e9, -1e9], ids=["nan", "inf", "far", "far-negative"])
def test_a_broken_lip_corner_skips_the_frame_without_rasterizing(dim, face_landmarks_720p, fake_ctx, monkeypatch, corner, value) -> None:
    """The corners are not part of the opening measure, so they reach this far.
    Rasterizing such a polygon stalled for up to 2 s per frame."""
    assert corner in regions.LIPS_INNER
    effect = TeethEffect()
    apply_teeth(dim, face_landmarks_720p, fake_ctx, effect=effect, **STRONG)
    references = effect.stats
    broken = face_landmarks_720p.copy()
    broken[corner] = value

    def fail(*args, **kwargs):
        raise AssertionError("image work on broken landmarks")

    with monkeypatch.context() as patched:
        patched.setattr(teeth_module.cv2, "fillPoly", fail)
        patched.setattr(teeth_module.blend, "bgr_to_lab", fail)
        assert np.array_equal(apply_teeth(dim, broken, fake_ctx, effect=effect, **STRONG), dim)
    assert effect.stats == references


def test_real_mouths_are_believable() -> None:
    """A wide-open mouth (the largest measured on real faces is about 1.15 face
    scales across) is nowhere near the limit, and a face filling the frame neither."""
    polygon = np.float32([[100, 100], [100 + 1.2 * 70, 100], [100 + 1.2 * 70, 100 + 0.8 * 70], [100, 100 + 0.8 * 70]])
    assert teeth_module._is_believable_mouth(polygon, 70.0, (720, 1280, 3))
    assert not teeth_module._is_believable_mouth(polygon * 20.0, 70.0, (720, 1280, 3))
    assert not teeth_module._is_believable_mouth(polygon, 1e6, (720, 1280, 3))


def test_feather_softens_the_edge(dim, face_landmarks_720p, fake_ctx) -> None:
    big, landmarks = zoomed(dim, face_landmarks_720p, 2.0)
    teeth, _ = painted_mouth(big, landmarks, lambda xs, ys: np.tile(bgr(72.0, 1.0, 24.0), (len(xs), 1)))
    sharp = change(teeth, apply_teeth(teeth, landmarks, fake_ctx, whitening=1.0, yellow_reduction=1.0, evenness=0.0, feather=0.0))
    soft = change(teeth, apply_teeth(teeth, landmarks, fake_ctx, whitening=1.0, yellow_reduction=1.0, evenness=0.0, feather=1.0))
    inner = opening_mask(teeth, landmarks, inset_px=0.0)
    edge = cv2.dilate(inner.astype(np.uint8), np.ones((3, 3), np.uint8)) > cv2.erode(inner.astype(np.uint8), np.ones((5, 5), np.uint8))
    core = opening_mask(teeth, landmarks, inset_px=8.0)
    assert core.sum() > 50
    # Same strength in the middle, a softer transition at the lip line.
    assert soft[core].mean() > 0.6 * sharp[core].mean()
    assert soft[edge].mean() / soft[core].mean() < sharp[edge].mean() / sharp[core].mean()


# --------------------------------------------------------------------------
# In the pipeline
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
    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        return SegMasks()

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_runs_in_the_pipeline_and_resets_when_the_face_is_lost(face_frame_720p, face_landmarks_720p) -> None:
    dim8 = np.clip(face_frame_720p * np.float32([0.62, 0.8, 0.86]), 0, 255).astype(np.uint8)
    settings = MakeupSettings()
    effect = TeethEffect()
    script = [face_landmarks_720p] * 8 + [None] * 10
    pipeline = MakeupPipeline(settings, tracker=_Tracker(script), segmenter=_Segmenter(), effects=[effect])
    try:
        results = [pipeline.process(dim8, timestamp_ms=1000 + 33 * i) for i in range(18)]
    finally:
        pipeline.close()
    teeth = teeth_mask(dim8.astype(np.float32) / 255, face_landmarks_720p)
    assert all(r.face_found for r in results[:8])
    assert "teeth" in results[7].timings_ms
    made_up = np.abs(results[7].output.astype(np.int16) - dim8).max(axis=2)
    assert made_up[teeth].mean() > 1.0
    outside = np.ones(teeth.shape, bool)
    outside[regions.roi_around(face_landmarks_720p[regions.LIPS_OUTER], dim8.shape).slices] = False
    assert made_up[outside].max() == 0
    assert np.array_equal(results[-1].output, dim8)  # faded out, then passed through
    assert effect.stats is None  # face lost -> reset

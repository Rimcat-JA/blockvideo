"""TeethEffect on the astronaut portrait (real landmarks): no-ops, locality,
whitening / yellow reduction / evenness behavior, what is never touched
(lips, tongue, an open mouth without teeth), temporal state, close-ups,
frame borders and a run through the pipeline."""

from __future__ import annotations

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
    scale = regions.face_scale(face_landmarks_720p)
    reach = teeth_module._FEATHER_REACH * teeth_module._feather(TeethSettings(**STRONG)) * scale + 2.0
    grown = regions.expand_polygon(face_landmarks_720p[regions.LIPS_INNER], reach)
    outside = regions.polygon_mask(grown, full_roi(dim)) == 0.0
    assert diff[outside].max() == 0.0
    assert np.array_equal(out[:50, :50], dim[:50, :50])  # background corner
    # The lips (away from the teeth edge) keep their color.
    lips = regions.lips_mask(face_landmarks_720p, full_roi(dim), outline=-0.15) > 0.99
    lips &= regions.polygon_mask(regions.expand_polygon(face_landmarks_720p[regions.LIPS_INNER], 3.0), full_roi(dim)) == 0
    assert lips.sum() > 200
    assert diff[lips].max() < 0.01
    assert diff[teeth_mask(dim, face_landmarks_720p)].mean() > 0.02


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
    grown = regions.expand_polygon(landmarks[regions.LIPS_INNER], 0.1 * regions.face_scale(landmarks))
    outside = regions.polygon_mask(grown, full_roi(big)) == 0.0
    assert change(big, fast)[outside].max() == 0.0


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

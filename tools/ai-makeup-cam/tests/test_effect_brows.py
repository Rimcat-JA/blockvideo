"""BrowEffect on the astronaut portrait (real landmarks) and on synthetic brows:
no-ops, locality, pigment that never lightens and keeps the hair texture,
hair-aware fill that stays on the real brow when the landmarks drift, the
outline's thickness / softness / tapered tail, robustness and a run through
the real pipeline."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from makeup_cam import blend, regions
from makeup_cam.config import BrowSettings, MakeupSettings
from makeup_cam.effects.base import registered_effects
from makeup_cam.effects.brows import BrowEffect

INVISIBLE = 1.0 / 255.0  # below one 8-bit output step
SKIN = 0.78
HAIR = 0.3
STRONG = {"opacity": 1.0, "thickness": 0.5, "softness": 1.0}


@pytest.fixture
def frame(face_frame_720p: np.ndarray) -> np.ndarray:
    return face_frame_720p.astype(np.float32) / np.float32(255.0)


@pytest.fixture
def skin_frame() -> np.ndarray:
    """A flat skin-colored frame without any brow hair."""
    return np.full((720, 1280, 3), SKIN, np.float32)


def apply_brows(
    frame: np.ndarray, landmarks: np.ndarray | None, fake_ctx, intensity: float = 1.0, **values
) -> np.ndarray:
    ctx = fake_ctx(frame, landmarks, intensity=intensity)
    return BrowEffect().apply(frame.copy(), ctx, BrowSettings(**values))


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def full_roi(frame: np.ndarray) -> regions.Roi:
    return regions.Roi(0, 0, frame.shape[1], frame.shape[0])


def band(landmarks: np.ndarray, side: regions.Side, start: float, end: float, roi: regions.Roi) -> np.ndarray:
    """Mask of the strip of the landmark brow between two fractions of its height (0 = upper edge)."""
    s = regions.side_indices(side)
    upper, lower = landmarks[s.brow_upper], landmarks[s.brow_lower]
    top, bottom = upper + start * (lower - upper), upper + end * (lower - upper)
    return regions.polygon_mask(np.vstack([top, bottom[::-1]]), roi)


def with_hairs(frame: np.ndarray, landmarks: np.ndarray, start: float, end: float) -> np.ndarray:
    """``frame`` with dark hair strokes (every other column) in a strip of both brows."""
    hairy = frame.copy()
    stripes = (np.arange(frame.shape[1]) % 2 == 0)[None, :]
    for side in regions.SIDES:
        strip = (band(landmarks, side, start, end, full_roi(frame)) > 0.5) & stripes
        hairy[strip] = HAIR
    return hairy


def test_registered_under_its_settings_section() -> None:
    assert BrowEffect in registered_effects()
    assert (BrowEffect.name, BrowEffect.order, BrowEffect.needs_face, BrowEffect.needs_hair_mask) == (
        "brows",
        30,
        True,
        False,
    )
    assert isinstance(MakeupSettings().brows, BrowSettings)


@pytest.mark.parametrize(
    ("intensity", "values", "use_landmarks"),
    [(1.0, {**STRONG, "opacity": 0.0}, True), (0.0, STRONG, True), (1.0, STRONG, False)],
    ids=["opacity-0", "intensity-0", "no-face"],
)
def test_exact_noop(frame, face_landmarks_720p, fake_ctx, intensity, values, use_landmarks) -> None:
    landmarks = face_landmarks_720p if use_landmarks else None
    out = apply_brows(frame, landmarks, fake_ctx, intensity=intensity, **values)
    assert out.dtype == np.float32
    assert np.array_equal(out, frame)


@pytest.mark.parametrize("thickness", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("softness", [0.0, 1.0])
@pytest.mark.parametrize("color", ["#000000", "#FFFFFF", "#5C4334"])
def test_every_setting_gives_a_valid_frame(frame, face_landmarks_720p, fake_ctx, thickness, softness, color) -> None:
    values = {"opacity": 1.0, "thickness": thickness, "softness": softness, "color": color}
    out = apply_brows(frame, face_landmarks_720p, fake_ctx, **values)
    assert out.dtype == np.float32 and out.shape == frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_settings_cover_the_whole_slider_ranges(frame, face_landmarks_720p, fake_ctx) -> None:
    for f in fields(BrowSettings):
        if f.metadata.get("kind") == "float":
            for value in (f.metadata["min"], f.metadata["max"]):
                values = {"opacity": 1.0, f.name: value}
                out = apply_brows(frame, face_landmarks_720p, fake_ctx, **values)
                assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_defaults_are_visible_but_subtle(frame, face_landmarks_720p, fake_ctx) -> None:
    diff = change(frame, apply_brows(frame, face_landmarks_720p, fake_ctx))
    assert 6 * INVISIBLE < diff.max() < 0.2


def test_strength_grows_with_intensity_and_opacity(frame, face_landmarks_720p, fake_ctx) -> None:
    def strength(intensity: float, opacity: float) -> float:
        out = apply_brows(frame, face_landmarks_720p, fake_ctx, intensity, opacity=opacity)
        return float(change(frame, out).sum())

    by_intensity = [strength(i, 0.8) for i in (0.25, 0.5, 1.0)]
    by_opacity = [strength(1.0, o) for o in (0.2, 0.5, 1.0)]
    for series in (by_intensity, by_opacity):
        assert series == sorted(series) and len(set(series)) == 3 and series[0] > 0


def test_changes_stay_on_the_brows(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_brows(frame, face_landmarks_720p, fake_ctx, **STRONG)
    diff = change(frame, out)
    brows = regions.roi_around(face_landmarks_720p[regions.LEFT_BROW + regions.RIGHT_BROW], frame.shape, pad=0.3)
    outside = np.ones(frame.shape[:2], bool)
    outside[brows.slices] = False
    assert np.array_equal(out[outside], frame[outside])
    for side in regions.SIDES:
        eye = regions.eye_opening_mask(face_landmarks_720p, side, full_roi(frame)) > 0
        assert diff[eye].max() < INVISIBLE
    # The flat gray margins of the frame are background: bit-identical.
    assert np.array_equal(out[:, :280], frame[:, :280]) and np.array_equal(out[:, 1000:], frame[:, 1000:])


def test_brow_pigment_never_lightens(frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_brows(frame, face_landmarks_720p, fake_ctx, **STRONG, color="#E0C0A0")
    assert (out <= frame).all()


def test_hairs_stay_darker_than_the_skin_between_them(frame, face_landmarks_720p, fake_ctx) -> None:
    """Even a full-strength near-black fill keeps the brow's texture: no flat paint, no inversion."""
    out = apply_brows(frame, face_landmarks_720p, fake_ctx, opacity=1.0, thickness=0.5, softness=0.0, color="#1E140F")
    before, after = blend.luminance(frame), blend.luminance(out)
    for side in regions.SIDES:
        core = regions.polygon_mask(face_landmarks_720p[regions.side_indices(side).brow], full_roi(frame)) >= 1.0
        luma = before[core]
        hairs, skin = luma <= np.percentile(luma, 15), luma >= np.percentile(luma, 70)
        assert after[core][hairs].mean() < after[core][skin].mean()
        assert np.corrcoef(luma, after[core])[0, 1] > 0.8


def test_fill_goes_where_the_hairs_are(skin_frame, face_landmarks_720p, fake_ctx) -> None:
    """Hairs only in the upper half of the outline: the skin between them darkens
    more than the bare skin in the lower half."""
    hairy = with_hairs(skin_frame, face_landmarks_720p, 0.0, 0.5)
    diff = change(hairy, apply_brows(hairy, face_landmarks_720p, fake_ctx, opacity=1.0))
    skin = hairy[..., 0] == SKIN
    for side in regions.SIDES:
        upper = (band(face_landmarks_720p, side, 0.1, 0.35, full_roi(hairy)) >= 1.0) & skin
        lower = band(face_landmarks_720p, side, 0.65, 0.9, full_roi(hairy)) >= 1.0
        assert diff[upper].mean() > 1.5 * diff[lower].mean() > 0


def test_fill_stays_on_the_brow_when_the_landmarks_drift(skin_frame, face_landmarks_720p, fake_ctx) -> None:
    """Landmarks raised by half the brow height (e.g. lagging behind a brow raise):
    inside the drifted outline, the color lands mostly on the real brow."""
    hairy = with_hairs(skin_frame, face_landmarks_720p, 0.0, 1.0)
    drifted = face_landmarks_720p.copy()
    for side in regions.SIDES:
        s = regions.side_indices(side)
        height = float(np.mean(np.linalg.norm(drifted[s.brow_upper] - drifted[s.brow_lower], axis=1)))
        drifted[s.brow] -= np.array([0.0, height / 2], np.float32)
    diff = change(hairy, apply_brows(hairy, drifted, fake_ctx, opacity=1.0))
    skin = hairy[..., 0] == SKIN
    for side in regions.SIDES:
        outline = band(drifted, side, 0.0, 1.0, full_roi(hairy)) >= 1.0
        real_brow = band(face_landmarks_720p, side, 0.0, 1.0, full_roi(hairy)) >= 1.0
        assert diff[outline & real_brow & skin].mean() > 1.4 * diff[outline & ~real_brow].mean() > 0


def test_thickness_and_softness_shape_the_outline(skin_frame, face_landmarks_720p, fake_ctx) -> None:
    def painted(**values) -> int:
        out = apply_brows(skin_frame, face_landmarks_720p, fake_ctx, opacity=1.0, **values)
        return int((change(skin_frame, out) > 0.01).sum())

    thin, normal, thick = (painted(thickness=t, softness=0.0) for t in (-0.5, 0.0, 0.5))
    assert thin < normal < thick
    assert painted(softness=1.0) > 1.2 * painted(softness=0.0)


def test_brow_tapers_toward_the_tail(skin_frame, face_landmarks_720p, fake_ctx) -> None:
    out = apply_brows(skin_frame, face_landmarks_720p, fake_ctx, opacity=1.0, softness=0.0)
    painted = change(skin_frame, out) > 0.02
    for side in regions.SIDES:
        upper = face_landmarks_720p[regions.side_indices(side).brow_upper]
        middle = int(upper[2, 0])
        near_tail = int(upper[3, 0] + 0.6 * (upper[4, 0] - upper[3, 0]))
        assert painted[:, middle].sum() > 1.3 * painted[:, near_tail].sum() > 0


def test_segmentation_masks_are_neither_needed_nor_used(frame, face_landmarks_720p, fake_ctx) -> None:
    without = apply_brows(frame, face_landmarks_720p, fake_ctx)
    half = np.full(frame.shape[:2], 0.5, np.float32)
    masks = {name: half for name in ("hair_mask", "face_skin_mask", "body_skin_mask")}
    with_masks = BrowEffect().apply(frame.copy(), fake_ctx(frame, face_landmarks_720p, **masks), BrowSettings())
    # Equal up to float rounding: OpenCV's SIMD kernels may round the last bit
    # differently depending on buffer alignment (far below one 8-bit step).
    np.testing.assert_allclose(with_masks, without, rtol=0, atol=1e-6)


def test_read_only_landmarks_are_left_untouched(frame, face_landmarks_720p, fake_ctx) -> None:
    landmarks = face_landmarks_720p.copy()
    landmarks.setflags(write=False)  # the pipeline shares its landmarks read-only
    apply_brows(frame, landmarks, fake_ctx, **STRONG)
    assert np.array_equal(landmarks, face_landmarks_720p)


@pytest.mark.parametrize("dx", [-600.0, -560.0, 760.0, 2000.0], ids=["one-brow-out", "edge", "other-edge", "gone"])
def test_brows_at_and_beyond_the_frame_border(frame, face_landmarks_720p, fake_ctx, dx) -> None:
    moved = face_landmarks_720p + np.array([dx, 0.0], np.float32)
    out = apply_brows(frame, moved, fake_ctx, **STRONG)
    assert out.shape == frame.shape and np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    if dx == 2000.0:
        assert np.array_equal(out, frame)


def test_tiny_and_degenerate_faces_give_valid_frames(frame, face_landmarks_720p, fake_ctx) -> None:
    center = face_landmarks_720p.mean(axis=0)
    tiny = (face_landmarks_720p - center) * np.float32(0.03) + center
    collapsed = np.repeat(center[None], len(face_landmarks_720p), axis=0)
    for landmarks in (tiny, collapsed):
        out = apply_brows(frame, landmarks, fake_ctx, **STRONG)
        assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
        assert change(frame, out).max() < 0.2


def test_real_pipeline_fills_the_brows(face_frame_720p, face_landmarks_720p, models_dir) -> None:
    from makeup_cam.pipeline import MakeupPipeline

    pipeline = MakeupPipeline(MakeupSettings(), model_dir=models_dir, effects=[BrowEffect()])
    try:
        for ts in range(0, 300, 33):  # past the fade-in
            result = pipeline.process(face_frame_720p, ts)
    finally:
        pipeline.close()
    assert result.face_found and "brows" in result.timings_ms
    diff = np.abs(result.output.astype(np.int16) - face_frame_720p.astype(np.int16)).max(axis=2)
    brows = regions.roi_around(face_landmarks_720p[regions.LEFT_BROW + regions.RIGHT_BROW], face_frame_720p.shape, 0.4)
    assert diff[brows.slices].max() >= 6
    outside = np.ones(diff.shape, bool)
    outside[brows.slices] = False
    assert diff[outside].max() == 0

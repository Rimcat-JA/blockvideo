"""Tests for makeup_cam.regions: ROI math, rasterization accuracy, landmark
index sets (checked on real landmarks of the astronaut portrait) and the
region mask helpers."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam import regions as R

TESTS_DIR = Path(__file__).resolve().parent


def assert_mask(mask: np.ndarray, roi: R.Roi) -> None:
    assert mask.dtype == np.float32
    assert mask.shape == (roi.h, roi.w)
    assert mask.flags["C_CONTIGUOUS"]
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def square(x0: float, y0: float, size: float) -> np.ndarray:
    return np.array([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)], np.float32)


def inside(points: np.ndarray, point: np.ndarray) -> bool:
    contour = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) > 0


def value_at(mask: np.ndarray, roi: R.Roi, point: np.ndarray) -> float:
    x, y = int(point[0]) - roi.x0, int(point[1]) - roi.y0
    return float(mask[y, x])


# --------------------------------------------------------------------------
# Roi
# --------------------------------------------------------------------------


def test_roi_properties() -> None:
    roi = R.Roi(10, 20, 50, 80)
    frame = np.zeros((100, 100), np.uint8)
    assert (roi.w, roi.h, roi.offset) == (40, 60, (10, 20))
    assert frame[roi.slices].shape == (60, 40)


def test_roi_around_pads_by_the_larger_side_and_clips() -> None:
    pts = np.array([[100, 100], [140, 120]], np.float32)  # 40 x 20 box
    assert R.roi_around(pts, (720, 1280), pad=0.25) == R.Roi(90, 90, 150, 130)
    assert R.roi_around(pts, (720, 1280), pad=0.0, margin_px=5) == R.Roi(95, 95, 145, 125)
    assert R.roi_around(pts, (110, 120, 3), pad=1.0) == R.Roi(60, 60, 120, 110)


def test_roi_around_returns_none_for_empty_input() -> None:
    assert R.roi_around(np.zeros((0, 2)), (720, 1280)) is None
    assert R.roi_around(np.full((3, 2), np.nan), (720, 1280)) is None
    assert R.roi_around(np.array([[-50.0, -50.0], [-20.0, -10.0]]), (720, 1280), pad=0.0) is None


# --------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------


def test_polygon_mask_coverage_is_accurate_and_antialiased() -> None:
    roi = R.Roi(0, 0, 40, 40)
    aligned = R.polygon_mask(square(10, 10, 10), roi)
    assert_mask(aligned, roi)
    assert aligned.sum() == pytest.approx(100, rel=0.06)
    assert aligned[15, 15] == 1.0 and aligned[15, 5] == 0.0
    shifted = R.polygon_mask(square(10.5, 10.5, 10), roi)
    assert shifted[15, 10] == pytest.approx(0.5, abs=0.15)  # half-covered edge pixel
    triangle = np.array([(5, 5), (35, 8), (12, 33)], np.float32)
    assert R.polygon_mask(triangle, roi).sum() == pytest.approx(409.5, rel=0.04)


def test_polygon_mask_is_independent_of_roi_position() -> None:
    triangle = np.array([(25.3, 21.7), (61.2, 30.1), (33.9, 58.4)], np.float32)
    a = R.polygon_mask(triangle, R.Roi(20, 20, 70, 70))
    b = R.polygon_mask(triangle, R.Roi(0, 0, 100, 100))[20:70, 20:70]
    np.testing.assert_array_equal(a, b)


def test_degenerate_polygon_vanishes() -> None:
    roi = R.Roi(0, 0, 40, 40)
    sliver = np.array([(5, 20), (35, 20), (5, 20.0001)], np.float32)
    assert not R.polygon_mask(sliver, roi).any()
    thin = np.array([(5, 20), (35, 20), (35, 21), (5, 21)], np.float32)
    assert R.polygon_mask(thin, roi).sum() == pytest.approx(30, rel=0.02)


def test_feathered_masks_match_an_exact_gaussian_blur() -> None:
    roi = R.Roi(0, 0, 240, 240)
    shapes = [square(80, 80, 40), np.array([(60.3, 70.6), (170.2, 90.1), (95.7, 180.4)], np.float32)]
    for points in shapes:
        hard = R.polygon_mask(points, roi)
        for sigma in (1.5, 4.0, 6.0, 12.0, 20.0):  # >= 6 is blurred at reduced resolution
            soft = R.polygon_mask(points, roi, feather_px=sigma)
            assert_mask(soft, roi)
            exact = cv2.GaussianBlur(hard, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
            assert np.abs(soft - exact).max() < 0.025
            assert soft.sum() == pytest.approx(exact.sum(), rel=0.03)
    soft_square = R.polygon_mask(shapes[0], roi, feather_px=1.5)
    assert (soft_square[100, 79] + soft_square[100, 80]) / 2 == pytest.approx(0.5, abs=0.02)  # edge x=80 at 50%


def test_feather_function() -> None:
    mask = np.zeros((90, 130), np.float32)
    mask[30:60, 40:90] = 1.0
    assert R.feather(mask, 0.0) is mask
    for sigma in (2.0, 9.0):  # full-resolution and downscaled paths
        exact = cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        out = R.feather(mask, sigma)
        assert out.dtype == np.float32 and out.shape == mask.shape
        assert np.abs(out - exact).max() < 0.03


def test_polyline_mask_width_and_taper() -> None:
    roi = R.Roi(0, 0, 40, 40)
    line = np.array([(5, 20), (35, 20)], np.float32)
    for width in (0.5, 1.0, 3.0):
        expected = 30 * width + math.pi * width**2 / 4  # body + round caps
        assert R.polyline_mask(line, width, roi).sum() == pytest.approx(expected, rel=0.05)
    tapered = R.polyline_mask(line, 4.0, roi, thickness_profile=[1.0, 0.2])
    assert tapered[:, 8].sum() > 2 * tapered[:, 32].sum()
    assert_mask(R.polyline_mask(line, 2.0, roi, feather_px=1.0), roi)
    assert not R.polyline_mask(line, 0.0, roi).any()
    with pytest.raises(ValueError):
        R.polyline_mask(line, 2.0, roi, thickness_profile=[1.0])


def test_ellipse_mask_area_and_rotation() -> None:
    roi = R.Roi(0, 0, 60, 60)
    flat = R.ellipse_mask((30, 30), (20, 8), 0, roi)
    assert_mask(flat, roi)
    assert flat.sum() == pytest.approx(math.pi * 20 * 8, rel=0.03)
    upright = R.ellipse_mask((30, 30), (20, 8), 90, roi)
    np.testing.assert_allclose(upright, flat.T, atol=0.07)  # cv2 approximates ellipses by polygons
    assert not R.ellipse_mask((30, 30), (0, 8), 0, roi).any()


def test_expand_polygon_moves_vertices_radially() -> None:
    pts = square(-5, -5, 10)  # centroid at the origin
    grown = R.expand_polygon(pts, 2.0)
    np.testing.assert_allclose(np.linalg.norm(grown, axis=1), np.linalg.norm(pts, axis=1) + 2.0, rtol=1e-5)
    collapsed = R.expand_polygon(pts, -100.0)
    np.testing.assert_allclose(collapsed, 0.0, atol=1e-5)
    unchanged = R.expand_polygon(pts, 0.0)
    np.testing.assert_array_equal(unchanged, pts)
    assert unchanged is not pts


def test_resample_polyline_is_even_and_keeps_ends() -> None:
    pts = np.array([(0, 0), (10, 0), (10, 30)], np.float32)
    out = R.resample_polyline(pts, 5)
    np.testing.assert_allclose(out[[0, -1]], pts[[0, -1]])
    np.testing.assert_allclose(np.linalg.norm(np.diff(out, axis=0), axis=1), 10.0, rtol=1e-5)


# --------------------------------------------------------------------------
# Landmark index sets on the real astronaut landmarks
# --------------------------------------------------------------------------

ALL_LISTS = {name: value for name, value in vars(R).items() if name.isupper() and isinstance(value, list)}


def test_index_lists_are_valid() -> None:
    for name, indices in ALL_LISTS.items():
        assert all(0 <= i < 478 for i in indices), name
        assert len(set(indices)) == len(indices), name
    assert 468 <= R.LEFT_IRIS_CENTER < 478 and 468 <= R.RIGHT_IRIS_CENTER < 478
    with pytest.raises(ValueError):
        R.side_indices("center")  # type: ignore[arg-type]


def test_subject_left_is_on_the_image_right(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p
    left, right = R.side_indices("left"), R.side_indices("right")
    for field in ("eye", "brow", "iris", "cheek_center", "nose_wing", "under_eye_top"):
        assert lm[getattr(left, field), 0].mean() > lm[getattr(right, field), 0].mean(), field
    assert lm[R.MOUTH_LEFT_CORNER, 0] > lm[R.MOUTH_RIGHT_CORNER, 0]
    for side in R.SIDES:
        s = R.side_indices(side)
        assert inside(lm[s.eye], lm[s.iris_center])
        assert inside(lm[s.iris], lm[s.iris_center])


def test_eye_lines_run_from_inner_to_outer_corner(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p
    nose_x = lm[R.NOSE_BRIDGE, 0].mean()
    for side in R.SIDES:
        s = R.side_indices(side)
        for line in (s.eye_upper, s.eye_lower, s.eye_crease, s.brow_upper, s.brow_lower, s.under_eye_top, s.under_eye_bottom):
            assert abs(lm[line[0], 0] - nose_x) < abs(lm[line[-1], 0] - nose_x), line
        assert s.eye_upper[0] == s.eye_lower[0] and s.eye_upper[-1] == s.eye_lower[-1]


def test_vertical_order_of_eye_features(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p

    def mean_y(indices: list[int]) -> float:
        return float(lm[indices, 1].mean())

    for side in R.SIDES:
        s = R.side_indices(side)
        assert mean_y(s.brow_upper) < mean_y(s.brow_lower) < mean_y(s.eye_crease) < mean_y(s.eye_upper[1:-1])
        assert mean_y(s.eye_upper[1:-1]) < mean_y(s.eye_lower[1:-1]) < mean_y(s.under_eye_top) < mean_y(s.under_eye_bottom)
        assert mean_y(s.under_eye_bottom) < mean_y(s.cheek_center) < lm[R.LIPS_TOP, 1]
    assert lm[R.NOSE_BRIDGE[0], 1] < lm[R.NOSE_TIP, 1] < lm[R.NOSE_BOTTOM, 1] < lm[R.LIPS_TOP, 1]
    assert lm[R.LIPS_TOP, 1] < lm[R.LIPS_INNER_TOP, 1] < lm[R.LIPS_INNER_BOTTOM, 1] < lm[R.LIPS_BOTTOM, 1]


def test_left_and_right_sets_mirror_each_other(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p
    midline = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 13, 14, 17, 18, 200, 199, 175, 152]
    slope, intercept = np.polyfit(lm[midline, 1], lm[midline, 0], 1)  # x = slope * y + intercept
    normal = np.array([1.0, -slope]) / math.hypot(1.0, slope)
    scale = R.face_scale(lm)
    left, right = R.side_indices("left"), R.side_indices("right")
    for field in ("eye_upper", "eye_lower", "eye_crease", "brow_upper", "brow_lower", "under_eye_top",
                  "under_eye_bottom", "cheek_center", "nose_wing"):
        pts = lm[getattr(left, field)].astype(np.float64)
        dist = (pts @ normal - intercept * normal[0])[:, None]
        mirrored = pts - 2 * dist * normal
        error = np.linalg.norm(mirrored - lm[getattr(right, field)], axis=1) / scale
        assert error.max() < 0.06, field


def test_lip_contours_nest(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p
    for i in R.LIPS_INNER:
        if i not in (78, 308):  # inner corners touch the outer line
            assert inside(lm[R.LIPS_OUTER], lm[i])
    assert R.LIPS_OUTER[0] == R.MOUTH_RIGHT_CORNER and R.MOUTH_LEFT_CORNER in R.LIPS_OUTER


# --------------------------------------------------------------------------
# face_scale
# --------------------------------------------------------------------------


def test_face_scale_is_the_inter_ocular_distance(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p
    iod = float(np.linalg.norm(R.eye_center(lm, "left") - R.eye_center(lm, "right")))
    assert R.face_scale(lm) == pytest.approx(iod, rel=1e-5)
    assert 50 < iod < 70  # the portrait's face is ~135 px wide


def test_face_scale_follows_size_rotation_and_turning(face_landmarks_720p: np.ndarray) -> None:
    lm = face_landmarks_720p.astype(np.float64)
    base = R.face_scale(lm)
    center = lm.mean(axis=0)
    assert R.face_scale((lm - center) * 2 + center) == pytest.approx(2 * base, rel=1e-5)
    angle = math.radians(25)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    assert R.face_scale((lm - center) @ rot.T + center) == pytest.approx(base, rel=1e-4)
    turned = lm.copy()
    turned[:, 0] = (lm[:, 0] - center[0]) * 0.6 + center[0]  # head turned ~53 degrees
    assert R.face_scale(turned) > 0.85 * base


# --------------------------------------------------------------------------
# Region masks on the real landmarks
# --------------------------------------------------------------------------


@pytest.fixture
def face_roi(face_landmarks_720p: np.ndarray) -> R.Roi:
    roi = R.roi_around(face_landmarks_720p, (720, 1280), pad=0.1)
    assert roi is not None
    return roi


def test_lips_mask_excludes_the_open_mouth(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    lips = R.lips_mask(lm, face_roi, feather_px=1.0)
    assert_mask(lips, face_roi)
    opening_center = lm[R.LIPS_INNER].mean(axis=0)
    lower_lip = (lm[R.LIPS_INNER_BOTTOM] + lm[R.LIPS_BOTTOM]) / 2
    assert value_at(lips, face_roi, opening_center) == 0.0
    assert value_at(lips, face_roi, lower_lip) > 0.95
    assert value_at(lips, face_roi, lm[R.NOSE_TIP]) == 0.0


def test_lips_mask_closed_mouth_keeps_the_whole_lip(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p.copy()
    lm[R.LIPS_LOWER_INNER] = lm[R.LIPS_UPPER_INNER]  # close the mouth
    lips = R.lips_mask(lm, face_roi)
    outer = R.polygon_mask(lm[R.LIPS_OUTER], face_roi)
    assert lips.sum() > 0.97 * outer.sum()


def test_lips_outline_grows_and_shrinks(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    areas = [R.lips_mask(lm, face_roi, outline=o).sum() for o in (-0.2, 0.0, 0.2)]
    assert areas[0] < areas[1] < areas[2]


def test_eye_opening_mask(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    for side in R.SIDES:
        eye = R.eye_opening_mask(lm, side, face_roi)
        assert_mask(eye, face_roi)
        assert value_at(eye, face_roi, lm[R.side_indices(side).iris_center]) == 1.0
        assert R.eye_opening_mask(lm, side, face_roi, expand_px=2.0).sum() > eye.sum()


def test_eyelid_mask_covers_the_lid_not_the_eye(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    for side in R.SIDES:
        s = R.side_indices(side)
        lid = R.eyelid_mask(lm, side, face_roi, height=0.6, feather_px=1.0)
        assert_mask(lid, face_roi)
        assert value_at(lid, face_roi, lm[s.iris_center]) == 0.0
        crease = lm[s.eye_crease[len(s.eye_crease) // 2]]
        assert value_at(lid, face_roi, crease) > 0.9
        assert value_at(R.eyelid_mask(lm, side, face_roi, height=0.3), face_roi, lm[s.brow_lower[2]]) == 0.0
        areas = [R.eyelid_mask(lm, side, face_roi, height=h).sum() for h in (0.0, 0.3, 0.6, 1.0)]
        assert areas[0] == 0.0 and areas[1] < areas[2] < areas[3]


def test_under_eye_mask(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    for side in R.SIDES:
        s = R.side_indices(side)
        area = R.under_eye_mask(lm, side, face_roi, depth=1.0, feather_px=1.0)
        assert_mask(area, face_roi)
        middle = (lm[s.under_eye_top[4]] + lm[s.under_eye_bottom[4]]) / 2
        assert value_at(area, face_roi, middle) > 0.95
        assert value_at(area, face_roi, lm[s.iris_center]) == 0.0
        assert R.under_eye_mask(lm, side, face_roi, depth=0.5).sum() < R.under_eye_mask(lm, side, face_roi).sum()
        assert not R.under_eye_mask(lm, side, face_roi, depth=0.0).any()


def test_brow_mask_thickness(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    for side in R.SIDES:
        normal = R.brow_mask(lm, side, face_roi)
        assert_mask(normal, face_roi)
        thick = R.brow_mask(lm, side, face_roi, thickness=0.5).sum()
        thin = R.brow_mask(lm, side, face_roi, thickness=-0.5).sum()
        assert thick == pytest.approx(1.5 * normal.sum(), rel=0.08)
        assert thin == pytest.approx(0.5 * normal.sum(), rel=0.12)


def test_face_oval_mask(face_landmarks_720p: np.ndarray, face_roi: R.Roi) -> None:
    lm = face_landmarks_720p
    oval = R.face_oval_mask(lm, face_roi, feather_px=4.0)
    assert_mask(oval, face_roi)
    for i in (R.NOSE_TIP, R.LIPS_INNER_TOP, R.LEFT_IRIS_CENTER, R.RIGHT_IRIS_CENTER):
        assert value_at(oval, face_roi, lm[i]) > 0.99
    assert R.face_oval_mask(lm, face_roi, expand_px=5).sum() > R.face_oval_mask(lm, face_roi).sum()


# --------------------------------------------------------------------------
# Shared fixtures, and the committed landmarks against the real model
# --------------------------------------------------------------------------


def test_shared_fixtures(face_frame_720p: np.ndarray, face_landmarks_720p: np.ndarray, fake_ctx) -> None:
    frame, lm = face_frame_720p, face_landmarks_720p
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
    assert (frame[:, :280] == (70, 75, 80)).all()
    assert lm.shape == (478, 2) and lm.dtype == np.float32
    assert 280 < lm[:, 0].min() and lm[:, 0].max() < 1000  # on the pasted portrait
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for side in R.SIDES:  # the landmarks belong to this frame: dark iris, bright cheek
        iris = lm[R.side_indices(side).iris_center]
        cheek = R.cheek_center(lm, side)
        assert gray[int(iris[1]), int(iris[0])] + 60 < gray[int(cheek[1]), int(cheek[0])]
    ctx = fake_ctx(frame, lm, intensity=0.5)
    assert (ctx.height, ctx.width, ctx.intensity, ctx.has_face) == (720, 1280, 0.5, True)
    assert not fake_ctx(frame, None).has_face


def test_committed_landmarks_match_the_model(models_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("mediapipe")
    pytest.importorskip("skimage")
    model = models_dir / "face_landmarker.task"
    if not model.is_file():
        pytest.skip("face_landmarker.task is not available")
    out = tmp_path / "landmarks.npy"
    script = TESTS_DIR / "data" / "generate_face_landmarks.py"
    subprocess.run(
        [sys.executable, str(script), "--model", str(model), "--out", str(out)],
        check=True, capture_output=True, timeout=300,
    )
    committed = np.load(TESTS_DIR / "data" / "face_landmarks_720p.npy")
    np.testing.assert_allclose(np.load(out), committed, atol=0.5)

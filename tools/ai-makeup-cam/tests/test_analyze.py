"""Plan A (makeup_cam.analyze): measuring makeup in a photo and building a preset.

The first half needs no models: pure helpers, synthetic color problems, and the
whole analysis with the precomputed landmarks of the astronaut portrait and
fake segmentation. The second half runs round trips with the real models:
a known preset is rendered by the real pipeline, analyzed, and the recovered
settings are compared. Those skip without ``AI_MAKEUP_CAM_MODELS_DIR``, and the
CREMA-D ones also without ``.test-media`` (see ``scripts/fetch_test_media.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam import analyze, blend, presets, regions
from makeup_cam.analyze import AnalysisError, AnalysisResult, analyze_makeup, delta_e00, save_as_preset, trimmed_median
from makeup_cam.config import BUILTIN_PRESETS_DIR, MakeupSettings, hex_to_bgr
from makeup_cam.segmentation import BACKGROUND, BODY_SKIN, CLOTHES, FACE_SKIN, HAIR, OTHERS

TEST_VIDEOS = Path(__file__).resolve().parents[1] / ".test-media" / "testvids"


def lab_of(value: str) -> np.ndarray:
    return blend.bgr_to_lab(np.array(hex_to_bgr(value), np.float32).reshape(1, 1, 3))[0, 0]


def color_distance(a: str, b: str) -> float:
    return float(delta_e00(lab_of(a), lab_of(b)))


def hue_of(value: str) -> float:
    lab = lab_of(value)
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


# --------------------------------------------------------------------------
# Pure helpers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lab1", "lab2", "expected"),
    [
        # Sharma, Wu & Dalal (2005) test data.
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ],
)
def test_delta_e00_matches_the_published_test_data(lab1, lab2, expected) -> None:
    assert float(delta_e00(lab1, lab2)) == pytest.approx(expected, abs=1e-3)


def test_trimmed_median_ignores_speculars_and_shadows() -> None:
    lips = np.tile([45.0, 40.0, 15.0], (80, 1))
    speculars = np.tile([95.0, 2.0, 2.0], (10, 1))
    shadows = np.tile([5.0, 3.0, 1.0], (10, 1))
    out = trimmed_median(np.vstack([lips, speculars, shadows]))
    assert out == pytest.approx([45.0, 40.0, 15.0])
    assert trimmed_median(np.zeros((0, 3))) is None


@pytest.mark.parametrize(("opacity", "color"), [(0.4, (0.55, 0.53, 0.91)), (0.8, (0.35, 0.38, 0.58))])
def test_canonical_tint_reproduces_the_change(opacity, color) -> None:
    from makeup_cam.effects import blush as blush_module

    skin = np.array([0.55, 0.65, 0.8])
    at_0, at_1 = analyze._response_line(blush_module._flush, skin)
    gain = opacity * (at_0 + np.array(color) * (at_1 - at_0))
    got_opacity, got_color = analyze.canonical_tint(gain, at_0, at_1, analyze._chroma_of, 43.0)
    rendered = got_opacity * (at_0 + got_color * (at_1 - at_0))
    assert rendered == pytest.approx(gain, abs=0.005)
    assert 0.0 < got_opacity <= 1.0


def test_canonical_tint_picks_the_color_nearest_the_goal() -> None:
    at_0, at_1 = -np.ones(3), np.zeros(3)  # plain multiply
    gain = 0.5 * (np.array([0.2, 0.26, 0.36]) - 1.0)  # #5C4334 at opacity 0.5
    opacity, color = analyze.canonical_tint(gain, at_0, at_1, analyze._lightness_of, 31.0)
    assert opacity == pytest.approx(0.5, abs=0.06)
    assert analyze._lightness_of(color) == pytest.approx(31.0, abs=3.0)


def test_canonical_tint_with_no_change_is_nearly_invisible() -> None:
    at_0, at_1 = -np.ones(3), np.zeros(3)
    opacity, color = analyze.canonical_tint(np.zeros(3), at_0, at_1, analyze._lightness_of, 31.0)
    assert opacity * np.abs(color - 1.0).max() < 1e-3


def test_natural_lip_color_follows_the_skin() -> None:
    light = analyze.natural_lip_lab((85.0, 8.0, 18.0))
    deep = analyze.natural_lip_lab((45.0, 15.0, 30.0))
    assert light[0] > deep[0]
    assert light[1] > 8.0, "light skin: lips clearly redder than the skin"
    assert deep[2] < 30.0, "lips are less yellow than the skin"


def test_lipstick_distance_counts_color_and_darkness_but_not_lighter_lips() -> None:
    natural = np.array([50.0, 20.0, 15.0])
    assert analyze.lipstick_distance(natural + (20.0, 0.0, 0.0), natural) == pytest.approx(0.0)
    assert analyze.lipstick_distance(natural + (0.0, 20.0, 0.0), natural) == pytest.approx(20.0)
    assert analyze.lipstick_distance(natural + (-16.0, 0.0, 0.0), natural) == pytest.approx(10.0)


def test_upsample_grid_interpolates_nodes_exactly() -> None:
    step = 5
    nodes_y, nodes_x = np.mgrid[0:7, 0:9] * step
    out = analyze._upsample_grid(2.0 * nodes_x + nodes_y.astype(np.float64), step, 37, 28)
    ys, xs = np.mgrid[0:28, 0:37]
    assert out == pytest.approx(2.0 * xs + ys, abs=1e-4)


def test_thin_plate_map_is_the_identity_and_follows_a_shift() -> None:
    pts = np.random.default_rng(3).uniform(10.0, 190.0, (40, 2)).astype(np.float32)
    map_x, map_y = analyze._tps_map(pts, pts, (200, 200), 4, 60.0)
    ys, xs = np.mgrid[0:200, 0:200]
    assert map_x == pytest.approx(xs, abs=1e-2)
    assert map_y == pytest.approx(ys, abs=1e-2)
    map_x, map_y = analyze._tps_map(pts, pts + (3.0, -2.0), (200, 200), 4, 60.0)
    assert map_x == pytest.approx(xs + 3.0, abs=1e-2)
    assert map_y == pytest.approx(ys - 2.0, abs=1e-2)


def test_limited_hue_only_moves_colors_outside_the_range() -> None:
    pink = np.array([65.0, 40.0, 14.0])
    assert analyze._limited_hue(pink, (0.0, 65.0)) == pytest.approx(pink)
    purple = np.array([65.0, 30.0, -30.0])
    moved = analyze._limited_hue(purple, (0.0, 65.0))
    assert moved[2] == pytest.approx(0.0, abs=1e-6)
    assert np.hypot(moved[1], moved[2]) == pytest.approx(np.hypot(30.0, 30.0))


# --------------------------------------------------------------------------
# White balance.
# --------------------------------------------------------------------------


def photo_with_room(color: tuple[float, float, float], noise: float = 0.0, seed: int = 0, share: float = 1.0) -> analyze._Photo:
    rng = np.random.default_rng(seed)
    room = np.full((60, 80, 3), color, np.float32) + rng.normal(0.0, noise, (60, 80, 3)).astype(np.float32)
    room_mask = np.zeros((60, 80), np.float32)
    room_mask[: int(60 * share)] = 1.0
    image = np.full((100, 100, 3), 0.6, np.float32)
    return analyze._Photo(
        image=image,
        landmarks=np.zeros((478, 2), np.float32),
        scale=50.0,
        source_scale=50.0,
        hair=np.zeros((100, 100), np.float32),
        face_skin=np.ones((100, 100), np.float32),
        body_skin=np.zeros((100, 100), np.float32),
        room=np.clip(room, 0.0, 1.0),
        room_mask=room_mask,
        yaw=0.0,
        eyes_open=0.3,
    )


def test_gray_world_partly_neutralizes_a_near_gray_room() -> None:
    warm = photo_with_room((0.5, 0.6, 0.66))
    gains, chroma = analyze.gray_world_gains(warm)
    assert chroma is not None and chroma < analyze._GRAY_WORLD_MAX_CHROMA
    corrected = np.array((0.5, 0.6, 0.66)) * gains
    assert corrected.max() - corrected.min() < 0.16 * 0.5
    assert gains[0] > 1.0 > gains[2]


def test_gray_world_leaves_a_colored_room_alone() -> None:
    gains, chroma = analyze.gray_world_gains(photo_with_room((0.2, 0.7, 0.2)))
    assert chroma > analyze._GRAY_WORLD_MAX_CHROMA
    assert gains == pytest.approx(1.0)


def test_gray_world_without_a_room_does_nothing() -> None:
    gains, chroma = analyze.gray_world_gains(photo_with_room((0.5, 0.6, 0.7), share=0.0))
    assert chroma is None
    assert gains == pytest.approx(1.0)


def test_balance_uses_the_room_when_it_is_the_same_room() -> None:
    makeup = photo_with_room((0.4, 0.5, 0.6), noise=0.005, seed=1)
    bare = photo_with_room((0.5, 0.5, 0.5), noise=0.005, seed=2)
    empty = np.zeros((100, 100), np.float32)
    balance = analyze.balance_gains(makeup, bare, empty, empty)
    assert balance.source == "room"
    assert balance.trust > 0.9
    assert balance.gains == pytest.approx((0.8, 1.0, 1.2), abs=0.02)


def test_balance_does_not_trust_another_room() -> None:
    rng = np.random.default_rng(5)
    makeup = photo_with_room((0.4, 0.5, 0.6))
    bare = photo_with_room((0.5, 0.5, 0.5))
    bare.room = rng.uniform(0.1, 0.9, bare.room.shape).astype(np.float32)  # posters, furniture
    stable = np.ones((100, 100), np.float32)
    balance = analyze.balance_gains(makeup, bare, stable, stable)
    assert balance.source == "forehead"
    assert balance.trust < analyze._MIN_TONE_TRUST


def test_balance_does_not_trust_an_even_wall_in_another_room() -> None:
    makeup = photo_with_room((0.3, 0.3, 0.3))
    bare = photo_with_room((0.7, 0.7, 0.7))  # a clean ratio, but the faces are equally bright
    stable = np.ones((100, 100), np.float32)
    balance = analyze.balance_gains(makeup, bare, stable, stable)
    assert balance.source == "forehead"
    assert balance.gains == pytest.approx(1.0)


def test_balance_falls_back_to_the_neck() -> None:
    makeup = photo_with_room((0.4, 0.5, 0.6), share=0.0)
    bare = photo_with_room((0.5, 0.5, 0.5), share=0.0)
    for photo, level in ((makeup, 0.6), (bare, 0.3)):
        photo.landmarks[analyze._CHIN_BOTTOM] = (50.0, 40.0)
        photo.body_skin[60:] = 1.0
        photo.image[60:] = level
    balance = analyze.balance_gains(makeup, bare, np.zeros((100, 100)), np.zeros((100, 100)))
    assert balance.source == "neck"
    assert balance.gains == pytest.approx(2.0, abs=0.01)


# --------------------------------------------------------------------------
# The whole analysis without models: real effects, known landmarks, fake masks.
# --------------------------------------------------------------------------


class FakeModels:
    """Everything is face skin inside the frame's center, room elsewhere."""

    def __init__(self, other_faces: list[np.ndarray] | None = None) -> None:
        self.other_faces = other_faces or []

    def video_landmarker(self):  # pragma: no cover - _stable_landmarks is replaced
        raise AssertionError("not used")

    def faces(self, rgb: np.ndarray) -> list[np.ndarray]:
        return self.other_faces

    def segment(self, rgb: np.ndarray) -> list[np.ndarray]:
        height, width = rgb.shape[:2]
        zeros = np.zeros((height, width), np.float32)
        center = np.zeros((height, width), np.float32)
        center[height // 8 : -height // 8, width // 8 : -width // 8] = 1.0
        masks = [zeros] * 6
        masks[BACKGROUND] = 1.0 - center
        masks[FACE_SKIN] = center
        masks[HAIR] = masks[BODY_SKIN] = masks[CLOTHES] = masks[OTHERS] = zeros
        return masks

    def close(self) -> None:
        pass


def effect_chain() -> list:
    effects = analyze._load_effects()
    return [effects[name] for name in analyze.MEASURED_SECTIONS]


def render_known(frame: np.ndarray, landmarks: np.ndarray, settings: MakeupSettings) -> np.ndarray:
    """``settings`` on ``frame`` with fixed landmarks and the fake masks."""
    from makeup_cam.effects.base import FrameContext

    height, width = frame.shape[:2]
    image = frame.astype(np.float32) * np.float32(1.0 / 255.0)
    for effect in effect_chain():
        ctx = FrameContext(height=height, width=width, landmarks=landmarks.astype(np.float32))
        image = effect.apply(image, ctx, getattr(settings, effect.name))
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def custom(**sections) -> MakeupSettings:
    settings = MakeupSettings()
    for section, values in sections.items():
        for key, value in values.items():
            setattr(getattr(settings, section), key, value)
    return settings


@pytest.fixture
def fixed_landmarks(monkeypatch, face_landmarks_720p):
    monkeypatch.setattr(analyze, "_stable_landmarks", lambda image, models: face_landmarks_720p.copy())
    return face_landmarks_720p


def everything_off() -> dict:
    return {
        "skin": {"smoothing": 0.0, "blemish": 0.0, "redness": 0.0, "under_eye": 0.0, "tone_strength": 0.0, "brightness": 0.0},
        "contour": {"shading": 0.0, "highlight": 0.0},
        "tearbag": {"strength": 0.0},
        "brows": {"opacity": 0.0},
        "eyeshadow": {"opacity": 0.0},
        "eyeliner": {"opacity": 0.0},
        "blush": {"opacity": 0.0},
        "lips": {"opacity": 0.0},
    }


def test_analysis_with_fixed_landmarks_recovers_each_section(face_frame_720p, fixed_landmarks) -> None:
    sections = everything_off()
    sections["lips"] = {"color": "#B3122A", "opacity": 0.9, "finish": "matte", "gloss": 0.0}
    sections["blush"] = {"color": "#E8878C", "opacity": 0.5, "shape": "oval"}
    sections["eyeshadow"] = {"color": "#946150", "opacity": 0.85, "height": 0.7}
    sections["eyeliner"] = {"color": "#1A1412", "opacity": 0.9, "thickness": 0.6, "wing": 0.7}
    sections["brows"] = {"color": "#5C4334", "opacity": 0.5}
    target = render_known(face_frame_720p, fixed_landmarks, custom(**sections))
    result = analyze_makeup(target, face_frame_720p, models=FakeModels())
    got = result.settings
    assert color_distance(got.lips.color, "#B3122A") < 8.0
    assert got.lips.opacity == pytest.approx(0.9, abs=0.15)
    assert got.lips.finish == "matte"
    assert got.blush.opacity > 0.1
    assert abs(hue_of(got.blush.color) - hue_of("#E8878C")) < 15.0
    assert got.eyeshadow.opacity > 0.3
    assert got.eyeliner.opacity > 0.4
    assert got.eyeliner.wing > 0.3
    assert got.brows.opacity > 0.2
    assert got.tearbag.strength == pytest.approx(0.0, abs=0.1)
    assert got.contour.shading < 0.15


def test_identical_photos_mean_no_makeup_at_all(face_frame_720p, fixed_landmarks) -> None:
    got = analyze_makeup(face_frame_720p, face_frame_720p, models=FakeModels()).settings
    for section in (got.lips, got.blush, got.eyeshadow, got.eyeliner, got.brows):
        assert section.opacity == 0.0
    assert got.tearbag.strength == 0.0
    assert got.contour.shading == 0.0 and got.contour.highlight == 0.0
    assert got.skin.smoothing == 0.0 and got.skin.tone_strength == 0.0 and got.skin.brightness == 0.0


def test_not_measured_sections_keep_the_base_values(face_frame_720p, fixed_landmarks) -> None:
    base = MakeupSettings()
    base.teeth.whitening = 0.77
    base.reshape.eye_size = 0.33
    base.hair.gloss = 0.66
    base.hairshape.flyaway = 0.11
    base.general.intensity = 0.9
    got = analyze_makeup(face_frame_720p, None, base=base, models=FakeModels()).settings
    assert got.teeth.whitening == 0.77
    assert got.reshape.eye_size == 0.33
    assert got.hair.gloss == 0.66
    assert got.hairshape.flyaway == 0.11
    assert base.general.intensity == 0.9, "the base preset is not modified"
    assert base.lips.color == MakeupSettings().lips.color, "the base preset is not modified"


def test_contour_and_tear_bag_need_a_bare_photo(face_frame_720p, fixed_landmarks) -> None:
    base = MakeupSettings()
    base.contour.shading = 0.41
    base.tearbag.strength = 0.27
    result = analyze_makeup(face_frame_720p, None, base=base, models=FakeModels())
    assert result.settings.contour.shading == 0.41
    assert result.settings.tearbag.strength == 0.27
    assert result.confidence["contour"] == 0.0
    assert result.confidence["tearbag"] == 0.0
    assert any("涙袋" in note for note in result.notes)


def test_result_shape_and_json(face_frame_720p, fixed_landmarks) -> None:
    result = analyze_makeup(face_frame_720p, None, models=FakeModels())
    assert isinstance(result, AnalysisResult)
    assert set(result.confidence) >= {"skin", "lips", "blush", "eyeshadow", "eyeliner", "brows"}
    assert all(0.0 <= value <= 1.0 for value in result.confidence.values())
    assert any("すっぴん写真がない" in note for note in result.notes)
    data = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert MakeupSettings.from_dict(data["settings"]).lips.color == result.settings.lips.color
    assert data["measurements"]["photos"]["source_scale"] > 0
    assert data["notes"] == result.notes


def test_every_recovered_value_is_inside_its_slider(face_frame_720p, fixed_landmarks) -> None:
    target = render_known(face_frame_720p, fixed_landmarks, custom(lips={"opacity": 1.0}, blush={"opacity": 1.0}))
    settings = analyze_makeup(target, face_frame_720p, models=FakeModels()).settings
    assert MakeupSettings.from_dict(settings.to_dict()) == settings


def test_a_hidden_section_keeps_the_base_values(monkeypatch, face_frame_720p, fixed_landmarks) -> None:
    base = MakeupSettings()
    base.lips.opacity = 0.61
    monkeypatch.setattr(analyze, "_lip_core", lambda scene: np.zeros(scene.target.shape[:2], np.float32))
    result = analyze_makeup(face_frame_720p, None, base=base, models=FakeModels())
    assert result.settings.lips.opacity == 0.61
    assert result.confidence["lips"] == 0.0
    assert any("唇" in note for note in result.notes)


# --------------------------------------------------------------------------
# Errors.
# --------------------------------------------------------------------------


def test_no_face_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(analyze, "_stable_landmarks", lambda image, models: None)
    with pytest.raises(AnalysisError, match="顔を見つけられません"):
        analyze_makeup(np.zeros((300, 300, 3), np.uint8), models=FakeModels())


def test_two_faces_are_a_clear_error(face_frame_720p, fixed_landmarks) -> None:
    other = fixed_landmarks.copy()
    other = other - other.mean(axis=0) + (250.0, 300.0)
    normalized = np.hstack([other / 720.0, np.zeros((478, 1))]).astype(np.float32)
    with pytest.raises(AnalysisError, match="複数の顔"):
        analyze_makeup(face_frame_720p, models=FakeModels([normalized]))


def test_a_small_second_face_is_ignored(face_frame_720p, fixed_landmarks) -> None:
    other = (fixed_landmarks - fixed_landmarks.mean(axis=0)) * 0.2 + (100.0, 100.0)
    normalized = np.hstack([other / 720.0, np.zeros((478, 1))]).astype(np.float32)
    assert analyze_makeup(face_frame_720p, models=FakeModels([normalized])).settings is not None


def test_a_tiny_face_is_a_clear_error(monkeypatch, face_landmarks_720p, face_frame_720p) -> None:
    center = face_landmarks_720p.mean(axis=0)
    small = (face_landmarks_720p - center) * 0.3 + center
    monkeypatch.setattr(analyze, "_stable_landmarks", lambda image, models: small)
    with pytest.raises(AnalysisError, match="小さすぎます"):
        analyze_makeup(face_frame_720p, models=FakeModels())


def test_a_profile_is_a_clear_error(monkeypatch, face_landmarks_720p, face_frame_720p) -> None:
    turned = face_landmarks_720p.copy()
    turned[regions.NOSE_TIP] = turned[regions.RIGHT_EYE_OUTER_CORNER] + (2.0, 5.0)
    monkeypatch.setattr(analyze, "_stable_landmarks", lambda image, models: turned)
    with pytest.raises(AnalysisError, match="横を向きすぎ"):
        analyze_makeup(face_frame_720p, models=FakeModels())


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), np.float32), np.zeros((0, 0, 3), np.uint8), np.zeros((5, 5, 2), np.uint8), "photo.jpg", None],
)
def test_bad_input_is_a_clear_error(bad) -> None:
    with pytest.raises(AnalysisError, match="メイクの写真"):
        analyze_makeup(bad, models=FakeModels())


def test_gray_and_alpha_photos_are_accepted(monkeypatch, face_frame_720p, fixed_landmarks) -> None:
    gray = cv2.cvtColor(face_frame_720p, cv2.COLOR_BGR2GRAY)
    assert analyze_makeup(gray, models=FakeModels()).settings is not None
    alpha = cv2.cvtColor(face_frame_720p, cv2.COLOR_BGR2BGRA)
    assert analyze_makeup(alpha, models=FakeModels()).settings is not None


# --------------------------------------------------------------------------
# Saving and the command line.
# --------------------------------------------------------------------------


@pytest.fixture
def user_home(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path))
    return tmp_path


def fake_result() -> AnalysisResult:
    settings = MakeupSettings()
    settings.lips.color = "#112233"
    return AnalysisResult(settings, {"photos": {"yaw": 0.1}}, {"lips": 0.5}, ["メモです"])


def test_save_as_preset_writes_a_user_preset(user_home) -> None:
    info = save_as_preset(fake_result(), "いつものメイク")
    assert info.name == "いつものメイク"
    assert info.path.parent == user_home / "presets"
    assert presets.load_preset("いつものメイク").lips.color == "#112233"


def write_photo(path: Path) -> None:
    ok, data = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    assert ok
    data.tofile(path)


def test_cli_saves_the_preset_and_the_json(monkeypatch, user_home, tmp_path, capsys) -> None:
    makeup = tmp_path / "メイク.png"
    bare = tmp_path / "すっぴん.png"
    write_photo(makeup)
    write_photo(bare)
    seen = {}

    def fake_analyze(makeup_bgr, bare_bgr=None, *, base=None, models_dir=None, models=None):
        seen.update(makeup=makeup_bgr, bare=bare_bgr, base=base)
        return fake_result()

    monkeypatch.setattr(analyze, "analyze_makeup", fake_analyze)
    out = tmp_path / "out" / "report.json"
    code = analyze.main([str(makeup), "--bare", str(bare), "--base", "polished", "--name", "テスト", "--json", str(out)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "『テスト』を保存しました" in captured
    assert "メモです" in captured
    assert "リップ" in captured
    assert seen["bare"] is not None
    assert seen["base"] == presets.load_preset("polished")
    assert presets.load_preset("テスト").lips.color == "#112233"
    assert json.loads(out.read_text(encoding="utf-8"))["confidence"] == {"lips": 0.5}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["ない.jpg", "--name", "x"], "読み込めませんでした"),
        (["photo.heic", "--name", "x"], "HEIC"),
    ],
)
def test_cli_reports_unreadable_photos_in_japanese(user_home, tmp_path, monkeypatch, capsys, args, message) -> None:
    monkeypatch.chdir(tmp_path)
    assert analyze.main(args) == 1
    assert message in capsys.readouterr().err


def test_cli_reports_a_missing_base_preset(user_home, tmp_path, capsys) -> None:
    photo = tmp_path / "a.png"
    write_photo(photo)
    assert analyze.main([str(photo), "--base", "そんなものはない", "--name", "x"]) == 1
    assert "見つかりません" in capsys.readouterr().err


def test_cli_reports_analysis_errors(monkeypatch, user_home, tmp_path, capsys) -> None:
    photo = tmp_path / "a.png"
    write_photo(photo)

    def failing(*args, **kwargs):
        raise AnalysisError("顔を見つけられませんでした")

    monkeypatch.setattr(analyze, "analyze_makeup", failing)
    assert analyze.main([str(photo), "--name", "x"]) == 1
    assert "顔を見つけられませんでした" in capsys.readouterr().err
    assert not presets.find_preset("x")


# --------------------------------------------------------------------------
# Round trips with the real models: preset -> real pipeline -> analyze -> compare.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_models():
    import os

    value = os.environ.get("AI_MAKEUP_CAM_MODELS_DIR")
    if not value or not Path(value).is_dir():
        pytest.skip("AI_MAKEUP_CAM_MODELS_DIR is not set to an existing directory")
    models = analyze.MediaPipeStillModels(value)
    yield models
    models.close()


@pytest.fixture(scope="module")
def models_path(real_models) -> Path:
    import os

    return Path(os.environ["AI_MAKEUP_CAM_MODELS_DIR"])


def builtin_preset(name: str) -> MakeupSettings:
    return MakeupSettings.load(BUILTIN_PRESETS_DIR / f"{name}.json")


def look(name: str) -> MakeupSettings:
    settings = builtin_preset("polished")
    changes = {
        "polished": {},
        "red_matte": {"lips": {"color": "#B3122A", "opacity": 0.9, "finish": "matte", "gloss": 0.0}},
        "coral_blush": {"blush": {"color": "#FF7F6A", "opacity": 0.45, "shape": "round", "size": 1.1}},
        "brown_smoky": {"eyeshadow": {"color": "#5A3A2E", "opacity": 0.9, "height": 0.8, "softness": 0.7}},
        "winged": {"eyeliner": {"color": "#141010", "opacity": 0.9, "thickness": 0.6, "wing": 0.7, "softness": 0.3}},
        "no_blush_no_liner": {"blush": {"opacity": 0.0}, "eyeliner": {"opacity": 0.0}},
    }[name]
    for section, values in changes.items():
        for key, value in values.items():
            setattr(getattr(settings, section), key, value)
    return settings


_RENDERS: dict[tuple[str, str], np.ndarray] = {}


def pipeline_render(frame: np.ndarray, settings: MakeupSettings, models_path: Path) -> np.ndarray:
    """The real pipeline with every effect the analysis fits (not all are registered yet)."""
    from makeup_cam.models import FACE_LANDMARKER, SELFIE_MULTICLASS, ensure_model
    from makeup_cam.pipeline import MakeupPipeline
    from makeup_cam.segmentation import Segmenter
    from makeup_cam.tracking import FaceTracker

    still = settings.copy()
    still.general.mirror_output = False
    tracker = FaceTracker(ensure_model(FACE_LANDMARKER, directory=models_path), still.general.landmark_smoothing)
    segmenter = Segmenter(ensure_model(SELFIE_MULTICLASS, directory=models_path), background=False)
    pipeline = MakeupPipeline(still, tracker=tracker, segmenter=segmenter, effects=effect_chain())
    try:
        for index in range(20):
            result = pipeline.process(frame, index * 33)
        return result.output.copy()
    finally:
        pipeline.close()


def target_of(key: str, frame: np.ndarray, name: str, models_path: Path) -> np.ndarray:
    if (key, name) not in _RENDERS:
        _RENDERS[(key, name)] = pipeline_render(frame, look(name), models_path)
    return _RENDERS[(key, name)]


def zone_errors(a: np.ndarray, b: np.ndarray, models) -> dict[str, float]:
    """Mean delta-E 2000 per facial zone between two uint8 frames (slightly blurred)."""
    landmarks = analyze._stable_landmarks(a, models)
    scale = regions.face_scale(landmarks)
    height, width = a.shape[:2]
    roi = regions.Roi(0, 0, width, height)
    lab_a = blend.bgr_to_lab(analyze._blur(a.astype(np.float32) / 255.0, 0.008 * scale))
    lab_b = blend.bgr_to_lab(analyze._blur(b.astype(np.float32) / 255.0, 0.008 * scale))
    delta = delta_e00(lab_a, lab_b)
    cheeks = np.zeros((height, width), np.float32)
    for side in regions.SIDES:
        cheeks = np.maximum(cheeks, regions.ellipse_mask(regions.cheek_center(landmarks, side), (0.3 * scale, 0.2 * scale), 0, roi))
    lid, _ = analyze._lid_zones(landmarks, roi, scale)
    zones = {
        "lips": regions.lips_mask(landmarks, roi, outline=-0.05) > 0.5,
        "cheeks": cheeks > 0.5,
        "lids": lid,
        "face": regions.face_oval_mask(landmarks, roi, expand_px=-0.03 * scale) > 0.5,
    }
    return {name: float(delta[zone].mean()) for name, zone in zones.items()}


# The real models are slow to run many times: the expensive results are cached per test module.
_ANALYSES: dict[tuple[str, str, bool], AnalysisResult] = {}


def analysis_of(key: str, frame: np.ndarray, name: str, models, models_path: Path, with_bare: bool) -> AnalysisResult:
    if (key, name, with_bare) not in _ANALYSES:
        target = target_of(key, frame, name, models_path)
        _ANALYSES[(key, name, with_bare)] = analyze_makeup(target, frame if with_bare else None, models=models)
    return _ANALYSES[(key, name, with_bare)]


@pytest.fixture(scope="module")
def astronaut() -> np.ndarray:
    pytest.importorskip("skimage")
    from conftest import build_face_frame_720p

    return build_face_frame_720p()


@pytest.mark.parametrize("name", ["polished", "red_matte", "coral_blush", "brown_smoky", "winged"])
def test_round_trip_with_the_bare_photo(astronaut, real_models, models_path, name) -> None:
    truth = look(name)
    result = analysis_of("astronaut", astronaut, name, real_models, models_path, True)
    got = result.settings
    assert color_distance(got.lips.color, truth.lips.color) < 8.0
    assert got.lips.opacity == pytest.approx(truth.lips.opacity, abs=0.15)
    assert got.lips.finish == truth.lips.finish
    assert got.blush.opacity > 0.1, "blush present"
    assert abs(hue_of(got.blush.color) - hue_of(truth.blush.color)) < 15.0
    assert got.eyeshadow.opacity >= 0.3, "eyeshadow present"
    assert got.eyeliner.opacity > 0.4, "eyeliner present"
    assert got.brows.opacity == pytest.approx(truth.brows.opacity, abs=0.2)
    if truth.eyeliner.wing >= 0.5:
        assert got.eyeliner.wing >= 0.4
    # What really matters: the same pipeline with the recovered preset looks the same.
    target = target_of("astronaut", astronaut, name, models_path)
    errors = zone_errors(target, pipeline_render(astronaut, got, models_path), real_models)
    assert errors["face"] < 1.5, errors
    assert errors["lips"] < 3.0, errors
    assert errors["cheeks"] < 2.0, errors
    assert errors["lids"] < 4.0, errors


def test_round_trip_without_the_bare_photo_stays_in_range(astronaut, real_models, models_path) -> None:
    truth = look("red_matte")
    got = analysis_of("astronaut", astronaut, "red_matte", real_models, models_path, False).settings
    assert color_distance(got.lips.color, truth.lips.color) < 12.0
    assert got.lips.opacity == pytest.approx(truth.lips.opacity, abs=0.3)
    # Contour and tear bags stay at the base preset's values.
    base = presets.load_preset(presets.DEFAULT_PRESET_NAME)
    assert got.contour.shading == base.contour.shading
    assert got.tearbag.strength == base.tearbag.strength


def test_no_blush_and_no_liner_are_found_absent(astronaut, real_models, models_path) -> None:
    got = analysis_of("astronaut", astronaut, "no_blush_no_liner", real_models, models_path, True).settings
    assert got.blush.opacity < 0.15
    assert got.eyeliner.opacity < 0.1


def test_the_same_photo_twice_is_no_makeup(astronaut, real_models) -> None:
    got = analyze_makeup(astronaut, astronaut, models=real_models).settings
    for section in (got.lips, got.blush, got.eyeshadow, got.eyeliner, got.brows):
        assert section.opacity == 0.0
    assert got.tearbag.strength == 0.0
    assert got.contour.shading == 0.0 and got.contour.highlight == 0.0


def test_color_cast_exposure_and_jpeg_do_not_change_the_colors(astronaut, real_models, models_path) -> None:
    truth = look("polished")
    target = target_of("astronaut", astronaut, "polished", models_path)

    def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    warm = np.clip(target.astype(np.float32) * (0.85, 1.0, 1.1), 0, 255).astype(np.uint8)
    dim = np.clip(target.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    for label, makeup, bare in (("cast", warm, astronaut), ("exposure", dim, astronaut), ("jpeg", jpeg(target, 70), jpeg(astronaut, 70))):
        got = analyze_makeup(makeup, bare, models=real_models).settings
        assert color_distance(got.lips.color, truth.lips.color) < 10.0, label
        assert got.lips.opacity == pytest.approx(truth.lips.opacity, abs=0.2), label
        assert abs(hue_of(got.blush.color) - hue_of(truth.blush.color)) < 25.0, label


def test_resolution_does_not_matter_much(astronaut, real_models, models_path) -> None:
    truth = look("polished")
    target = target_of("astronaut", astronaut, "polished", models_path)
    phone_target = cv2.resize(target, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    phone_bare = cv2.resize(astronaut, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    got = analyze_makeup(phone_target, phone_bare, models=real_models).settings
    assert color_distance(got.lips.color, truth.lips.color) < 8.0
    assert got.lips.opacity == pytest.approx(truth.lips.opacity, abs=0.2)


def test_real_failures_are_clear(astronaut, real_models) -> None:
    with pytest.raises(AnalysisError, match="顔を見つけられません"):
        analyze_makeup(np.full((480, 640, 3), 128, np.uint8), models=real_models)
    two = np.hstack([astronaut[:, 280:1000], astronaut[:, 280:1000]])
    with pytest.raises(AnalysisError, match="複数の顔"):
        analyze_makeup(two, models=real_models)


def test_cli_end_to_end(astronaut, real_models, models_path, user_home, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_MODELS_DIR", str(models_path))
    makeup = tmp_path / "makeup.jpg"
    ok, data = cv2.imencode(".jpg", target_of("astronaut", astronaut, "polished", models_path), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    data.tofile(makeup)
    bare = tmp_path / "bare.png"
    ok, data = cv2.imencode(".png", astronaut)
    assert ok
    data.tofile(bare)
    assert analyze.main([str(makeup), "--bare", str(bare), "--name", "いつものメイク"]) == 0
    assert "『いつものメイク』を保存しました" in capsys.readouterr().out
    assert color_distance(presets.load_preset("いつものメイク").lips.color, look("polished").lips.color) < 10.0


# --------------------------------------------------------------------------
# Women of different skin tones (CREMA-D).
# --------------------------------------------------------------------------


def crema_frame(actor: str, clip: str = "DFA_NEU_XX", index: int = 30) -> np.ndarray:
    path = TEST_VIDEOS / f"{actor}_{clip}.flv"
    if not path.is_file():
        pytest.skip("CREMA-D clips are not in .test-media")
    capture = cv2.VideoCapture(str(path))
    frame = None
    for _ in range(index + 1):
        ok, image = capture.read()
        if not ok:
            break
        frame = image
    capture.release()
    assert frame is not None
    upscaled = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(upscaled, 0, 0, 160, 160, cv2.BORDER_CONSTANT, value=(70, 75, 80))


@pytest.mark.parametrize("actor", ["1072", "1013", "1029"], ids=["asian", "hispanic", "african-american"])
def test_round_trip_on_real_talking_faces(real_models, models_path, actor) -> None:
    frame = crema_frame(actor)
    truth = look("polished")
    result = analysis_of(actor, frame, "polished", real_models, models_path, True)
    got = result.settings
    assert color_distance(got.lips.color, truth.lips.color) < 8.0
    assert got.lips.opacity == pytest.approx(truth.lips.opacity, abs=0.15)
    target = target_of(actor, frame, "polished", models_path)
    errors = zone_errors(target, pipeline_render(frame, got, models_path), real_models)
    assert errors["face"] < 1.5, errors
    assert errors["lips"] < 3.0, errors
    assert errors["lids"] < 3.0, errors
    without = analysis_of(actor, frame, "polished", real_models, models_path, False).settings
    assert color_distance(without.lips.color, truth.lips.color) < 12.0


@pytest.mark.parametrize("actor", ["1072", "1013", "1029", "1060", "1082", "1076"])
def test_a_bare_face_alone_gets_almost_no_makeup(real_models, actor) -> None:
    got = analyze_makeup(crema_frame(actor), None, models=real_models).settings
    assert got.lips.opacity == 0.0
    assert got.blush.opacity <= 0.3
    assert got.eyeshadow.opacity <= 0.25
    assert got.eyeliner.opacity == 0.0
    assert got.brows.opacity <= 0.1


def test_smiling_makeup_photo_with_a_neutral_bare_photo_gives_no_nonsense(real_models, models_path) -> None:
    neutral = crema_frame("1072")
    smiling = crema_frame("1072", "IEO_HAP_HI")
    truth = look("polished")
    target = pipeline_render(smiling, truth, models_path)
    result = analyze_makeup(target, neutral, models=real_models)
    got = result.settings
    assert color_distance(got.lips.color, truth.lips.color) < 12.0
    assert any("表情や顔の向き" in note for note in result.notes)
    assert result.confidence["contour"] == 0.0
    # Lighter than a real shadow or brow never comes out at full opacity.
    assert analyze._lightness_of(np.array(hex_to_bgr(got.eyeshadow.color))) < 80.0
    assert analyze._lightness_of(np.array(hex_to_bgr(got.brows.color))) < 80.0
    errors = zone_errors(target, pipeline_render(smiling, got, models_path), real_models)
    assert errors["face"] < 3.0, errors


# --------------------------------------------------------------------------
# Regressions: what two photos do not agree on must not become makeup.
# --------------------------------------------------------------------------


def test_a_tone_curve_is_told_from_a_gain() -> None:
    rng = np.random.default_rng(1)
    bare = rng.uniform(0.05, 0.9, (4000, 3)).astype(np.float32)
    plain, curved, power, spread, plain_spread = analyze._paired_curve(np.clip(bare**1.25 * 0.9, 0.0, 1.0), bare)
    assert power == pytest.approx(1.25, abs=0.02)
    assert curved == pytest.approx(0.9, abs=0.02)
    assert spread < 0.02 < plain_spread
    plain, curved, power, spread, plain_spread = analyze._paired_curve(bare * np.float32(0.8), bare)
    assert power == pytest.approx(1.0)
    assert plain == pytest.approx(0.8, abs=0.01)


def test_to_bare_light_undoes_a_curve_and_is_exact_without_one() -> None:
    bare = np.random.default_rng(2).uniform(0.05, 0.9, (40, 40, 3)).astype(np.float32)
    balance = analyze._Balance(np.array((0.9, 0.9, 0.9), np.float32), "person", 1.0, np.array((1.25, 1.25, 1.25)))
    assert analyze._to_bare_light(bare**1.25 * np.float32(0.9), balance) == pytest.approx(bare, abs=1e-3)
    same = analyze._Balance(np.ones(3, np.float32), "person", 1.0)
    assert analyze._to_bare_light(bare, same) is bare


def test_balance_prefers_the_person_over_a_wall_that_changed() -> None:
    def photo(wall: float) -> analyze._Photo:
        out = photo_with_room((wall, wall, wall))
        out.room_person = np.zeros((60, 80), np.float32)
        out.room_person[30:] = 1.0
        out.room[30:] = np.tile(np.linspace(0.1, 0.8, 80)[None, :, None], (30, 1, 3))
        out.room_mask = 1.0 - out.room_person
        return out

    stable = np.ones((100, 100), np.float32)
    balance = analyze.balance_gains(photo(0.4), photo(0.5), stable, stable)
    assert balance.source == "person"
    assert balance.gains == pytest.approx(1.0, abs=1e-3)
    assert balance.power is None


def test_a_large_exposure_difference_is_undone_not_clipped() -> None:
    makeup = photo_with_room((0.12, 0.12, 0.12))
    bare = photo_with_room((0.48, 0.48, 0.48))
    makeup.image[:] = 0.15
    stable = np.ones((100, 100), np.float32)
    assert analyze.balance_gains(makeup, bare, stable, stable).gains == pytest.approx(0.25, abs=0.01)


def test_bangs_in_the_makeup_photo_alone_count_as_hidden(monkeypatch, face_frame_720p, fixed_landmarks) -> None:
    measured = []
    original = analyze._inputs_with_bare

    def spy(target, bare, gray, measurements, notes):
        target.hair = target.hair.copy()
        target.hair[:] = 0.0
        y = int(target.landmarks[regions.RIGHT_BROW][:, 1].mean())
        target.hair[: y + 30] = 1.0
        out = original(target, bare, gray, measurements, notes)
        measured.append(out["hair"])
        return out

    monkeypatch.setattr(analyze, "_inputs_with_bare", spy)
    base = MakeupSettings()
    base.brows.color = "#123456"
    result = analyze_makeup(face_frame_720p, face_frame_720p, base=base, models=FakeModels())
    assert measured[0].max() == 1.0
    assert result.confidence["brows"] == 0.0
    assert result.settings.brows.color == "#123456"
    assert any("眉が前髪" in note for note in result.notes)


def test_brow_colors_stay_brown_or_gray() -> None:
    for gain in ((-0.2, -0.3, -0.5), (-0.1, -0.45, -0.1), (-0.6, -0.5, -0.4), (-0.05, -0.2, -0.45)):
        opacity, color = analyze._brow_color(np.array(gain))
        lab = analyze._lab_of_bgr(color)
        chroma = float(np.hypot(lab[1], lab[2]))
        assert 0.0 < opacity <= 1.0
        assert chroma <= analyze._BROW_MAX_CHROMA + 1.0
        if chroma > 6.0:
            hue = float(np.degrees(np.arctan2(lab[2], lab[1])))
            assert analyze._BROW_HUE_RANGE[0] - 2.0 <= hue <= analyze._BROW_HUE_RANGE[1] + 2.0, (gain, hue)


def test_lipstick_needs_distance_in_the_direction_of_a_lipstick() -> None:
    natural = np.array((60.0, 20.0, 25.0))
    assert not analyze._looks_like_lipstick(natural, natural)
    assert not analyze._looks_like_lipstick((80.0, 31.0, 32.0), natural), "a smile: lighter and warmer"
    assert analyze._looks_like_lipstick((52.0, 30.0, 24.0), natural), "rose, not far away"
    assert analyze._looks_like_lipstick((42.0, 47.0, 34.0), natural), "red"


def test_a_dark_patch_on_the_cheeks_is_not_a_black_blush(face_frame_720p, fixed_landmarks) -> None:
    dark = face_frame_720p.copy()
    roi = regions.Roi(0, 0, dark.shape[1], dark.shape[0])
    for side in regions.SIDES:
        patch = regions.ellipse_mask(regions.cheek_center(fixed_landmarks, side), (45.0, 35.0), 0.0, roi)[..., None]
        dark = (dark * (1.0 - 0.8 * patch)).astype(np.uint8)
    result = analyze_makeup(dark, face_frame_720p, models=FakeModels())
    got = result.settings.blush
    if got.opacity > 0.0:
        lab = lab_of(got.color)
        assert lab[0] >= analyze._BLUSH_MIN_LIGHTNESS and np.hypot(lab[1], lab[2]) >= analyze._BLUSH_MIN_CHROMA


def test_a_skin_shift_without_evening_is_not_foundation(face_frame_720p, fixed_landmarks) -> None:
    """The whole face is greener and the room is the same: light on the face, no foundation."""
    height, width = face_frame_720p.shape[:2]
    target = face_frame_720p.astype(np.float32) * np.float32(1.0 / 255.0)
    center = (slice(height // 8, -height // 8), slice(width // 8, -width // 8))
    target[center] = target[center] * np.float32((0.97, 1.03, 0.97))
    photo = (np.clip(target, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    base = MakeupSettings()
    base.skin.tone_color = "#ABCDEF"
    result = analyze_makeup(photo, face_frame_720p, base=base, models=FakeModels())
    assert result.settings.skin.tone_color == "#ABCDEF"
    assert result.settings.skin.tone_strength == base.skin.tone_strength


def test_identical_photos_in_doubtful_light_mean_no_makeup_either(monkeypatch, face_frame_720p, fixed_landmarks) -> None:
    """The base's foundation stays in the preset when the skin cannot be compared, but
    it must not be taken for a change of the lips and the rest."""
    monkeypatch.setattr(analyze, "_MIN_TONE_TRUST", 2.0)
    base = MakeupSettings()
    base.skin.tone_strength = 0.6
    base.skin.brightness = 0.1
    got = analyze_makeup(face_frame_720p, face_frame_720p, base=base, models=FakeModels()).settings
    assert got.skin.tone_strength == 0.6, "kept in the preset"
    for section in (got.lips, got.blush, got.eyeshadow, got.eyeliner, got.brows):
        assert section.opacity == 0.0
    assert got.contour.shading == 0.0 and got.contour.highlight == 0.0


@pytest.mark.parametrize("kind", ["gray", "negative"])
def test_a_photo_without_skin_colors_keeps_everything_at_base(face_frame_720p, fixed_landmarks, kind) -> None:
    photo = cv2.cvtColor(face_frame_720p, cv2.COLOR_BGR2GRAY) if kind == "gray" else 255 - face_frame_720p
    base = MakeupSettings()
    base.lips.opacity = 0.61
    result = analyze_makeup(photo, face_frame_720p, base=base, models=FakeModels())
    assert result.settings.lips.opacity == 0.61
    assert set(result.confidence) == set(analyze.MEASURED_SECTIONS)
    assert not any(result.confidence.values())
    assert any("色を測れません" in note for note in result.notes)


def test_the_overall_intensity_does_not_thin_the_result(face_frame_720p, fixed_landmarks) -> None:
    base = MakeupSettings()
    base.general.intensity = 0.5
    result = analyze_makeup(face_frame_720p, None, base=base, models=FakeModels())
    assert result.settings.general.intensity == 1.0
    assert base.general.intensity == 0.5
    assert any("100%" in note for note in result.notes)


def test_a_built_in_preset_is_not_replaced(user_home) -> None:
    with pytest.raises(presets.PresetError, match="標準プリセット"):
        save_as_preset(fake_result(), "natural")
    assert presets.find_preset("natural").builtin


def test_cli_saves_nothing_when_the_json_cannot_be_written(monkeypatch, user_home, tmp_path, capsys) -> None:
    photo = tmp_path / "a.png"
    write_photo(photo)
    monkeypatch.setattr(analyze, "analyze_makeup", lambda *args, **kwargs: fake_result())
    assert analyze.main([str(photo), "--name", "x", "--json", str(tmp_path)]) == 1
    assert "エラー" in capsys.readouterr().err
    assert not presets.find_preset("x")


def test_cli_usage_errors_and_help_are_in_japanese(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        analyze.main([])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "必須" in err and "error:" not in err and "usage:" not in err
    with pytest.raises(SystemExit):
        analyze.main(["--help"])
    out = capsys.readouterr().out
    assert "この説明を表示" in out and "show this help" not in out and "usage:" not in out


# --------------------------------------------------------------------------
# Regressions on real faces (CREMA-D).
# --------------------------------------------------------------------------


def assert_no_makeup_on(got: MakeupSettings) -> None:
    for name, section in (("lips", got.lips), ("eyeshadow", got.eyeshadow), ("eyeliner", got.eyeliner)):
        assert section.opacity == 0.0, name
    assert got.blush.opacity <= 0.1
    assert got.brows.opacity <= 0.15
    assert got.contour.shading <= 0.15 and got.contour.highlight <= 0.15


def test_two_nearby_frames_of_a_bare_face_are_no_makeup(real_models) -> None:
    """The mouth moves a little between them: the lips used to come out at 0.7."""
    got = analyze_makeup(crema_frame("1076", index=35), crema_frame("1076", index=38), models=real_models).settings
    assert got.lips.opacity == 0.0
    assert got.eyeshadow.opacity == 0.0
    assert got.eyeliner.opacity == 0.0


def test_a_tone_curve_on_a_made_up_face_is_not_a_strong_foundation(real_models, models_path) -> None:
    bare = crema_frame("1013")
    target = pipeline_render(bare, look("polished"), models_path)
    curved = np.clip(255.0 * (target.astype(np.float32) / 255.0) ** 0.8, 0, 255).astype(np.uint8)
    straight = analyze_makeup(target, bare, models=real_models).settings
    got = analyze_makeup(curved, bare, models=real_models).settings
    assert got.skin.tone_strength < 0.35, "the foundation of 'polished' is light"
    assert got.skin.tone_strength == pytest.approx(straight.skin.tone_strength, abs=0.1)
    assert got.lips.opacity > 0.4


@pytest.mark.parametrize("actor", ["1074", "1013"])
@pytest.mark.parametrize("power", [0.8, 1.25])
def test_another_tone_curve_is_not_makeup(real_models, actor, power) -> None:
    bare = crema_frame(actor)
    curved = np.clip(255.0 * (bare.astype(np.float32) / 255.0) ** power, 0, 255).astype(np.uint8)
    result = analyze_makeup(curved, bare, models=real_models)
    assert_no_makeup_on(result.settings)
    assert result.settings.skin.tone_strength <= 0.12


def test_bangs_in_the_makeup_photo_do_not_become_black_brows(real_models, models_path) -> None:
    bare = crema_frame("1029")
    target = pipeline_render(bare, look("polished"), models_path)
    landmarks = analyze._stable_landmarks(target, real_models)
    scale = regions.face_scale(landmarks)
    bottom = int(landmarks[regions.RIGHT_BROW + regions.LEFT_BROW][:, 1].max() + 0.05 * scale)
    top = int(landmarks[10][1] - 0.35 * scale)
    left, right = int(landmarks[234][0]), int(landmarks[454][0])
    strands = cv2.GaussianBlur(np.random.default_rng(1).normal(0.0, 30.0, (bottom - top, right - left, 1)).astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=20.0)
    banged = target.copy()
    banged[top:bottom, left:right] = np.clip(np.float32((28, 28, 32)) + strands[..., None], 0, 255).astype(np.uint8)
    got = analyze_makeup(banged, bare, models=real_models)
    assert got.confidence["brows"] == 0.0
    assert got.settings.brows.color == presets.load_preset(presets.DEFAULT_PRESET_NAME).brows.color


def test_dark_glasses_are_not_eye_makeup(real_models, models_path) -> None:
    bare = crema_frame("1072")
    target = pipeline_render(bare, look("polished"), models_path)
    landmarks = analyze._stable_landmarks(target, real_models)
    scale = regions.face_scale(landmarks)
    glasses = target.copy()
    for eye in (regions.RIGHT_EYE, regions.LEFT_EYE):
        center = landmarks[eye].mean(axis=0)
        cv2.ellipse(glasses, (int(center[0]), int(center[1])), (int(0.42 * scale), int(0.3 * scale)), 0, 0, 360, (15, 15, 15), -1, cv2.LINE_AA)
    for bare_photo in (bare, None):
        result = analyze_makeup(glasses, bare_photo, models=real_models)
        for section in ("eyeshadow", "eyeliner", "brows"):
            assert result.confidence[section] == 0.0, section
        assert any("メガネ" in note for note in result.notes)


@pytest.mark.parametrize(
    ("actor", "clip", "index"),
    [("1082", "ITS_HAP_XX", 23), ("1029", "ITS_HAP_XX", 31), ("1076", "IEO_HAP_HI", 44), ("1029", "TIE_SAD_XX", 59)],
)
def test_a_bare_face_talking_or_smiling_gets_no_eyeshadow(real_models, actor, clip, index) -> None:
    """These frames used to give a smoky eye (opacity up to 1.0) on one photo."""
    got = analyze_makeup(crema_frame(actor, clip, index), None, models=real_models).settings
    assert got.eyeshadow.opacity < 0.25
    assert got.eyeliner.opacity == 0.0
    assert got.lips.opacity == 0.0


@pytest.mark.parametrize(("actor", "factor"), [("1072", 1.45), ("1072", 1.6), ("1076", 1.6)])
def test_a_blown_out_photo_gives_no_dark_blush(real_models, actor, factor) -> None:
    bright = np.clip(crema_frame(actor).astype(np.float32) * factor, 0, 255).astype(np.uint8)
    result = analyze_makeup(bright, None, models=real_models)
    blush = result.settings.blush
    assert result.confidence["blush"] == 0.0
    assert blush.color == presets.load_preset(presets.DEFAULT_PRESET_NAME).blush.color
    assert any("白飛び" in note for note in result.notes)


def test_a_heavy_blush_does_not_tint_the_brows(real_models, models_path) -> None:
    bare = crema_frame("1029")
    truth = look("polished")
    truth.blush.color, truth.blush.opacity, truth.blush.shape, truth.blush.size = "#D6456E", 0.8, "round", 1.2
    got = analyze_makeup(pipeline_render(bare, truth, models_path), bare, models=real_models).settings
    lab = lab_of(got.brows.color)
    assert np.hypot(lab[1], lab[2]) < 6.0 or analyze._BROW_HUE_RANGE[0] - 2.0 <= np.degrees(np.arctan2(lab[2], lab[1])) <= analyze._BROW_HUE_RANGE[1] + 2.0
    assert got.blush.opacity > 0.5


def test_one_photo_finds_a_rose_lipstick_on_the_asian_test_face(real_models, models_path) -> None:
    truth = look("polished")
    truth.lips.color, truth.lips.opacity = "#B8606A", 0.6
    got = analyze_makeup(pipeline_render(crema_frame("1072"), truth, models_path), None, models=real_models).settings
    assert got.lips.opacity >= 0.5
    assert color_distance(got.lips.color, truth.lips.color) < 25.0


def test_numpy_runs_single_threaded_unless_the_user_says_otherwise() -> None:
    """Many BLAS threads on half the CPUs froze the video for over a second during a photo analysis."""
    code = "import makeup_cam, os; print(os.environ['OPENBLAS_NUM_THREADS'])"
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    run = lambda e: subprocess.run([sys.executable, "-c", code], cwd=root, env=e, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    assert run(env) == "1"
    assert run({**env, "OPENBLAS_NUM_THREADS": "3"}) == "3"

"""HairShapeEffect with real landmarks on a synthetic head: a dark hair cap and skin on a wall,
with thin strands drawn around it, so what is hair, wall and strand is known exactly.
Covers no-ops, flyaway removal with and without a plate, what must never be touched (the
hair's mass, the face, background lines), temporal behaviour, the warps and odd framings."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, fields

import cv2
import numpy as np
import pytest

from makeup_cam import regions
from makeup_cam.config import HairShapeSettings, MakeupSettings
from makeup_cam.effects import hairshape as hairshape_module
from makeup_cam.effects.base import FrameContext, registered_effects
from makeup_cam.effects.hairshape import HairShapeEffect
from makeup_cam.pipeline import FADE_FRAMES, MakeupPipeline
from makeup_cam.segmentation import SegMasks

SHAPE = (720, 1280)
WALL_BGR = (0.78, 0.84, 0.88)
HAIR_BGR = (0.08, 0.09, 0.11)
SKIN_BGR = (0.60, 0.70, 0.85)
OFF = {"flyaway": 0.0, "volume_top": 0.0, "width_sides": 0.0}


@dataclass(frozen=True)
class Scene:
    """A synthetic head; ``frame`` has strands, ``clean`` is the same without them."""

    frame: np.ndarray
    clean: np.ndarray
    plate: np.ndarray
    hair_mask: np.ndarray
    face_mask: np.ndarray
    landmarks: np.ndarray
    strands: np.ndarray  # where the strands are visible (bool)
    cap: np.ndarray  # the hair cap's true coverage (0..1)

    def ctx(self, frame: np.ndarray | None = None, *, plate: bool = False, intensity: float = 1.0) -> FrameContext:
        image = self.frame if frame is None else frame
        return FrameContext(
            height=image.shape[0],
            width=image.shape[1],
            landmarks=self.landmarks,
            hair_mask=self.hair_mask,
            face_skin_mask=self.face_mask,
            background=self.plate if plate else None,
            intensity=intensity,
        )


def wall_image(value: Sequence[float] = WALL_BGR) -> np.ndarray:
    """A wall with a soft light falloff, never perfectly flat."""
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(np.float32)
    falloff = 0.96 + 0.04 * (xx / SHAPE[1]) * (1.0 - 0.3 * yy / SHAPE[0])
    return np.dstack([np.float32(v) * falloff for v in value])


def draw_strands(
    image: np.ndarray, landmarks: np.ndarray, cap: np.ndarray, color: Sequence[float], count: int = 14
) -> None:
    """Thin curved strands starting at the cap's outline, going outward around the upper head."""
    scale = regions.face_scale(landmarks)
    center = landmarks[regions.FACE_OVAL].mean(axis=0)
    rng = np.random.default_rng(5)
    outline = (cap > 0.5).astype(np.uint8)
    for angle in np.linspace(-70.0, 70.0, count):
        direction = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
        distance = 0.0
        while outline[tuple(np.int32(center[::-1] + direction[::-1] * distance))]:
            distance += 1.0
        start = center + direction * (distance - 2.0)
        bend = np.array([-direction[1], direction[0]]) * rng.uniform(-0.1, 0.1) * scale
        ts = np.linspace(0.0, 1.0, 12)
        length = rng.uniform(0.25, 0.5) * scale
        points = start + np.outer(ts, direction) * length + np.outer(ts**2, bend)
        cv2.polylines(image, [np.round(points * 16).astype(np.int32)], False, tuple(color), 1, cv2.LINE_AA, 4)


def build_scene(
    landmarks: np.ndarray, plate_image: np.ndarray | None = None, strand_color: Sequence[float] = HAIR_BGR
) -> Scene:
    landmarks = landmarks.astype(np.float32)
    roi = regions.Roi(0, 0, SHAPE[1], SHAPE[0])
    scale = regions.face_scale(landmarks)
    oval = landmarks[regions.FACE_OVAL]
    lo, hi = oval.min(axis=0), oval.max(axis=0)
    center = np.array([(lo[0] + hi[0]) / 2, lo[1] + (hi[1] - lo[1]) * 0.35], np.float32)
    axes = ((hi[0] - lo[0]) / 2 + 0.5 * scale, center[1] - (lo[1] - 0.8 * scale))
    cap = regions.ellipse_mask(center, axes, 0.0, roi, feather_px=1.0)
    face = regions.polygon_mask(oval, roi, feather_px=1.0)
    plate = wall_image() if plate_image is None else plate_image
    hair = np.array(HAIR_BGR, np.float32) + 0.02 * np.sin(np.arange(SHAPE[1], dtype=np.float32) / 3.0)[None, :, None]
    clean = plate * (1.0 - cap[..., None]) + hair * cap[..., None]
    clean = clean * (1.0 - face[..., None]) + np.array(SKIN_BGR, np.float32) * face[..., None]
    frame = clean.copy()
    draw_strands(frame, landmarks, cap, strand_color)
    # A little sensor noise, the same in the frame and in the reference picture.
    noise = np.random.default_rng(1).normal(0.0, 1.0 / 255.0, frame.shape).astype(np.float32)
    frame = np.clip(frame + noise, 0.0, 1.0)
    clean = np.clip(clean + noise, 0.0, 1.0)
    # The hair mask as the segmenter gives it: blurry and a bit generous.
    hair_mask = cv2.GaussianBlur(np.maximum(cap - face, 0.0), (0, 0), 0.04 * scale)
    face_mask = cv2.GaussianBlur(face, (0, 0), 0.03 * scale)
    strands = np.abs(frame - clean).max(axis=2) > 0.03
    return Scene(frame, clean, plate, hair_mask, face_mask, landmarks, strands, cap)


@pytest.fixture
def scene(face_landmarks_720p: np.ndarray) -> Scene:
    return build_scene(face_landmarks_720p)


def run_effect(
    scene: Scene,
    frames: int = 6,
    *,
    plate: bool = False,
    frame: np.ndarray | None = None,
    intensity: float = 1.0,
    effect: HairShapeEffect | None = None,
    **values: float,
) -> np.ndarray:
    """Several frames, because a strand mask takes a few frames to rise fully."""
    effect = effect or HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0, **values})
    source = scene.frame if frame is None else frame
    out = source
    for _ in range(frames):
        out = effect.apply(source.copy(), scene.ctx(source, plate=plate, intensity=intensity), settings)
    return out


def residual(scene: Scene, image: np.ndarray) -> float:
    """How much is left of the strands (0 = gone) on the strand pixels."""
    return float(np.abs(image - scene.clean).max(axis=2)[scene.strands].mean())


def change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after - before).max(axis=2)


def hair_core(scene: Scene, inset: float = 0.03) -> np.ndarray:
    """Hair pixels this far (face scales) inside the cap's outline, away from the face."""
    size = 2 * max(1, round(inset * regions.face_scale(scene.landmarks))) + 1
    solid = cv2.erode((scene.cap > 0.99).astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    return solid & (scene.face_mask < 0.01)


def face_inside(scene: Scene) -> np.ndarray:
    return regions.face_oval_mask(scene.landmarks, regions.Roi(0, 0, SHAPE[1], SHAPE[0]), expand_px=-2.0) > 0.99


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_registered_under_its_settings_section() -> None:
    assert HairShapeEffect in registered_effects()
    assert (HairShapeEffect.name, HairShapeEffect.order) == ("hairshape", 8)
    assert HairShapeEffect.needs_hair_mask and HairShapeEffect.needs_face
    assert isinstance(getattr(MakeupSettings(), HairShapeEffect.name), HairShapeSettings)


def test_settings_are_what_the_effect_reads() -> None:
    assert {f.name for f in fields(HairShapeSettings)} == {
        "enabled",
        "flyaway",
        "volume_top",
        "width_sides",
        "use_background_plate",
    }


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_exact_noop_at_zero(scene: Scene, plate: bool) -> None:
    effect = HairShapeEffect()
    for _ in range(3):
        ctx = scene.ctx(plate=plate)
        mask = ctx.hair_mask
        out = effect.apply(scene.frame.copy(), ctx, HairShapeSettings(**OFF))
        assert np.array_equal(out, scene.frame)
        assert ctx.hair_mask is mask


@pytest.mark.parametrize("values", [{"flyaway": 1.0}, {"volume_top": 1.0}, {"width_sides": -1.0}])
def test_exact_noop_at_zero_intensity(scene: Scene, values: dict[str, float]) -> None:
    settings = HairShapeSettings(**{**OFF, **values})
    out = HairShapeEffect().apply(scene.frame.copy(), scene.ctx(intensity=0.0), settings)
    assert np.array_equal(out, scene.frame)


def test_nothing_without_a_face_or_a_hair_mask(scene: Scene) -> None:
    settings = HairShapeSettings(flyaway=1.0, volume_top=1.0, width_sides=1.0)
    ctx = scene.ctx()
    ctx.landmarks = None
    assert np.array_equal(HairShapeEffect().apply(scene.frame.copy(), ctx, settings), scene.frame)
    ctx = scene.ctx()
    ctx.hair_mask = None
    assert np.array_equal(HairShapeEffect().apply(scene.frame.copy(), ctx, settings), scene.frame)
    ctx = scene.ctx()
    ctx.hair_mask = np.zeros((10, 10), np.float32)  # from another resolution
    assert np.array_equal(HairShapeEffect().apply(scene.frame.copy(), ctx, settings), scene.frame)


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_output_is_a_valid_frame_with_everything_at_full(scene: Scene, plate: bool) -> None:
    out = run_effect(scene, plate=plate, flyaway=1.0, volume_top=1.0, width_sides=-1.0)
    assert out.dtype == np.float32 and out.shape == scene.frame.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# Flyaways
# ---------------------------------------------------------------------------


def test_scene_has_strands_worth_removing(scene: Scene) -> None:
    assert scene.strands.sum() > 200
    assert residual(scene, scene.frame) > 0.1


def test_strands_are_removed_on_a_plain_wall_without_a_plate(scene: Scene) -> None:
    out = run_effect(scene, flyaway=1.0)
    assert residual(scene, out) < 0.25 * residual(scene, scene.frame)


def test_strands_are_removed_with_a_plate(scene: Scene) -> None:
    out = run_effect(scene, plate=True, flyaway=1.0)
    assert residual(scene, out) < 0.15 * residual(scene, scene.frame)


def test_the_plate_also_works_on_a_busy_background(face_landmarks_720p: np.ndarray) -> None:
    busy = busy_room()
    scene = build_scene(face_landmarks_720p, busy)
    with_plate = run_effect(scene, plate=True, flyaway=1.0)
    assert residual(scene, with_plate) < 0.3 * residual(scene, scene.frame)


def test_bright_strands_on_a_dark_wall(face_landmarks_720p: np.ndarray) -> None:
    dark_wall = wall_image((0.12, 0.13, 0.15))
    scene = build_scene(face_landmarks_720p, dark_wall, strand_color=(0.55, 0.62, 0.7))
    # Light hair on a dark wall: the cap's color is the strands' color.
    light = np.array((0.55, 0.62, 0.7), np.float32)
    frame = scene.frame * (1.0 - scene.cap[..., None] * (scene.face_mask < 0.5)[..., None])
    frame += light * scene.cap[..., None] * (scene.face_mask < 0.5)[..., None]
    clean = scene.clean * (1.0 - scene.cap[..., None] * (scene.face_mask < 0.5)[..., None])
    clean += light * scene.cap[..., None] * (scene.face_mask < 0.5)[..., None]
    out = run_effect(scene, frame=frame, flyaway=1.0)
    strands = scene.strands & (scene.cap < 0.01)
    before = np.abs(frame - clean).max(axis=2)[strands].mean()
    after = np.abs(out - clean).max(axis=2)[strands].mean()
    assert after < 0.35 * before


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_the_main_hair_mass_is_never_eaten(scene: Scene, plate: bool) -> None:
    out = run_effect(scene, plate=plate, flyaway=1.0)
    diff = change(scene.frame, out)
    assert hair_core(scene).sum() > 5000
    assert diff[hair_core(scene)].max() < 0.02


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_the_face_is_never_touched(scene: Scene, plate: bool) -> None:
    out = run_effect(scene, plate=plate, flyaway=1.0)
    assert change(scene.frame, out)[face_inside(scene)].max() == 0.0


def test_a_hairline_with_loose_hair_on_the_skin_is_left_alone(scene: Scene) -> None:
    """Dark thin structures next to the skin look like strands; they are not flyaways."""
    frame = scene.frame.copy()
    top = scene.landmarks[regions.FACE_OVAL][:, 1].min()
    center_x = scene.landmarks[regions.FACE_OVAL][:, 0].mean()
    for dx in (-30, -10, 10, 30):
        start, end = (int(center_x + dx), int(top + 20)), (int(center_x + dx + 8), int(top + 60))
        cv2.line(frame, start, end, HAIR_BGR, 1, cv2.LINE_AA)
    out = run_effect(scene, frame=frame, flyaway=1.0)
    assert change(frame, out)[face_inside(scene)].max() == 0.0


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_a_highlight_as_bright_as_the_wall_stays_hair(scene: Scene, plate: bool) -> None:
    """Blond streaks can look just like the wall, thin dark hair between them like strands.
    Where the hair mask is sure of hair, nothing is replaced."""
    frame = scene.frame.copy()
    top = int(np.argmax(scene.cap[:, center_x(scene)] > 0.99))
    row = top + 6
    streaks = np.zeros(SHAPE, bool)
    streaks[row : row + 14, center_x(scene) - 40 : center_x(scene) + 40] = True
    frame[streaks] = np.array(WALL_BGR, np.float32)
    frame[row + 3 : row + 11 : 3, center_x(scene) - 38 : center_x(scene) + 38] = np.array(HAIR_BGR, np.float32)
    ctx = scene.ctx(frame, plate=plate)
    ctx.hair_mask = np.maximum(scene.hair_mask, cv2.GaussianBlur(scene.cap, (0, 0), 1.0))
    effect = HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    for _ in range(6):
        out = effect.apply(frame.copy(), ctx, settings)
    assert change(frame, out)[streaks].mean() < 0.01


def dense_wisps(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The frame with a soft, wispy edge beside the head: many hairs close together, with
    gaps of wall between them, so nothing there is thick and the mass does not protect it."""
    frame = scene.clean.copy()
    y = int(scene.landmarks[regions.LEFT_EYE_OUTER_CORNER][1]) - int(0.3 * scale_of(scene))
    for row in range(y - 45, y + 45, 6):
        x0 = side_of_hair(scene.clean, row, 1) - 2
        cv2.line(frame, (x0, row), (x0 + 24, row + 2), HAIR_BGR, 2, cv2.LINE_AA)
    wisps = (np.abs(frame - scene.clean).max(axis=2) > 0.03) & (scene.cap < 0.01)
    assert wisps.sum() > 300
    return frame, wisps


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_a_dense_wispy_edge_is_kept_while_loose_strands_go(scene: Scene, plate: bool) -> None:
    frame, wisps = dense_wisps(scene)
    out = run_effect(scene, plate=plate, frame=frame, flyaway=0.3)
    total = np.abs(frame - scene.clean).max(axis=2)[wisps].mean()
    assert change(frame, out)[wisps].mean() < (0.2 if plate else 0.05) * total  # removing it would look cut out
    assert residual(scene, run_effect(scene, plate=plate, flyaway=0.3)) < 0.8 * residual(scene, scene.frame)


def test_the_background_far_from_the_head_is_untouched(scene: Scene) -> None:
    for plate in (False, True):
        diff = change(scene.frame, run_effect(scene, plate=plate, flyaway=1.0))
        scale = regions.face_scale(scene.landmarks)
        solid = (scene.cap > 0.5).astype(np.uint8)
        far = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 5) > 1.3 * scale
        assert diff[far].max() == 0.0


def test_a_line_in_the_room_passing_the_head_stays(scene: Scene) -> None:
    """A door frame or picture edge is thin and dark but runs far beyond the hair."""
    room = wall_image()
    x = int(scene.landmarks[regions.FACE_OVAL][:, 0].max() + 0.75 * regions.face_scale(scene.landmarks))
    cv2.line(room, (x, 0), (x, SHAPE[0]), (0.25, 0.28, 0.3), 2, cv2.LINE_AA)
    with_line = build_scene(scene.landmarks, room)
    out = run_effect(with_line, flyaway=1.0)
    line = np.abs(room - wall_image()).max(axis=2) > 0.2
    line &= with_line.cap < 0.01
    assert line.sum() > 100
    assert change(with_line.frame, out)[line].mean() < 0.02


def busy_room() -> np.ndarray:
    """Book spines: many thin, strong structures everywhere."""
    rng = np.random.default_rng(3)
    room = np.full((*SHAPE, 3), 0.3, np.float32)
    for y in range(0, SHAPE[0], 120):
        x = 0
        while x < SHAPE[1]:
            width = int(rng.integers(6, 30))
            color = rng.uniform(0.1, 0.9, 3).astype(float)
            cv2.rectangle(room, (x, y), (x + width - 2, y + 115), tuple(color), -1)
            x += width
    return room


def test_a_busy_background_fades_the_method_without_a_plate(face_landmarks_720p: np.ndarray) -> None:
    scene = build_scene(face_landmarks_720p, busy_room())
    out = run_effect(scene, flyaway=1.0)
    outside = (scene.cap < 0.01) & (scene.face_mask < 0.01) & ~scene.strands
    assert change(scene.frame, out)[outside].mean() < 0.005  # books are not erased


def test_a_patterned_wall_keeps_its_pattern_without_a_plate(face_landmarks_720p: np.ndarray) -> None:
    """Short dark marks end inside the band like strands do; the same marks all over the
    wall farther out give them away."""
    rng = np.random.default_rng(11)
    wall = wall_image()
    for _ in range(9000):
        x, y = int(rng.integers(0, SHAPE[1])), int(rng.integers(0, SHAPE[0]))
        dx, dy = int(rng.integers(-5, 6)), int(rng.integers(-5, 6))
        cv2.line(wall, (x, y), (x + dx, y + dy), (0.3, 0.32, 0.35), 1, cv2.LINE_AA)
    scene = build_scene(face_landmarks_720p, wall)
    out = run_effect(scene, frame=scene.clean, flyaway=1.0)
    marks = (np.abs(wall - wall_image()).max(axis=2) > 0.2) & (scene.cap < 0.01) & (scene.face_mask < 0.01)
    assert marks.sum() > 1000
    near = cv2.dilate((scene.cap > 0.5).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    assert change(scene.clean, out)[marks & near].mean() < 0.03


def test_removal_grows_with_slider_and_intensity(scene: Scene) -> None:
    by_slider = [residual(scene, run_effect(scene, flyaway=value)) for value in (0.1, 0.3, 1.0)]
    assert by_slider[0] > by_slider[1] > by_slider[2]
    by_intensity = [residual(scene, run_effect(scene, flyaway=0.5, intensity=value)) for value in (0.2, 0.6, 1.0)]
    assert by_intensity[0] > by_intensity[1] > by_intensity[2]


def test_the_default_slider_visibly_works(scene: Scene) -> None:
    default = HairShapeSettings()
    assert default.flyaway > 0.0 and default.volume_top == 0.0 and default.width_sides == 0.0
    assert residual(scene, run_effect(scene, flyaway=default.flyaway)) < 0.8 * residual(scene, scene.frame)


# ---------------------------------------------------------------------------
# The plate
# ---------------------------------------------------------------------------


def test_plate_is_matched_to_the_live_exposure(scene: Scene) -> None:
    """Auto exposure changes after the plate was taken: no lighter or darker halo, and no
    need to give up on the plate."""
    brighter = np.clip(scene.frame * np.float32(1.12), 0.0, 1.0)
    effect = HairShapeEffect()
    out = run_effect(scene, frames=10, plate=True, frame=brighter, flyaway=1.0, effect=effect)
    expected = np.clip(scene.clean * np.float32(1.12), 0.0, 1.0)
    assert np.abs(out - expected).max(axis=2)[scene.strands & (scene.cap < 0.01)].mean() < 0.04
    assert effect._gain is not None and np.allclose(effect._gain, 1.12, atol=0.02)
    assert effect._trust is not None and effect._trust > 0.9


def test_plate_follows_a_slow_change_of_the_light(scene: Scene) -> None:
    """A lamp was switched on since the plate was taken: one side of the room is brighter.
    A single gain would leave a visible patch where strands were replaced."""
    left, right = scene.landmarks[regions.FACE_OVAL][:, 0].min(), scene.landmarks[regions.FACE_OVAL][:, 0].max()
    position = (np.arange(SHAPE[1], dtype=np.float32) - left) / (right - left)
    ramp = np.clip(0.75 + 0.25 * position, 0.7, 1.2)[None, :, None]
    lit_frame = np.clip(scene.frame * ramp, 0.0, 1.0)
    lit_clean = np.clip(scene.clean * ramp, 0.0, 1.0)
    effect = HairShapeEffect()
    out = run_effect(scene, frames=10, plate=True, frame=lit_frame, flyaway=1.0, effect=effect)
    strands = scene.strands & (scene.cap < 0.01)
    assert np.abs(out - lit_clean).max(axis=2)[strands].mean() < 0.03
    assert effect._trust is not None and effect._trust > 0.9


def test_a_plate_that_no_longer_fits_is_not_shown(scene: Scene) -> None:
    """The camera moved since: the old room must not appear around the head."""
    wrong = busy_room()
    ctx_scene = build_scene(scene.landmarks, wall_image())
    effect = HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    out = ctx_scene.frame
    for _ in range(30):
        ctx = FrameContext(
            height=SHAPE[0],
            width=SHAPE[1],
            landmarks=ctx_scene.landmarks,
            hair_mask=ctx_scene.hair_mask,
            face_skin_mask=ctx_scene.face_mask,
            background=wrong,
        )
        out = effect.apply(ctx_scene.frame.copy(), ctx, settings)
    near_strands = cv2.dilate(ctx_scene.strands.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    outside = (ctx_scene.cap < 0.01) & (ctx_scene.face_mask < 0.01) & ~near_strands
    assert change(ctx_scene.frame, out)[outside].max() < 0.05
    assert effect._trust is not None and effect._trust < 0.2
    # It fell back to the method without a plate, which still removes strands.
    assert residual(ctx_scene, out) < 0.5 * residual(ctx_scene, ctx_scene.frame)


def test_the_plate_can_be_switched_off(scene: Scene) -> None:
    switched_off = HairShapeSettings(**{**OFF, "flyaway": 1.0, "use_background_plate": False})
    with_flag = HairShapeEffect().apply(scene.frame.copy(), scene.ctx(plate=True), switched_off)
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    without = HairShapeEffect().apply(scene.frame.copy(), scene.ctx(plate=False), settings)
    assert np.array_equal(with_flag, without)


def test_a_plate_of_another_size_is_ignored(scene: Scene) -> None:
    ctx = scene.ctx()
    ctx.background = np.zeros((10, 10, 3), np.float32)
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    out = HairShapeEffect().apply(scene.frame.copy(), ctx, settings)
    assert np.array_equal(out, HairShapeEffect().apply(scene.frame.copy(), scene.ctx(), settings))


def test_a_read_only_plate_stays_as_it_was(scene: Scene) -> None:
    plate = scene.plate.copy()
    plate.setflags(write=False)
    ctx = scene.ctx()
    ctx.background = plate
    HairShapeEffect().apply(scene.frame.copy(), ctx, HairShapeSettings(flyaway=1.0, volume_top=-1.0))
    assert np.array_equal(plate, scene.plate)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_a_strand_does_not_flicker_with_noise(scene: Scene) -> None:
    rng = np.random.default_rng(9)
    effect = HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    previous_in = previous_out = None
    changes = []
    for _ in range(16):
        noisy = np.clip(scene.frame + rng.normal(0.0, 1.5 / 255.0, scene.frame.shape).astype(np.float32), 0.0, 1.0)
        out = effect.apply(noisy.copy(), scene.ctx(noisy), settings)
        if previous_in is not None:
            changes.append((np.abs(noisy - previous_in).max(axis=2), np.abs(out - previous_out).max(axis=2)))
        previous_in, previous_out = noisy, out.copy()
    near = cv2.dilate(scene.strands.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    settled = changes[4:]
    mean_before = np.mean([before[near].mean() for before, _ in settled])
    assert np.mean([after[near].mean() for _, after in settled]) <= 1.05 * mean_before
    assert max(float(cv2.blur(after - before, (5, 5))[near].max()) for before, after in settled) < 0.1


def test_memory_rises_fast_decays_slower_and_follows_the_head(scene: Scene) -> None:
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    assert head is not None
    memory = hairshape_module._StrandMemory()
    alpha = np.zeros((head.work_size[1], head.work_size[0]), np.float32)
    alpha[40, 40] = 1.0
    first = memory.update(alpha, head)
    assert first[40, 40] == pytest.approx(hairshape_module._MEMORY_RISE)
    second = memory.update(alpha, head)
    assert second[40, 40] > first[40, 40]
    gone = memory.update(np.zeros_like(alpha), head)
    assert gone[40, 40] == pytest.approx(second[40, 40] * hairshape_module._MEMORY_DECAY)
    # The head moves down by 4 working-copy pixels; the memory goes with it.
    shift = 4 * head.factor
    moved = dataclasses.replace(head, center=head.center + np.float32([0, shift]))
    followed = memory.update(np.zeros_like(alpha), moved)
    assert followed[44, 40] == pytest.approx(gone[40, 40] * hairshape_module._MEMORY_DECAY)
    assert followed[40, 40] == 0.0
    # A jump (the face was re-detected elsewhere) forgets it.
    far = dataclasses.replace(head, center=head.center + np.float32([head.scale, 0]))
    assert memory.update(np.zeros_like(alpha), far).max() == 0.0


def test_reset_forgets_everything(scene: Scene) -> None:
    fresh = run_effect(scene, frames=1, plate=True)
    effect = HairShapeEffect()
    run_effect(scene, frames=5, plate=True, volume_top=1.0, effect=effect)
    effect.reset()
    assert np.array_equal(run_effect(scene, frames=1, plate=True, effect=effect), fresh)


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_a_head_moving_through_the_lagging_mask_keeps_its_mass(scene: Scene, plate: bool) -> None:
    """The hair mask lags behind a nod, so real hair lies where it says "no hair"; what the
    image shows as the mass still stays."""
    effect = HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    worst = 0.0
    for step in range(16):
        dy = 20.0 * np.sin(step / 3.0)
        lag = 20.0 * np.sin((step - 3) / 3.0)
        matrix = np.float32([[1, 0, 0], [0, 1, dy]])
        frame = cv2.warpAffine(scene.frame, matrix, SHAPE[::-1], borderMode=cv2.BORDER_REPLICATE)
        cap = cv2.warpAffine(scene.cap, matrix, SHAPE[::-1])
        mask_matrix = np.float32([[1, 0, 0], [0, 1, lag]])
        ctx = FrameContext(
            height=SHAPE[0],
            width=SHAPE[1],
            landmarks=scene.landmarks + np.float32([0, dy]),
            hair_mask=cv2.warpAffine(scene.hair_mask, mask_matrix, SHAPE[::-1]),
            face_skin_mask=cv2.warpAffine(scene.face_mask, mask_matrix, SHAPE[::-1]),
            background=scene.plate if plate else None,
        )
        out = effect.apply(frame.copy(), ctx, settings)
        face = cv2.warpAffine(scene.face_mask, matrix, SHAPE[::-1])
        solid = cv2.erode((cap > 0.99).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (face < 0.01)
        worst = max(worst, float(change(frame, out)[solid].max()))
    assert worst < 0.03


# ---------------------------------------------------------------------------
# Warps
# ---------------------------------------------------------------------------


def top_of_hair(image: np.ndarray, x: int) -> int:
    """First row from the top at column ``x`` that is hair-dark."""
    return int(np.argmax(image[:, x].max(axis=1) < 0.35))


def side_of_hair(image: np.ndarray, y: int, side: int) -> int:
    """Outermost hair-dark column in row ``y`` on one side (-1 left, +1 right of the image)."""
    dark = np.flatnonzero(image[y].max(axis=1) < 0.35)
    return int(dark[0] if side < 0 else dark[-1])


def scale_of(scene: Scene) -> float:
    return regions.face_scale(scene.landmarks)


def center_x(scene: Scene) -> int:
    return int(scene.landmarks[regions.FACE_OVAL][:, 0].mean())


def eye_row(scene: Scene) -> int:
    return int((regions.eye_center(scene.landmarks, "left")[1] + regions.eye_center(scene.landmarks, "right")[1]) / 2)


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_volume_up_lifts_the_top_of_the_hair(scene: Scene, plate: bool) -> None:
    clean = run_effect(scene, plate=plate, frame=scene.clean, flyaway=0.0, volume_top=1.0)
    moved = top_of_hair(scene.clean, center_x(scene)) - top_of_hair(clean, center_x(scene))
    assert 0.05 * scale_of(scene) < moved <= 0.16 * scale_of(scene)


def test_shrinking_is_capped_very_small_without_a_plate(scene: Scene) -> None:
    out = run_effect(scene, frame=scene.clean, flyaway=0.0, volume_top=-1.0)
    moved = top_of_hair(out, center_x(scene)) - top_of_hair(scene.clean, center_x(scene))
    assert moved <= max(2.0, 0.03 * scale_of(scene))


def test_shrinking_with_a_plate_goes_further_and_shows_the_room(scene: Scene) -> None:
    out = run_effect(scene, plate=True, frame=scene.clean, flyaway=0.0, volume_top=-1.0)
    moved = top_of_hair(out, center_x(scene)) - top_of_hair(scene.clean, center_x(scene))
    assert 0.04 * scale_of(scene) < moved <= 0.12 * scale_of(scene)
    depth = cv2.distanceTransform((scene.cap > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    exposed = (scene.cap > 0.99) & (depth < 0.5 * moved)
    exposed &= np.arange(SHAPE[0])[:, None] < eye_row(scene)
    exposed &= np.abs(np.arange(SHAPE[1])[None, :] - center_x(scene)) < 0.3 * scale_of(scene)  # the very top
    assert exposed.sum() > 10
    assert np.abs(out - scene.plate).max(axis=2)[exposed].mean() < 0.1  # the real wall


def test_shrinking_leaves_no_ghost_of_a_soft_outline(scene: Scene) -> None:
    """The soft edge of the old outline must not stay behind, however faint."""
    soft = cv2.GaussianBlur(scene.clean, (0, 0), 2.5)
    out = run_effect(scene, plate=True, frame=soft, flyaway=0.0, volume_top=-1.0)
    x = center_x(scene)
    row = top_of_hair(soft, x)
    assert top_of_hair(out, x) > row + 2  # it did shrink
    assert np.abs(out - scene.plate).max(axis=2)[row - 10 : row - 4, x].max() < 0.006


@pytest.mark.parametrize(("value", "direction"), [(1.0, 1), (-1.0, -1)])
def test_width_moves_the_sides(scene: Scene, value: float, direction: int) -> None:
    out = run_effect(scene, plate=True, frame=scene.clean, flyaway=0.0, width_sides=value)
    row = eye_row(scene) - int(0.3 * scale_of(scene))
    moved_left = side_of_hair(scene.clean, row, -1) - side_of_hair(out, row, -1)
    moved_right = side_of_hair(out, row, 1) - side_of_hair(scene.clean, row, 1)
    assert direction * moved_left >= 2 and direction * moved_right >= 2
    assert abs(moved_left) <= 0.14 * scale_of(scene) and abs(moved_right) <= 0.14 * scale_of(scene)


@pytest.mark.parametrize("plate", [False, True], ids=["no-plate", "plate"])
def test_warps_leave_the_face_where_it_is(scene: Scene, plate: bool) -> None:
    out = run_effect(scene, plate=plate, flyaway=0.0, volume_top=1.0, width_sides=-1.0)
    assert change(scene.frame, out)[face_inside(scene)].max() < 1e-3


def test_with_a_plate_the_room_does_not_bend(scene: Scene) -> None:
    room = wall_image()
    x = int(scene.landmarks[regions.FACE_OVAL][:, 0].max() + 0.75 * scale_of(scene))
    cv2.line(room, (x, 0), (x, SHAPE[0]), (0.25, 0.28, 0.3), 2, cv2.LINE_AA)
    with_line = build_scene(scene.landmarks, room)
    out = run_effect(with_line, plate=True, frame=with_line.clean, flyaway=0.0, volume_top=1.0, width_sides=1.0)
    line = np.abs(room - wall_image()).max(axis=2) > 0.2
    line &= cv2.dilate((with_line.cap > 0.01).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    assert line.sum() > 100
    assert change(with_line.clean, out)[line].max() < 0.02


def test_warp_field_is_zero_at_the_border_of_its_area(scene: Scene) -> None:
    """Anything else would show as a seam where the work area ends."""
    effect = HairShapeEffect()
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    assert head is not None
    outline = hairshape_module._silhouette(head, scene.ctx())
    field = effect._warp_field(scene.landmarks, scene.frame.shape, head, outline, 1.0, 1.0, None)
    assert field is not None
    for component in field:
        border = np.concatenate((component[0], component[-1], component[:, 0], component[:, -1]))
        assert np.abs(border).max() < 0.5
        assert np.abs(component).max() > 2.0


def test_no_hair_means_no_warp(scene: Scene) -> None:
    ctx = scene.ctx(scene.clean)
    ctx.hair_mask = np.zeros(SHAPE, np.float32)
    settings = HairShapeSettings(**{**OFF, "volume_top": 1.0, "width_sides": 1.0})
    out = HairShapeEffect().apply(scene.clean.copy(), ctx, settings)
    assert np.array_equal(out, scene.clean)


def test_the_hair_mask_is_warped_along_for_later_effects(scene: Scene) -> None:
    ctx = scene.ctx()
    original = ctx.hair_mask.copy()
    HairShapeEffect().apply(scene.frame.copy(), ctx, HairShapeSettings(**{**OFF, "volume_top": 1.0}))
    assert np.array_equal(scene.hair_mask, original)  # the shared mask itself is not edited
    assert ctx.hair_mask is not scene.hair_mask and not ctx.hair_mask.flags.writeable
    x = center_x(scene)
    assert int(np.argmax(ctx.hair_mask[:, x] > 0.5)) < int(np.argmax(original[:, x] > 0.5))


def test_warp_strength_follows_the_intensity(scene: Scene) -> None:
    x = center_x(scene)

    def lift(intensity: float) -> int:
        out = run_effect(scene, frame=scene.clean, flyaway=0.0, volume_top=1.0, intensity=intensity)
        return top_of_hair(scene.clean, x) - top_of_hair(out, x)

    assert lift(0.25) < lift(1.0)


def test_warp_never_folds(scene: Scene) -> None:
    effect = HairShapeEffect()
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    assert head is not None
    outline = hairshape_module._silhouette(head, scene.ctx())
    for volume, width in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)):
        fit = hairshape_module._PlateFit(np.ones(3, np.float32), np.ones(outline.ring.shape, np.float32), 1.0)
        field = effect._warp_field(scene.landmarks, scene.frame.shape, head, outline, volume, width, fit)
        assert field is not None
        for component, axis in zip(field, (1, 0)):
            # Displacement per pixel well below 1: the map stays one-to-one.
            assert np.abs(np.diff(component, axis=axis)).max() / head.stride < 0.6


# ---------------------------------------------------------------------------
# Odd framings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [(0.0, -150.0), (0.0, 320.0), (-340.0, 0.0), (480.0, 0.0), (0.0, 2000.0)], ids=str)
def test_head_at_and_beyond_the_frame_border(scene: Scene, shift: tuple[float, float]) -> None:
    moved = scene.landmarks + np.float32(shift)
    ctx = scene.ctx(plate=True)
    ctx.landmarks = moved
    settings = HairShapeSettings(flyaway=1.0, volume_top=-1.0, width_sides=1.0)
    out = HairShapeEffect().apply(scene.frame.copy(), ctx, settings)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_small_and_close_up_faces(scene: Scene) -> None:
    for zoom in (0.4, 2.5):
        matrix = cv2.getRotationMatrix2D((640.0, 300.0), 0.0, zoom)
        size = SHAPE[::-1]
        frame = cv2.warpAffine(scene.frame, matrix, size, borderMode=cv2.BORDER_REPLICATE)
        plate = cv2.warpAffine(scene.plate, matrix, size, borderMode=cv2.BORDER_REPLICATE)
        landmarks = cv2.transform(scene.landmarks[None], matrix)[0]
        ctx = FrameContext(
            height=SHAPE[0],
            width=SHAPE[1],
            landmarks=landmarks,
            hair_mask=cv2.warpAffine(scene.hair_mask, matrix, size),
            face_skin_mask=cv2.warpAffine(scene.face_mask, matrix, size),
            background=plate,
        )
        out = HairShapeEffect().apply(frame.copy(), ctx, HairShapeSettings(flyaway=1.0, volume_top=1.0))
        assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_close_up_faces_use_a_coarser_working_copy() -> None:
    assert hairshape_module._work_factor(60.0, None) == 1
    assert hairshape_module._work_factor(125.0, None) == 2
    assert hairshape_module._work_factor(300.0, None) == 4
    # Hysteresis: a head hovering at a threshold does not flip resolutions.
    threshold = 1.5 * hairshape_module._WORK_SCALE
    assert hairshape_module._work_factor(threshold + 2.0, 1) == 1
    assert hairshape_module._work_factor(threshold - 2.0, 2) == 2


def test_rolled_head(scene: Scene) -> None:
    matrix = cv2.getRotationMatrix2D((640.0, 300.0), 12.0, 1.0)
    size = SHAPE[::-1]
    frame = cv2.warpAffine(scene.frame, matrix, size, borderMode=cv2.BORDER_REPLICATE)
    clean = cv2.warpAffine(scene.clean, matrix, size, borderMode=cv2.BORDER_REPLICATE)
    landmarks = cv2.transform(scene.landmarks[None], matrix)[0]
    effect = HairShapeEffect()
    out = frame
    for _ in range(6):
        ctx = FrameContext(
            height=SHAPE[0],
            width=SHAPE[1],
            landmarks=landmarks,
            hair_mask=cv2.warpAffine(scene.hair_mask, matrix, size),
            face_skin_mask=cv2.warpAffine(scene.face_mask, matrix, size),
        )
        out = effect.apply(frame.copy(), ctx, HairShapeSettings(flyaway=1.0))
    strands = np.abs(frame - clean).max(axis=2) > 0.03
    assert np.abs(out - clean).max(axis=2)[strands].mean() < 0.3 * np.abs(frame - clean).max(axis=2)[strands].mean()


def test_a_head_filling_the_frame_leaves_nothing_to_compare_with(scene: Scene) -> None:
    matrix = cv2.getRotationMatrix2D((640.0, 200.0), 0.0, 6.0)
    size = SHAPE[::-1]
    frame = cv2.warpAffine(scene.frame, matrix, size, borderMode=cv2.BORDER_REPLICATE)
    ctx = FrameContext(
        height=SHAPE[0],
        width=SHAPE[1],
        landmarks=cv2.transform(scene.landmarks[None], matrix)[0],
        hair_mask=np.ones(SHAPE, np.float32),
        face_skin_mask=None,
        background=scene.plate,
    )
    effect = HairShapeEffect()
    for _ in range(3):
        out = effect.apply(frame.copy(), ctx, HairShapeSettings(flyaway=1.0, volume_top=1.0))
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_works_without_skin_masks(scene: Scene) -> None:
    ctx = scene.ctx()
    ctx.face_skin_mask = None
    out = HairShapeEffect().apply(scene.frame.copy(), ctx, HairShapeSettings(**{**OFF, "flyaway": 1.0}))
    assert change(scene.frame, out)[face_inside(scene)].max() == 0.0


# ---------------------------------------------------------------------------
# In the pipeline
# ---------------------------------------------------------------------------


class ScriptedTracker:
    def __init__(self, script: list[np.ndarray | None]) -> None:
        self.script = list(script)

    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        landmarks = self.script.pop(0)
        return (None, None) if landmarks is None else (landmarks.copy(), np.zeros(len(landmarks), np.float32))

    def set_smoothing(self, smoothing: float) -> None:
        pass

    def close(self) -> None:
        pass


class FixedSegmenter:
    def __init__(self, masks: SegMasks) -> None:
        self.masks = masks

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        return self.masks

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_a_bow_fades_the_effect_out_and_back_in_smoothly(
    scene: Scene, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path))
    frame = (scene.frame * 255.0).astype(np.uint8)
    present = [scene.landmarks] * 8
    script = present + [None] * 12 + present
    masks = SegMasks(scene.hair_mask, scene.face_mask, None)
    settings = MakeupSettings()
    settings.general.mirror_preview = False
    settings.hairshape.flyaway = 1.0
    pipeline = MakeupPipeline(
        settings, tracker=ScriptedTracker(script), segmenter=FixedSegmenter(masks), effects=[HairShapeEffect()]
    )
    try:
        outputs = [pipeline.process(frame, 33 * i).output for i in range(len(script))]
    finally:
        pipeline.close()
    near = cv2.dilate(scene.strands.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    strength = [float(np.abs(out.astype(np.float32) - frame)[near].mean()) for out in outputs]
    assert strength[7] > 5.0  # working before the bow
    assert strength[8 + FADE_FRAMES + 1] == 0.0  # gone during it: the frame passes through
    # No jump larger than a fair share of the effect between two frames, in or out.
    assert max(abs(a - b) for a, b in zip(strength, strength[1:])) < 0.6 * strength[7]
    assert strength[-1] > 0.7 * strength[7]  # and back after it


def test_a_real_segmentation_run_keeps_the_portrait_intact(face_frame_720p, face_landmarks_720p, models_dir) -> None:
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
    landmarks = face_landmarks_720p.astype(np.float32)
    ctx = FrameContext(
        height=720,
        width=1280,
        landmarks=landmarks,
        hair_mask=masks.hair,
        face_skin_mask=masks.face_skin,
        body_skin_mask=masks.body_skin,
    )
    effect = HairShapeEffect()
    settings = HairShapeSettings(flyaway=1.0)
    out = frame
    for _ in range(5):
        out = effect.apply(frame.copy(), ctx, settings)
    diff = change(frame, out)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert diff[regions.face_oval_mask(landmarks, regions.Roi(0, 0, 1280, 720)) > 0.99].max() == 0.0
    solid = cv2.erode((masks.hair > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert solid.sum() > 300 and diff[solid].max() < 0.1
    assert diff[:60, :60].max() == 0.0


# ---------------------------------------------------------------------------
# A warp over the room: nothing but the person moves, and only what moves can change
# ---------------------------------------------------------------------------


def with_skin_like_the_wall(scene: Scene) -> Scene:
    """Very bright skin next to a wall of nearly the same color: the image alone cannot tell them apart."""
    wall = cv2.GaussianBlur(scene.plate, (0, 0), 3.0)
    face = scene.face_mask[..., None]
    skin = wall * np.array((0.99, 1.0, 1.01), np.float32)
    frame = np.clip(scene.frame * (1.0 - face) + skin * face, 0.0, 1.0)
    return dataclasses.replace(scene, frame=frame, clean=frame.copy())


@pytest.mark.parametrize("value", [0.3, -0.5, 1.0])
def test_a_warp_never_paints_the_wall_on_skin_that_looks_like_it(scene: Scene, value: float) -> None:
    pale = with_skin_like_the_wall(scene)
    out = run_effect(pale, plate=True, flyaway=0.0, volume_top=value, width_sides=value)
    assert change(pale.frame, out)[face_inside(pale)].max() < 1e-6


@pytest.mark.parametrize("value", [0.3, -0.5, 1.0])
def test_a_warp_never_shows_a_wrong_plate_on_the_face(scene: Scene, value: float) -> None:
    """Part of the room changed: the plate has a poster next to the head that the room no longer has."""
    plate = scene.plate.copy()
    oval = scene.landmarks[regions.FACE_OVAL]
    x, y, size = int(oval[:, 0].max()), int(oval[:, 1].min()), regions.face_scale(scene.landmarks)
    cv2.rectangle(plate, (x + int(0.9 * size), y - int(0.4 * size)), (x + int(1.9 * size), y + int(0.8 * size)), (0.5, 0.2, 0.2), -1)
    changed = dataclasses.replace(scene, plate=plate)
    effect = HairShapeEffect()
    out = run_effect(changed, 12, plate=True, effect=effect, flyaway=0.0, volume_top=value, width_sides=value)
    assert change(scene.frame, out)[face_inside(scene)].max() < 1e-6
    assert effect._trust is not None and effect._trust < 0.2


def test_the_forehead_between_hairline_and_face_oval_is_part_of_the_silhouette(scene: Scene) -> None:
    """A gap in the hair mask there is no background, however the mask looks."""
    oval = scene.landmarks[regions.FACE_OVAL]
    solid = (scene.cap > 0.5).astype(np.uint8)
    only_the_rim = solid - cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    holed = dataclasses.replace(scene, hair_mask=only_the_rim.astype(np.float32))
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    outline = hairshape_module._silhouette(head, holed.ctx())
    xs, ys = head.cells()
    top = oval[:, 1].min()
    column = np.argmin(np.abs(xs[0] - oval[:, 0].mean()))
    rows = (ys[:, 0] < top - 2) & (ys[:, 0] > top - 18)
    assert rows.any()
    assert (outline.distance[rows, column] < 0).all()


def moved(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def room_with_lines(landmarks: np.ndarray) -> np.ndarray:
    """A plain wall with a few lines around the head: nothing but them can show that the camera moved."""
    room = wall_image()
    oval = landmarks[regions.FACE_OVAL]
    size = regions.face_scale(landmarks)
    right, left, top = int(oval[:, 0].max() + size), int(oval[:, 0].min() - size), int(oval[:, 1].min() - 1.1 * size)
    cv2.line(room, (right, 0), (right, SHAPE[0]), (0.25, 0.3, 0.35), 3, cv2.LINE_AA)
    cv2.line(room, (0, top + 10), (SHAPE[1], top - 10), (0.3, 0.32, 0.36), 3, cv2.LINE_AA)
    cv2.rectangle(room, (left - 60, top + 40), (left, top + 160), (0.25, 0.25, 0.3), 3, cv2.LINE_AA)
    return room


@pytest.mark.parametrize("shift", [(1, 1), (1, 0), (0, 1), (3, 2), (20, 10), (40, 20)])
def test_a_camera_that_moved_a_little_does_not_double_the_lines_of_the_room(
    face_landmarks_720p: np.ndarray, shift: tuple[int, int]
) -> None:
    scene = build_scene(face_landmarks_720p, room_with_lines(face_landmarks_720p))
    old_plate = dataclasses.replace(scene, plate=moved(scene.plate, *shift))
    effect = HairShapeEffect()
    out = run_effect(old_plate, 12, plate=True, effect=effect, flyaway=1.0)
    assert effect._trust is not None and effect._trust < 0.2
    around = (scene.cap < 0.01) & (scene.face_mask < 0.01) & (np.abs(scene.frame - wall_image()).max(axis=2) > 0.1)
    around &= ~cv2.dilate(scene.strands.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert around.sum() > 100
    assert change(scene.frame, out)[around].mean() < 0.02


def test_a_plate_that_fits_stays_trusted_through_noise_and_exposure(face_landmarks_720p: np.ndarray) -> None:
    scene = build_scene(face_landmarks_720p, room_with_lines(face_landmarks_720p))
    rng = np.random.default_rng(2)
    effect = HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    for _ in range(12):
        frame = np.clip(scene.frame * 0.9 + rng.normal(0.0, 3.0 / 255.0, scene.frame.shape), 0.0, 1.0).astype(np.float32)
        effect.apply(frame, scene.ctx(frame, plate=True), settings)
    assert effect._trust is not None and effect._trust > 0.95


def test_a_moved_camera_is_no_problem_on_a_plain_wall(scene: Scene) -> None:
    """A shift cannot show there, so the plate stays usable."""
    effect = HairShapeEffect()
    shifted = dataclasses.replace(scene, plate=moved(scene.plate, 3, 2))
    out = run_effect(shifted, 12, plate=True, effect=effect, flyaway=1.0)
    assert effect._trust is not None and effect._trust > 0.95
    assert residual(scene, out) < 0.25 * residual(scene, scene.frame)


def test_alignment_is_checked_again_after_the_camera_moves(face_landmarks_720p: np.ndarray) -> None:
    scene = build_scene(face_landmarks_720p, room_with_lines(face_landmarks_720p))
    effect = HairShapeEffect()
    run_effect(scene, 12, plate=True, effect=effect, flyaway=1.0)
    assert effect._trust is not None and effect._trust > 0.95
    run_effect(dataclasses.replace(scene, plate=moved(scene.plate, 2, 1)), 16, plate=True, effect=effect, flyaway=1.0)
    assert effect._trust < 0.2


def test_with_a_textured_mass_strands_do_not_stand_or_fall_with_the_border(face_landmarks_720p: np.ndarray) -> None:
    """Hair's own texture connects to where the hair leaves the work area; its strands are still loose."""
    scene = build_scene(face_landmarks_720p)
    oval = scene.landmarks[regions.FACE_OVAL]
    size = regions.face_scale(scene.landmarks)
    locks = np.zeros(SHAPE, np.float32)
    bottom = int(oval[:, 1].max() + 0.6 * size)
    for x0, x1 in ((oval[:, 0].min() - 0.5 * size, oval[:, 0].min()), (oval[:, 0].max(), oval[:, 0].max() + 0.5 * size)):
        locks[int(oval[:, 1].mean()) : bottom, int(x0) : int(x1)] = 1.0
    locks = np.maximum(locks * (1.0 - scene.face_mask), np.maximum(scene.cap - scene.face_mask, 0.0))
    # Light streaks: the dark gaps between them look like thin dark structures, all over the hair.
    streaks = (np.arange(SHAPE[1], dtype=np.float32) % 4 < 2)[None, :, None] * np.float32(0.2)
    hair = np.array(HAIR_BGR, np.float32) + streaks
    frame = scene.frame * (1.0 - locks[..., None]) + hair * locks[..., None]
    clean = scene.clean * (1.0 - locks[..., None]) + hair * locks[..., None]
    mask = cv2.GaussianBlur(locks, (0, 0), 0.04 * size)
    long_hair = dataclasses.replace(scene, frame=frame, clean=clean, hair_mask=mask, cap=np.maximum(scene.cap, locks))
    strands = scene.strands & (locks < 0.01)
    assert strands.sum() > 100
    before = np.abs(frame - clean).max(axis=2)[strands].mean()
    effect = HairShapeEffect()
    settings = HairShapeSettings(**{**OFF, "flyaway": 1.0})
    left = []
    for _ in range(10):
        out = effect.apply(frame.copy(), long_hair.ctx(frame), settings)
        left.append(float(np.abs(out - clean).max(axis=2)[strands].mean()))
    assert left[-1] < 0.5 * before
    assert np.abs(np.diff(left[4:])).max() < 0.1 * before  # no pulsing either


def strands_alpha(head_shape: tuple[int, int], *lines: tuple[int, int, int, int]) -> np.ndarray:
    alpha = np.zeros(head_shape, np.float32)
    for x0, y0, x1, y1 in lines:
        cv2.line(alpha, (x0, y0), (x1, y1), 1.0, 1)
    return alpha


def test_the_tip_of_a_strand_does_not_make_it_background(scene: Scene) -> None:
    """Going far out of the band for a stretch is what a door frame does; a tip only touches that zone."""
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    w, h = head.work_size
    gw, gh = head.grid_size
    # Distance grows from the left of the area to the right: -0.5 at the start, 1.5 scales at the end.
    distance = np.tile(np.linspace(-0.5, 1.5, gw, dtype=np.float32), (gh, 1))
    band = np.ones((h, w), np.float32)
    reach = 0.3
    beyond = int(w * (reach + hairshape_module._LONG_STRUCTURE + 0.5) / 2.0)  # where distance gets beyond that
    no_memory = np.zeros((h, w), np.float32)
    length = hairshape_module._LEAVES_LENGTH * head.work_scale
    tip = strands_alpha((h, w), (w // 4, h // 2, beyond + int(0.3 * length), h // 2))
    assert hairshape_module._keep_attached(tip, band, distance, head, reach, no_memory).max() > 0.5
    long_line = strands_alpha((h, w), (w // 4, h // 2, beyond + int(1.5 * length) + 1, h // 2))
    assert hairshape_module._keep_attached(long_line, band, distance, head, reach, no_memory).max() == 0.0
    # The same structure while it is being removed stays removed: the verdict does not flip by a few pixels.
    held = cv2.dilate(long_line, np.ones((3, 3), np.uint8))
    assert hairshape_module._keep_attached(long_line, band, distance, head, reach, held).max() > 0.5


@pytest.mark.parametrize("value", [0.3, 1.0])
def test_without_a_plate_a_thin_line_next_to_the_hair_does_not_bend(face_landmarks_720p: np.ndarray, value: float) -> None:
    room = wall_image()
    size = regions.face_scale(face_landmarks_720p)
    base = build_scene(face_landmarks_720p, room)
    columns = np.flatnonzero((base.cap > 0.5).any(axis=0))
    xs = [int(columns.min() - 0.25 * size), int(columns.max() + 0.25 * size)]
    for x in xs:
        cv2.line(room, (x, 0), (x, SHAPE[0]), (0.2, 0.22, 0.25), 2, cv2.LINE_AA)
    scene = build_scene(face_landmarks_720p, room)
    out = run_effect(scene, frame=scene.clean, flyaway=0.0, volume_top=value, width_sides=value)
    rows = np.flatnonzero((scene.cap > 0.5).any(axis=1))
    for x in xs:
        strip = (slice(rows.min(), rows.max()), slice(x - 3, x + 4))
        assert change(scene.clean, out)[strip].max() < 0.02
    assert top_of_hair(out, center_x(scene)) < top_of_hair(scene.clean, center_x(scene))  # the hair did move


def test_without_a_plate_growing_hair_has_no_gap_or_ghost(face_landmarks_720p: np.ndarray) -> None:
    scene = build_scene(face_landmarks_720p)
    out = run_effect(scene, frame=scene.clean, flyaway=0.0, volume_top=1.0)
    x = center_x(scene)
    old_top, new_top = top_of_hair(scene.clean, x), top_of_hair(out, x)
    assert new_top < old_top
    assert out[new_top + 2 : old_top + 4, x].max() < 0.4  # solid hair from the new outline down
    wall = np.arange(SHAPE[0]) < new_top - 3
    assert change(scene.clean, out)[wall, x].max() < 0.02  # and the plain room above it as it was


def test_a_busy_background_is_known_as_such_without_a_plate(face_landmarks_720p: np.ndarray) -> None:
    """Books thick enough to look like the person would tear from the gaps between them, so they bend along."""
    scene = build_scene(face_landmarks_720p, busy_room())
    effect = HairShapeEffect()
    out = run_effect(scene, 8, effect=effect, frame=scene.clean, flyaway=0.0, width_sides=0.5, volume_top=0.5)
    assert effect._room_busy is not None and effect._room_busy > 0.9
    assert np.isfinite(out).all()


@pytest.mark.parametrize("share", [0.4, 0.6])
def test_something_thick_outside_the_silhouette_is_not_a_change_of_the_light(scene: Scene, share: float) -> None:
    """A shoulder or a ponytail the mask does not know of must not darken what the wall should look like,
    when it is much darker than any change of the light could explain."""
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    outline = hairshape_module._silhouette(head, scene.ctx())
    plate_colors = head.sample(scene.plate)
    colors = plate_colors.copy()
    outside = np.argwhere(outline.distance > 0.3)
    rows, cols = outside[:, 0], outside[:, 1]
    shoulder = rows > np.percentile(rows, 70)  # the lower part of what is outside
    colors[rows[shoulder], cols[shoulder]] *= np.float32(share)
    light = hairshape_module._light_change(colors, plate_colors, outline, head)
    assert np.abs(light[rows[~shoulder], cols[~shoulder]] - 1.0).max() < 0.03


def test_sampling_more_points_than_one_remap_can_take() -> None:
    image = np.dstack([np.tile(np.arange(100, dtype=np.float32), (50, 1))] * 3)
    xs = np.linspace(0.0, 99.0, 9001, dtype=np.float32)
    samples = hairshape_module._bilinear(image, xs, np.full_like(xs, 10.0))
    assert samples.shape == (9001, 3)
    assert np.abs(samples[:, 1] - xs).max() < 1e-2


def test_a_light_switch_near_the_hair_is_not_a_strand(face_landmarks_720p: np.ndarray) -> None:
    """Thin and dark like a strand, close to the hair, but it does not start at the hair."""
    room = wall_image()
    size = regions.face_scale(face_landmarks_720p)
    base = build_scene(face_landmarks_720p, room)
    rows = np.flatnonzero((base.cap > 0.5).any(axis=1))
    y = int(rows.min() + 0.5 * (rows.max() - rows.min()))
    x = int(np.flatnonzero(base.cap[y] > 0.5).max() + 0.3 * size)
    cv2.rectangle(room, (x, y), (x + int(0.3 * size), y + int(0.45 * size)), (0.25, 0.25, 0.28), 2, cv2.LINE_AA)
    scene = build_scene(face_landmarks_720p, room)
    switch = (np.abs(room - wall_image()).max(axis=2) > 0.2) & ~scene.strands
    assert switch.sum() > 50
    out = run_effect(scene, frame=scene.clean, flyaway=1.0)
    assert change(scene.clean, out)[switch].mean() < 0.02


# ---------------------------------------------------------------------------
# Speed: the same numbers, computed with fewer temporary images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(40, 30), (64,), (1, 50)])
def test_smoothstep_matches_the_formula_and_leaves_its_input(shape: tuple[int, ...]) -> None:
    values = np.random.default_rng(6).uniform(-0.5, 1.5, shape).astype(np.float32)
    before = values.copy()
    result = hairshape_module._smoothstep_t(values)
    t = np.clip(values, 0.0, 1.0)
    np.testing.assert_array_equal(result, t * t * (3.0 - 2.0 * t))
    np.testing.assert_array_equal(values, before)
    assert result.dtype == np.float32
    ranged = hairshape_module._smoothstep(values, 0.2, 0.7)
    t = np.clip((values - np.float32(0.2)) * np.float32(1.0 / (0.7 - 0.2)), 0.0, 1.0)
    np.testing.assert_array_equal(ranged, t * t * (3.0 - 2.0 * t))
    np.testing.assert_array_equal(values, before)


def test_clip_leaves_its_input() -> None:
    values = np.array([[-1.0, 0.5, 2.0]], np.float32)
    before = values.copy()
    np.testing.assert_array_equal(hairshape_module._clip01(values), [[0.0, 0.5, 1.0]])
    np.testing.assert_array_equal(values, before)


def test_above_matches_the_comparison() -> None:
    values = np.random.default_rng(7).uniform(0.0, 1.0, (50, 60)).astype(np.float32)
    level = float(values[3, 4])  # an exact tie counts as not above
    np.testing.assert_array_equal(hairshape_module._above(values, level), (values > level).astype(np.float32))


def _keep_attached_as_written_first(alpha, band, distance, head, reach, previous):  # type: ignore[no-untyped-def]
    """The straightforward form of ``_keep_attached``, as a reference for the fast one."""
    m = hairshape_module
    candidates = (alpha > m._CANDIDATE).astype(np.uint8)
    if not candidates.any():
        return np.zeros_like(alpha)
    bridged = cv2.dilate(candidates, m._RECT_3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    d = head.grid_to_work(distance)
    attached = np.zeros(count, bool)
    attached[labels[d <= m._ATTACH]] = True
    limit = m._LEAVES_LENGTH * head.work_scale * m._LEAVES_WIDTH
    beyond = np.bincount(labels[d > reach + m._LONG_STRUCTURE], minlength=count).astype(np.float32)
    for border in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        beyond[border] = np.float32(limit * m._LEAVES_HOLD)
    area = stats[:, cv2.CC_STAT_AREA]
    held = np.bincount(labels[previous > 0.5], minlength=count) > m._HELD_SHARE * area
    leaves = beyond >= np.where(held, limit * m._LEAVES_HOLD, limit)
    keep = attached & ~leaves & (area >= m._MIN_AREA * head.work_scale * head.work_scale)
    keep[0] = False
    return m._finish_alpha(alpha * keep[labels].astype(np.float32)) * band


@pytest.mark.parametrize("seed", range(6))
def test_keep_attached_matches_its_plain_form(face_landmarks_720p: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    head = hairshape_module._locate_head(face_landmarks_720p.astype(np.float32), (720, 1280, 3), None)
    assert head is not None
    w, h = head.work_size
    grid_w, grid_h = head.grid_size
    alpha = np.zeros((h, w), np.float32)
    for _ in range(40):  # blobs and lines of all sizes, some touching the border
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        end = (x + int(rng.integers(-40, 40)), y + int(rng.integers(-40, 40)))
        cv2.line(alpha, (x, y), end, float(rng.uniform(0.1, 1.0)), int(rng.integers(1, 4)))
    distance = cv2.resize(rng.uniform(-1.0, 2.0, (6, 6)).astype(np.float32), (grid_w, grid_h))
    band = rng.uniform(0.0, 1.0, (h, w)).astype(np.float32)
    previous = cv2.GaussianBlur(alpha, (0, 0), 2.0) * float(rng.uniform(0.0, 3.0))
    alpha_before = alpha.copy()
    fast = hairshape_module._keep_attached(alpha, band, distance, head, 0.5, previous)
    np.testing.assert_array_equal(alpha, alpha_before)
    np.testing.assert_array_equal(fast, _keep_attached_as_written_first(alpha, band, distance, head, 0.5, previous))


def test_keep_attached_without_candidates(face_landmarks_720p: np.ndarray) -> None:
    head = hairshape_module._locate_head(face_landmarks_720p.astype(np.float32), (720, 1280, 3), None)
    assert head is not None
    w, h = head.work_size
    zero = np.zeros((h, w), np.float32)
    distance = np.zeros(head.grid_size[::-1], np.float32)
    assert not hairshape_module._keep_attached(zero, zero, distance, head, 0.5, zero).any()


def _head_at(landmarks: np.ndarray, shift: tuple[float, float] = (0.0, 0.0), zoom: float = 1.0):  # type: ignore[no-untyped-def]
    center = landmarks.mean(axis=0)
    moved = (landmarks - center) * np.float32(zoom) + center + np.float32(shift)
    head = hairshape_module._locate_head(moved.astype(np.float32), (720, 1280, 3), None)
    assert head is not None
    return head


def test_strand_memory_projects_once_per_frame(
    face_landmarks_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = hairshape_module._StrandMemory()
    head = _head_at(face_landmarks_720p)
    shape = head.work_size[::-1]
    mask = memory.update(np.full(shape, 0.5, np.float32), head)
    assert mask.max() == pytest.approx(0.3)

    next_head = _head_at(face_landmarks_720p, (6.0, 0.0))
    calls: list[int] = []
    real = hairshape_module._StrandMemory._project

    def counting(*args: object) -> np.ndarray:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(hairshape_module._StrandMemory, "_project", counting)
    shape = next_head.work_size[::-1]
    projected = memory.projected(next_head, shape)
    assert memory.projected(next_head, shape) is projected
    updated = memory.update(np.zeros(shape, np.float32), next_head)
    assert len(calls) == 1
    assert memory.projected(next_head, shape) is updated  # the new mask, not the projection of the old one


def test_strand_memory_follows_a_work_area_that_changes_size(face_landmarks_720p: np.ndarray) -> None:
    memory = hairshape_module._StrandMemory()
    for zoom, shift in ((1.0, (0.0, 0.0)), (1.08, (2.0, 0.0)), (0.9, (0.0, 3.0)), (1.0, (0.0, 0.0))):
        head = _head_at(face_landmarks_720p, shift, zoom)
        shape = head.work_size[::-1]
        before = memory.projected(head, shape)
        assert before.shape == shape
        mask = memory.update(np.full(shape, 0.5, np.float32), head)
        assert mask.shape == shape and mask.dtype == np.float32
        assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


@pytest.mark.parametrize("plate", [False, True])
def test_the_work_area_may_change_size_between_frames(scene: Scene, plate: bool) -> None:
    """A head coming closer: the working images change size and everything follows."""
    effect = HairShapeEffect()
    settings = HairShapeSettings(flyaway=1.0, volume_top=0.3, width_sides=-0.3, use_background_plate=True)
    center = scene.landmarks.mean(axis=0)
    for zoom in (1.0, 1.05, 1.15, 1.3, 1.3, 1.0, 0.8, 1.0):
        landmarks = ((scene.landmarks - center) * np.float32(zoom) + center).astype(np.float32)
        ctx = dataclasses.replace(scene.ctx(plate=plate), landmarks=landmarks)
        out = effect.apply(scene.frame.copy(), ctx, settings)
        assert out.shape == scene.frame.shape and np.isfinite(out).all()


def test_the_room_is_only_measured_for_a_warp_and_then_at_once(scene: Scene) -> None:
    """Sliders moved at runtime: the same result as if the warp had been on from the start of it."""
    plain = HairShapeEffect()
    ctx = scene.ctx
    off = HairShapeSettings(**{**OFF, "flyaway": 0.3})
    on = HairShapeSettings(**{**OFF, "flyaway": 0.3, "volume_top": 0.5})
    for _ in range(3):
        plain.apply(scene.frame.copy(), ctx(), off)
    assert plain._room_busy is None
    out = plain.apply(scene.frame.copy(), ctx(), on)
    assert plain._room_busy is not None and not np.array_equal(out, scene.frame)
    turned_off = plain.apply(scene.frame.copy(), ctx(), off)
    assert turned_off.shape == scene.frame.shape


def test_no_flyaway_slider_means_no_work_at_all(scene: Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object) -> None:
        raise AssertionError("must not run")

    monkeypatch.setattr(hairshape_module, "_silhouette", forbidden)
    effect = HairShapeEffect()
    frame = scene.frame.copy()
    assert effect.apply(frame, scene.ctx(), HairShapeSettings(**OFF)) is frame


def test_a_mask_without_any_hair_causes_no_overflow(scene: Scene) -> None:
    """In a deep bow the model may find no hair at all."""
    head = hairshape_module._locate_head(scene.landmarks, scene.frame.shape, None)
    bald = dataclasses.replace(scene, hair_mask=np.zeros_like(scene.hair_mask))
    with np.errstate(over="raise", invalid="raise"):
        outline = hairshape_module._silhouette(head, bald.ctx())
    assert np.isfinite(outline.distance).all()
    assert outline.near_hair.max() == 0.0

"""Plan A: build a preset from a photo of the user wearing her real makeup.

``analyze_makeup(makeup_bgr, bare_bgr)`` measures a made-up face and returns
settings that make the app's own effects reproduce it. The real effects are
the forward model (they render the candidates), so the result stays right
when an effect is retuned.

Steps
    1. Each photo: stable landmarks (the Face Landmarker's first frames are
       stretched, so a face-centered square crop is fed as a short static
       video until the points stop moving), a check for one usable face, and
       the segmentation masks. Everything then works on a crop whose face
       scale is at most ``_WORK_SCALE``, which makes phone and webcam photos
       comparable and averages away JPEG noise.
    2. Colour: the preset will be applied to a bare live face, so the bare
       photo's light is the reference. The makeup photo is brought to it with
       per-channel gains, and a tone curve when the lightness range allows it,
       from paired pixels makeup does not touch: the person outside the face,
       then the room (then the neck, then the forehead, each less trusted). A
       near-neutral room also gives a partial gray-world correction of both. A
       colored room (a green screen) is left alone: nothing tells a cast from
       the wall's own color.
    3. Geometry: the bare photo is warped onto the makeup photo's landmarks
       with a thin-plate spline, so small pose differences shrink to a small
       residual. Without a bare photo, ``_approximate_bare`` takes the makeup
       off instead: the lips go to a natural-lip color derived from the skin,
       a chroma bump on the cheeks and a dark or tinted lid go, and the lash
       line is inpainted. Differences within the natural variation (measured
       on the test faces) count as no makeup, so a bare face stays bare. The
       same happens in the areas where the two photos are not comparable:
       squinting changes the lids, and a smile or a turned head changes the
       light on the skin, more than makeup does.
    4. Fit: each section is fitted against that bare image with the earlier
       sections already applied, in pipeline order, by rendering its effect.
    5. Plausibility: what no real product does is not makeup. Foundation evens the
       skin's color out, a blush keeps its red, a brow pencil stays on the brow,
       so a change that does not fit (another light, bangs or glasses in one photo
       only) leaves the section at the base preset's values, with a note.

What a photo can and cannot tell
    The effects tint relative to the skin, so opacity and color trade off:
    for blush and eyeshadow a pale color at high opacity and a deep color at
    low opacity give the same pixels. The fit recovers the visible change,
    then picks the member of that family whose color looks like a real
    product (``_BLUSH_CHROMA``, ``_SHADOW_LIGHTNESS``, ``_BROW_LIGHTNESS``), so
    only the appearance is meant to round-trip exactly. For lips the color
    variation left inside the lips tells the opacity apart (only with a bare
    photo).

    Contour and tear bags are only measured with a bare photo: a shadow on a
    single photo may be makeup, bone structure or light. Teeth, reshape, hair
    and hairshape are never measured and keep the base preset's values.

    A section that cannot be measured (hidden, blown out, or a change no product
    makes) keeps the base preset's values, with confidence 0 and a note. A section
    measured as absent gets opacity 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from . import blend, regions
from .config import BUILTIN_PRESETS_DIR, SECTION_LABELS, MakeupSettings, hex_to_bgr
from .effects.base import Effect, FrameContext
from .models import FACE_LANDMARKER, SELFIE_MULTICLASS, ModelDownloadError, ensure_model
from .presets import (
    DEFAULT_PRESET_NAME,
    PRESET_SUFFIX,
    PresetError,
    PresetInfo,
    load_preset,
    sanitize_preset_name,
    save_preset,
)
from .segmentation import BACKGROUND, BODY_SKIN, CLOTHES, FACE_SKIN, HAIR, OTHERS, MediaPipeSegmenter, model_input
from .tracking import LandmarkModel, MediaPipeFaceLandmarker, search_windows, window_rgb


class AnalysisError(Exception):
    """The photo cannot be analyzed. The message is Japanese and shown to the user."""


@dataclass
class AnalysisResult:
    """What ``analyze_makeup`` found.

    ``settings``: the base preset with the measured sections overwritten.
    ``measurements``: raw numbers per section, for debugging (JSON-friendly).
    ``confidence``: 0..1 per measured section (0 = not measured, base values).
    ``notes``: Japanese remarks for the user.
    """

    settings: MakeupSettings
    measurements: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "measurements": _jsonable(self.measurements),
            "confidence": {k: round(float(v), 3) for k, v in self.confidence.items()},
            "notes": list(self.notes),
        }


# Sections this module fits, in pipeline order.
MEASURED_SECTIONS = ("skin", "contour", "brows", "eyeshadow", "tearbag", "eyeliner", "blush", "lips")

# All lengths below are fractions of the face scale (inter-ocular distance)
# unless noted otherwise.

# ---- photo preparation ------------------------------------------------------

_MAX_INPUT_SIDE = 2400  # longer photos are downscaled first
_WORK_SCALE = 130.0  # face scale (px) of the working crop, at most
_MIN_FACE_SCALE = 28.0  # below this (px): "face too small"
_LOW_RES_SCALE = 60.0  # below this (px): thin features become estimates
_CROP_MARGIN = 1.1
_STABLE_FRAMES = 40
_STABLE_RUN = 3  # frames in a row that must move less than _STABLE_MOVE x face size
_STABLE_MOVE = 0.003
_LANDMARK_INPUT = 512
_LANDMARK_CROP = 2.0  # crop side / face size
_SEGMENT_INPUT = 384
_OTHER_FACE_MIN_SIZE = 0.4  # a second face counts when at least this x the main one
_MAX_YAW = 0.55  # (distance difference / sum) of the eyes to the nose: about 45 degrees
_WARN_YAW = 0.25
_LARGE_MISMATCH = 0.03  # landmarks of the two photos differ by this (x face scale): warn
_MAX_EYE_STATE_CHANGE = 0.25
_MAX_MOUTH_STATE_CHANGE = 0.1
_MAX_FACE_MISMATCH = 0.04  # beyond this the light on the face differs more than makeup does

# ---- colour -----------------------------------------------------------------

_ROOM_CONFIDENCE = 0.7  # background probability that counts as room
_MIN_ROOM_SHARE = 0.03
_SAME_ROOM_SPREAD = 0.06  # MAD of the log ratio of paired room pixels: same room
_OTHER_ROOM_SPREAD = 0.14  # ... different room (the ratio tells nothing)
_MAX_ROOM_SKIN_DISAGREEMENT = 0.2  # log gain between the room and the skin
_GAIN_RANGE = (0.1, 10.0)
_LARGE_GAIN = (0.5, 2.0)  # the two exposures differ this much: note it, and trust the colors less
_CURVE_MIN_RANGE = 0.7  # log range of a channel's paired pixels needed to tell a tone curve from a gain
_CURVE_POWER_RANGE = (0.6, 1.7)  # steeper or flatter is not the same camera: the ratio is not a curve
_CURVE_MIN_POWER_CHANGE = 0.03
_CURVE_MIN_IMPROVEMENT = 0.015  # the curve must explain the skin this much better (log) than a gain
_CURVE_MIN_SPREAD = 0.015  # log; pairs tighter than this around a gain need no curve
_CURVE_PROOF = 0.6  # spread around the curve / around the gain that proves a curve
_MIN_GAIN_CHANGE = 0.004
_CLIPPED_LEVEL = 0.985
_MAX_CLIPPED_SKIN = 0.8  # share of the stable skin; more: the skin's color cannot be measured
_MIN_SKIN_CHROMA = 3.0  # less: a black-and-white photo, or a blown-out one above this lightness
_WHITE_SKIN_LIGHTNESS = 92.0
_SKIN_HUE_RANGE = (-20.0, 115.0)  # Lab; outside: a negative or a strong color filter
_GRAY_WORLD_MAX_CHROMA = 16.0  # a more colorful room is taken for a colored wall
_GRAY_WORLD_ADAPTATION = 0.7
_GRAY_WORLD_MAX_GAIN = 0.2

# ---- natural baselines (no bare photo) --------------------------------------
# Measured on the female test faces (11 women, 2 frames each): see
# docs/plan-a-photo-preset.md. Lab units.

_LIP_FROM_SKIN_L = -20.0
_LIP_A_SLOPE = (0.35, 62.0, 1.0)  # a* above the skin's: slope x (L - pivot) + offset
_LIP_A_MIN_OFFSET = -3.0
_LIP_B_RATIO = 0.78
# Lips: distance from the natural color in a*/b*, plus how much darker than
# natural beyond a free margin (lighter ones do not count: smiles lighten lips a lot).
# Bare lips of the test faces reached 13.8, all of it in the smile's lighter, warmer
# direction, with their hue within 14 degrees of the natural one. A lipstick is pinker:
# what counts as one is (distance, hue this much below natural), see ``_looks_like_lipstick``.
_LIPSTICK_RULES = ((16.0, -360.0), (10.0, 11.0), (7.0, 15.0))
_NATURAL_LIP_MARGIN = 11.0  # the opacity ramps from here to _NATURAL_LIP_FULL
_NATURAL_LIP_FULL = 19.0
_NATURAL_LIP_DARK_FREE = 6.0
_NO_BARE_LIP_OPACITY = 0.85
_MIN_NO_BARE_LIP_OPACITY = 0.6
# Highlight above the median lightness (97th percentile) that separates the
# finishes when there is no bare lip to compare with.
_MATTE_HIGHLIGHT = 9.0
_GLOSS_HIGHLIGHT = 24.0
# Cheeks: amplitude of the blush shape in log(R/G) over a plane fitted per cheek.
# Bare test faces reached 0.1 (the face's redness is not uniform), so a light
# blush cannot be told from a natural flush on one photo: that is what the bare
# photo is for. Near the margin only a faint blush comes out (KEEP is left on the face).
_NATURAL_CHEEK_MARGIN = 0.12
_NATURAL_CHEEK_KEEP = 0.1
_NATURAL_LID = (-3.0, 0.5, -1.0)  # Lab of a bare lid minus the brow bone
# Delta-E 2000 from that still read as natural: bare test faces reached 8.6 with a relaxed
# face, 18.7 with a smile (the cheeks push up, the lid sinks into shadow). A medium
# shadow changes the lid less than that, so one photo only finds strong ones.
_NATURAL_LID_MARGIN = 9.0
_EXPRESSION_LID_MARGIN = 19.0
_SMILE_WIDTH = (0.62, 0.72)  # mouth width / inter-ocular distance: relaxed to smiling
_NARROWED_EYES = (0.22, 0.30)  # eye opening / width: narrowed to relaxed
_MAX_LID_DARKENING = 10.0  # Lab L, the most a single photo may ascribe to eyeshadow
_SQUINT = (0.14, 0.24)  # eye opening / width: squinting lids look dark, so they are not trusted
# Liner on one photo: a natural lash line looks just like a thin liner, and only
# a dark stroke past the outer corner (darkness relative to the skin around
# it) tells them apart. Bare test faces reached 0.45.
_LINER_WING_DEPTH = (0.5, 0.7)
_NO_BARE_LINER_OPACITY = 0.8
_NATURAL_LINER_THICKNESS = 0.2  # what the lash line itself adds to the fitted thickness
_NO_BARE_LINER_THICKNESS = (0.2, 0.8)
# Brow darkness (1 - luma / skin luma) that reads as filled, no bare photo.
# Bare test brows reached 0.44, the same brows with 'polished' 0.5.
_BROW_CONTRAST = (0.42, 0.62)
_NO_BARE_BROW_MAX = 0.5

# ---- thresholds with a bare photo -------------------------------------------

_GAIN_MASK_LEVEL = 0.3
_GAIN_TRIM = (10.0, 90.0)
_MIN_EXPLAINED = 0.15  # a shadow fit must explain this share of the change in its zone
_MIN_BROW_EXPLAINED = 0.5  # the same for a brow, where the mask is sharp: true brows reached 0.75
_POSE_NOISE = 60.0  # thresholds grow by this x the photos' landmark mismatch (x face scale)
# Lips of two bare photos stayed below these (with the noise factors), lipsticks above.
_MIN_LIP_CHANGE = 4.5  # delta-E 2000
_MIN_LIP_SHINE = 5.0  # Lab L of the 97th percentile above the median
_MIN_LIP_CHROMA = 2.5  # Lab a/b
_MOUTH_NOISE = 6.0  # lip thresholds also grow by this x how much the mouth differs
_MOUTH_FREE = 0.02
_ALWAYS_LIPSTICK_CHROMA = 9.0  # this much is lipstick however much the mouth moved (bare lips: 7.2)
_MAX_CLIPPED_LIPS = 0.5  # share of the lips, beyond the bare photo's; more: their color cannot be measured
_CONCEALER_ZONE_END = 0.55  # along the eye: 0 inner corner, 1 outer corner
_MIN_BLUSH_AMPLITUDE = 0.9  # Lab a/b units at full mask
_MIN_SHADOW_CHANGE = 3.0  # Lab units at full mask
_MIN_SHADOW_SCALE = 0.5  # the lash band may reduce the fitted shadow to this share at most
_MAX_SHADOW_LIGHTENING = 0.03  # a lid much lighter than bare is a different light or expression
_MIN_BROW_CHANGE = 0.03  # relative darkening
_MIN_LINER_OPACITY = 0.3
_MIN_LINER_CONTRAST = 0.05  # how much darker than around it the stroke got, relative
_MIN_LINER_EXPLAINED = 0.1  # share of the change around the lash line that the stroke must explain
_MIN_TONE_TRUST = 0.4  # how far the photos' colors must be comparable to measure the foundation
_MIN_TONE_CHANGE = 1.0
_MIN_LIGHTNESS_CHANGE = 0.5
# A foundation evens the skin's own color out, at the strength its shift implies. A shift
# with no evening (below this share of that strength) is a change of light.
_TONE_EXPLAINED = 0.5
# Shades of real foundations: a warm hue, not too colorful (Lab).
_TONE_HUE_RANGE = (30.0, 90.0)
_TONE_MAX_CHROMA = 40.0
_MIN_FOREHEAD_PIXELS = 60
_MIN_TONE_STRENGTH = 0.05
_MAX_TONE_STRENGTH = 0.6
_MAX_BRIGHTNESS = 0.1  # more than this between two photos is the light, not foundation
_TONE_ABOVE_SKIN = 5.0  # Lab L of the fitted shade; the slider moves L only slightly anyway
_MIN_TONE_SPREAD = 0.8  # a*/b* spread of the skin below this tells nothing about the strength
_MAX_TONE_DISTANCE = 22.0  # how far from the skin a foundation shade may be, Lab a/b
# From skin.py.
_TONE_COVERAGE = 0.6
_BRIGHTNESS_GAIN = 1.5
_MIN_CONTOUR_WEIGHT = 0.06
_MIN_CONTOUR_EXPLAINED = 0.33  # share of the color change; two bare photos stayed below
_MIN_ENERGY_CHANGE = 0.05

# ---- what real products can do (beyond this a measurement is not makeup) -----

# Blush: the red channel of a real blush color stays above this (a change darker than that
# is a shadow, a hand or another light), and its color is neither dark nor gray.
_BLUSH_MIN_RED = 0.5
_BLUSH_MIN_LIGHTNESS = 35.0
_BLUSH_MIN_CHROMA = 12.0
_MIN_VISIBLE_CHEEKS = 0.5  # share of the cheeks not blown out or hidden, or they are not measured
# Eye makeup: hue ranges in Lab (degrees), and the deepest color and largest darkening
# that are still a product rather than something in front of the face.
_BROW_HUE_RANGE = (15.0, 85.0)
_BROW_MAX_CHROMA = 32.0
_MAX_BROW_DARKENING = 0.86
_MAX_BROW_DARKENING_APART = 0.45  # the same when the two photos differ in expression or pose
_BROW_RING_SHIFT = 0.24  # the band above the brow (x face scale): a brow pencil does not reach it
_BROW_RING_DARKENING = 0.12  # relative: this much darker there is hair or a frame over the brow
_BROW_RING_LUMA = 0.45  # luma of that band / skin; below it is hair on one photo too (bare faces stayed above 0.6)
_SHADOW_HUE_RANGE = (-55.0, 85.0)
_SHADOW_MAX_CHROMA = 48.0
_SHADOW_MIN_LIGHTNESS = 16.0
_MIN_EYE_OPENING_PIXELS = 25
_MIN_SCLERA_CONTRAST = 0.55  # brightest part of the eye opening / skin; less: dark glasses
# Below the eyes makeup leaves no dark lines: this dark (x the skin's luma), this much of
# the band means the rim of a pair of glasses.
_RIM_DARKNESS = 0.55
_RIM_SHARE = 0.08  # big smiles reached 0.05 (creases under the eyes), rims 0.13 and more
_MIN_LINER_INK = 0.0095  # opacity x stroke width (x face scale): fainter is not a stroke
_MIN_LINER_THICKNESS = 0.15  # the slider: a fit below this is narrower than the photo resolves

# ---- canonical colours (see the module docstring) ---------------------------

_BLUSH_CHROMA = 43.0
_BLUSH_HUE_RANGE = (0.0, 65.0)  # pink to apricot; anything else is noise on a blush
_SHADOW_LIGHTNESS = 47.0
_BROW_LIGHTNESS = 31.0
_DEFAULT_LIP_OPACITY = 0.7
# Pull toward the default opacity (Lab units per unit of opacity): the spread is noisy on
# small lips, and a higher opacity depends less on the bare lips anyway.
_LIP_OPACITY_PRIOR = 8.0
_MIN_SPREAD_PIXELS = 40

_OPACITY_STEPS = np.linspace(0.02, 1.0, 50)
_TINT_MISS_WEIGHT = 2000.0  # 0.01 of relative change counts like 20 units of the criterion
_LIP_OPACITIES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_LIP_LIGHTNESS_FOLLOW = 0.7  # lips.py: share of the lightness gap an opaque lipstick closes
_MIN_LIP_SPREAD = 1.0  # bare lips' a*/b* spread below this tells nothing about the opacity

# ---- zones (mesh indices, subject's right / left mirrored) ------------------

_CHIN_BOTTOM = 152
_FOREHEAD_ZONE = [108, 151, 337, 336, 9, 107]
_PERIORAL_ZONES = ([203, 206, 92, 165], [423, 426, 322, 391])
_WARP_POINTS = sorted(
    set(regions.FACE_OVAL)
    | set(regions.LIPS_OUTER)
    | set(regions.LIPS_INNER)
    | set(regions.RIGHT_EYE)
    | set(regions.LEFT_EYE)
    | set(regions.RIGHT_BROW)
    | set(regions.LEFT_BROW)
    | set(regions.NOSE_BRIDGE)
    | set(regions.RIGHT_CHEEK_CENTER)
    | set(regions.LEFT_CHEEK_CENTER)
    | set(range(0, 468, 4))
)


# --------------------------------------------------------------------------
# Small helpers.
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), 4)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _ramp(value: float, low: float, high: float) -> float:
    return _clip((value - low) / max(high - low, 1e-6), 0.0, 1.0)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return image
    return cv2.GaussianBlur(image, (0, 0), float(sigma), borderType=cv2.BORDER_REPLICATE)


def _robust_std(values: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def trimmed_median(pixels: np.ndarray, trim: tuple[float, float] = (15.0, 85.0)) -> np.ndarray | None:
    """Median of (N, 3) Lab pixels between two lightness percentiles.

    The trim drops speculars and teeth at the top, shadows and hairs at the
    bottom. None without pixels.
    """
    if len(pixels) == 0:
        return None
    low, high = np.percentile(pixels[:, 0], trim)
    kept = pixels[(pixels[:, 0] >= low) & (pixels[:, 0] <= high)]
    return np.median(kept if len(kept) else pixels, axis=0).astype(np.float64)


def delta_e00(lab1: Sequence[float] | np.ndarray, lab2: Sequence[float] | np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference of Lab arrays (..., 3)."""
    lab1, lab2 = np.asarray(lab1, np.float64), np.asarray(lab2, np.float64)
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c_mean = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_mean**7 / (c_mean**7 + 25.0**7)))
    a1p, a2p = a1 * (1.0 + g), a2 * (1.0 + g)
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(c1p * c2p == 0.0, 0.0, dh)
    dh_big = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2.0)
    l_mean = (l1 + l2) / 2.0
    c_mean_p = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    h_mean = np.where(
        np.abs(h1p - h2p) > 180.0,
        np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        h_sum / 2.0,
    )
    h_mean = np.where(c1p * c2p == 0.0, h_sum, h_mean)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_mean - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_mean))
        + 0.32 * np.cos(np.radians(3.0 * h_mean + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_mean - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_mean - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt(c_mean_p**7 / (c_mean_p**7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_mean - 50.0) ** 2 / np.sqrt(20.0 + (l_mean - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_mean_p
    s_h = 1.0 + 0.015 * c_mean_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c
    return np.sqrt(
        (dl / s_l) ** 2 + (dc / s_c) ** 2 + (dh_big / s_h) ** 2 + r_t * (dc / s_c) * (dh_big / s_h)
    )


def _lab_of_bgr(color: Sequence[float] | np.ndarray) -> np.ndarray:
    return blend.bgr_to_lab(np.asarray(color, np.float32).reshape(1, 1, 3))[0, 0].astype(np.float64)


def _hex_of_bgr(color: Sequence[float] | np.ndarray) -> str:
    b, g, r = (int(round(_clip(float(v), 0.0, 1.0) * 255.0)) for v in color)
    return f"#{r:02X}{g:02X}{b:02X}"


def _hex_of_lab(lab: Sequence[float] | np.ndarray) -> str:
    return _hex_of_bgr(blend.lab_to_bgr(np.asarray(lab, np.float32).reshape(1, 1, 3))[0, 0])


# --------------------------------------------------------------------------
# Models.
# --------------------------------------------------------------------------


class StillModels(Protocol):
    """The models a still photo needs (tests inject fakes)."""

    def video_landmarker(self) -> LandmarkModel: ...

    def faces(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Every face in ``rgb``: a list of (478, 3) normalized landmark arrays."""

    def segment(self, rgb: np.ndarray) -> Sequence[np.ndarray]:
        """Per-class confidence masks (``segmentation`` class order) at the size of ``rgb``."""

    def close(self) -> None: ...


class MediaPipeStillModels:
    """MediaPipe models for still photos (see ``StillModels``)."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        directory = None if models_dir is None else Path(models_dir)
        self._landmarker_path = ensure_model(FACE_LANDMARKER, directory=directory)
        self._segmenter = MediaPipeSegmenter(ensure_model(SELFIE_MULTICLASS, directory=directory))
        self._segment_ts = 0
        self._face_finder: Any = None

    def video_landmarker(self) -> LandmarkModel:
        return MediaPipeFaceLandmarker(self._landmarker_path)

    def faces(self, rgb: np.ndarray) -> list[np.ndarray]:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if self._face_finder is None:
            # Bytes instead of a path: MediaPipe cannot open non-ASCII paths on Windows.
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=Path(self._landmarker_path).read_bytes()),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.5,
            )
            self._face_finder = vision.FaceLandmarker.create_from_options(options)
        result = self._face_finder.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return [np.array([(p.x, p.y, p.z) for p in face], np.float32) for face in result.face_landmarks]

    def segment(self, rgb: np.ndarray) -> Sequence[np.ndarray]:
        self._segment_ts += 33
        return self._segmenter.segment(rgb, self._segment_ts)

    def close(self) -> None:
        self._segmenter.close()
        if self._face_finder is not None:
            self._face_finder.close()


# --------------------------------------------------------------------------
# Preparing one photo.
# --------------------------------------------------------------------------


@dataclass
class _Photo:
    """One photo, ready to measure. All arrays are in working-crop coordinates."""

    image: np.ndarray  # float32 BGR [0, 1]
    landmarks: np.ndarray  # (478, 2) float32
    scale: float  # face scale of the crop, px
    source_scale: float  # face scale of the original photo, px
    hair: np.ndarray
    face_skin: np.ndarray
    body_skin: np.ndarray
    room: np.ndarray  # low-resolution full frame, float32 BGR
    room_mask: np.ndarray  # background confidence of ``room``
    yaw: float
    eyes_open: float  # opening height / eye width, smaller of the two eyes
    others: np.ndarray | None = None  # accessories in the crop (glasses, a mask)
    room_person: np.ndarray | None = None  # hair, clothes and body skin confidence of ``room``


def _validated_image(image: Any, label: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        raise AnalysisError(f"{label}を読み取れませんでした（画像のデータではありません）。")
    if image.dtype != np.uint8:
        raise AnalysisError(f"{label}の形式に対応していません（8 ビットの画像を指定してください）。")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise AnalysisError(f"{label}の形式に対応していません（カラー画像を指定してください）。")
    return image


def _face_box(points: np.ndarray) -> tuple[np.ndarray, float]:
    lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return (lo + hi) / 2.0, float((hi - lo).max())


def _locate_face(image: np.ndarray, models: StillModels) -> np.ndarray | None:
    """Rough landmarks of the face, searching square windows over the frame."""
    landmarker = models.video_landmarker()
    try:
        height, width = image.shape[:2]
        ts = 0
        for _ in range(2):
            for window in search_windows(height, width):
                ts += 33
                raw = landmarker.detect(window_rgb(image, window, _LANDMARK_INPUT), ts)
                if raw is not None:
                    return raw[:, :2] * window.size + (window.x0, window.y0)
    finally:
        landmarker.close()
    return None


def _stable_landmarks(image: np.ndarray, models: StillModels) -> np.ndarray | None:
    """Landmarks of a still (full-frame px, float32), or None without a face.

    The first results of the Face Landmarker are stretched, so a square crop
    around the face is fed as a static video until the points stop moving.
    """
    rough = _locate_face(image, models)
    if rough is None:
        return None
    center, size = _face_box(rough)
    side = max(int(round(size * _LANDMARK_CROP)), 16)
    x0 = int(round(center[0] - side / 2.0))
    y0 = int(round(center[1] - side / 2.0))
    pad = max(0, -x0, -y0, x0 + side - image.shape[1], y0 + side - image.shape[0])
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else image
    crop = padded[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side]
    interpolation = cv2.INTER_AREA if side > _LANDMARK_INPUT else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (_LANDMARK_INPUT, _LANDMARK_INPUT), interpolation=interpolation)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    landmarker = models.video_landmarker()
    history: list[np.ndarray] = []
    run = 0
    try:
        for index in range(_STABLE_FRAMES):
            raw = landmarker.detect(rgb, (index + 1) * 33)
            if raw is None:
                continue
            points = raw[:, :2] * side + (x0, y0)
            if history:
                moved = float(np.linalg.norm(points - history[-1], axis=1).max())
                run = run + 1 if moved < _STABLE_MOVE * size else 0
            history.append(points)
            if run >= _STABLE_RUN:
                break
    finally:
        landmarker.close()
    if len(history) < 4:
        return None
    return np.mean(history[-3:], axis=0).astype(np.float32)


def _count_other_faces(image: np.ndarray, landmarks: np.ndarray, models: StillModels) -> int:
    """How many faces besides the main one are big enough to matter."""
    center, size = _face_box(landmarks)
    others: list[np.ndarray] = []
    height, width = image.shape[:2]
    for window in search_windows(height, width):
        for face in models.faces(window_rgb(image, window, _LANDMARK_INPUT)):
            points = face[:, :2] * window.size + (window.x0, window.y0)
            other_center, other_size = _face_box(points)
            if other_size < _OTHER_FACE_MIN_SIZE * size:
                continue
            if float(np.linalg.norm(other_center - center)) < 0.6 * size:
                continue
            if all(float(np.linalg.norm(other_center - seen)) >= 0.6 * size for seen in others):
                others.append(other_center)
    return len(others)


def _yaw_of(landmarks: np.ndarray) -> float:
    """0 for a frontal face, toward 1 for a profile."""
    right = float(np.linalg.norm(landmarks[regions.RIGHT_EYE_OUTER_CORNER] - landmarks[regions.NOSE_TIP]))
    left = float(np.linalg.norm(landmarks[regions.LEFT_EYE_OUTER_CORNER] - landmarks[regions.NOSE_TIP]))
    return abs(right - left) / max(right + left, 1e-6)


def _eyes_open(landmarks: np.ndarray) -> float:
    ratios = []
    for side in regions.SIDES:
        s = regions.side_indices(side)
        width = float(np.linalg.norm(landmarks[s.eye_upper[0]] - landmarks[s.eye_upper[-1]]))
        height = float(np.linalg.norm(landmarks[s.eye_upper[4]] - landmarks[s.eye_lower[4]]))
        ratios.append(height / max(width, 1e-6))
    return min(ratios)


def _mouth_shape(landmarks: np.ndarray) -> tuple[float, float]:
    """(mouth width, opening height), both x the inter-ocular distance."""
    iod = float(np.linalg.norm(landmarks[regions.LEFT_EYE_OUTER_CORNER] - landmarks[regions.RIGHT_EYE_OUTER_CORNER]))
    width = float(np.linalg.norm(landmarks[regions.MOUTH_LEFT_CORNER] - landmarks[regions.MOUTH_RIGHT_CORNER]))
    opening = float(np.linalg.norm(landmarks[regions.LIPS_INNER_TOP] - landmarks[regions.LIPS_INNER_BOTTOM]))
    return width / max(iod, 1e-6), opening / max(iod, 1e-6)


def _prepare(image_bgr: Any, models: StillModels, label: str) -> _Photo:
    """Find the one face of a photo and build its working crop."""
    image = _validated_image(image_bgr, label)
    long_side = max(image.shape[:2])
    if long_side > _MAX_INPUT_SIDE:
        ratio = _MAX_INPUT_SIDE / long_side
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    landmarks = _stable_landmarks(image, models)
    if landmarks is None:
        raise AnalysisError(
            f"{label}から顔を見つけられませんでした。顔が小さすぎる、暗い、髪や手で隠れているなどの可能性があります。"
            "顔全体が大きく明るく写っている写真を選んでください。"
        )
    if _count_other_faces(image, landmarks, models) > 0:
        raise AnalysisError(f"{label}に複数の顔が写っています。ご本人だけが写っている写真を選んでください。")
    source_scale = regions.face_scale(landmarks)
    if source_scale < _MIN_FACE_SCALE:
        raise AnalysisError(
            f"{label}の顔が小さすぎます（目の間隔が {source_scale:.0f} ピクセル）。顔を大きく写すか、高い解像度の写真を選んでください。"
        )
    yaw = _yaw_of(landmarks)
    if yaw > _MAX_YAW:
        raise AnalysisError(f"{label}の顔が横を向きすぎています。正面に近い向きの写真を選んでください。")
    factor = min(1.0, _WORK_SCALE / source_scale)
    if factor < 1.0:
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        landmarks = landmarks * np.float32(factor)
    scale = regions.face_scale(landmarks)
    roi = regions.roi_around(landmarks[regions.FACE_OVAL], image.shape, pad=0.0, margin_px=_CROP_MARGIN * scale)
    if roi is None:
        raise AnalysisError(f"{label}から顔を見つけられませんでした。顔全体が写っている写真を選んでください。")
    crop = image[roi.slices]
    masks = models.segment(model_input(crop, _SEGMENT_INPUT))

    def up(mask: np.ndarray) -> np.ndarray:
        return cv2.resize(np.asarray(mask, np.float32), (roi.w, roi.h), interpolation=cv2.INTER_LINEAR)

    room_rgb = model_input(image, _SEGMENT_INPUT)
    room_masks = models.segment(room_rgb)
    return _Photo(
        image=crop.astype(np.float32) * np.float32(1.0 / 255.0),
        landmarks=(landmarks - np.asarray(roi.offset, np.float32)).astype(np.float32),
        scale=scale,
        source_scale=source_scale,
        hair=up(masks[HAIR]),
        face_skin=up(masks[FACE_SKIN]),
        body_skin=up(masks[BODY_SKIN]),
        room=cv2.cvtColor(room_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) * np.float32(1.0 / 255.0),
        room_mask=np.asarray(room_masks[BACKGROUND], np.float32),
        yaw=yaw,
        eyes_open=_eyes_open(landmarks),
        others=up(masks[OTHERS]),
        room_person=np.maximum.reduce([np.asarray(room_masks[i], np.float32) for i in (HAIR, BODY_SKIN, CLOTHES)]),
    )


# --------------------------------------------------------------------------
# Colour normalization and geometry between the photos.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Balance:
    gains: np.ndarray  # BGR multipliers: makeup photo = gains * bare photo ** power
    source: str  # "person", "room", "neck", "forehead" or "none"
    trust: float  # 0..1: how far colors of the two photos can be compared
    power: np.ndarray | None = None  # BGR tone curve exponents (None: all 1)
    skin_disagreement: float = 0.0  # log difference left on the skin of the stable zones


def _median_gains(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    gains = np.median(numerator, axis=0) / np.maximum(np.median(denominator, axis=0), 1e-3)
    return np.clip(gains, *_GAIN_RANGE).astype(np.float32)


def _usable(pixels: np.ndarray) -> np.ndarray:
    """Pixels neither clipped nor nearly black in any channel."""
    return pixels[(pixels.min(axis=1) > 0.03) & (pixels.max(axis=1) < 0.97)]


def _channel_power(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of log(target) over log(bare), from medians per lightness bin so the big areas
    (a wall) do not outvote the small ones (hair). 1 without enough range to tell."""
    low, high = np.percentile(x, (2.0, 98.0))
    if high - low < _CURVE_MIN_RANGE:
        return 1.0
    edges = np.linspace(low, high, 9)
    index = np.clip(np.digitize(x, edges) - 1, 0, 7)
    xs, ys, weights = [], [], []
    for i in range(8):
        chosen = index == i
        if chosen.sum() >= 10:
            xs.append(float(np.median(x[chosen])))
            ys.append(float(np.median(y[chosen])))
            weights.append(math.sqrt(float(chosen.sum())))
    if len(xs) < 3 or max(xs) - min(xs) < _CURVE_MIN_RANGE:
        return 1.0
    slope = float(np.polyfit(xs, ys, 1, w=weights)[0])
    return slope if _CURVE_POWER_RANGE[0] <= slope <= _CURVE_POWER_RANGE[1] else 1.0


def _paired_curve(t_px: np.ndarray, b_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """From paired (N, 3) pixels: (gains with power 1, gains with the curve, power, spread
    of the pairs around the curve, spread around the plain gain)."""
    usable = (np.minimum(t_px.min(axis=1), b_px.min(axis=1)) > 0.03) & (
        np.maximum(t_px.max(axis=1), b_px.max(axis=1)) < 0.97
    )
    if usable.sum() < 50:
        return None
    x, y = np.log(b_px[usable]), np.log(t_px[usable])
    power = np.array([_channel_power(x[:, c], y[:, c]) for c in range(3)])
    if np.abs(power - 1.0).max() < _CURVE_MIN_POWER_CHANGE:
        power = np.ones(3)
    plain = np.clip(np.exp(np.median(y - x, axis=0)), *_GAIN_RANGE).astype(np.float32)
    curved = np.clip(np.exp(np.median(y - power * x, axis=0)), *_GAIN_RANGE).astype(np.float32)

    def spread(residual: np.ndarray) -> float:
        return float(np.median(np.abs(residual - np.median(residual, axis=0))) * 1.4826)

    return plain, curved, power, spread(y - power * x), spread(y - x)


def balance_gains(target: _Photo, bare: _Photo, stable_target: np.ndarray, stable_bare: np.ndarray) -> _Balance:
    """How the makeup photo's light differs from the bare photo's (see ``_Balance``).

    What makeup does not touch decides, as long as the two photos really show the same
    thing there, checked by pairing the pixels of their low-resolution frames: first the
    person outside the face (hair, clothes, neck: they are in the same light as the face,
    a wall need not be), then the room. A wide enough range of lightness also tells a tone
    curve (another exposure or camera mode) from a plain gain. Otherwise the neck is
    compared; last, the forehead and mouth area, which removes any foundation change along
    with the cast, so the skin tone cannot be measured then (``trust`` is low). An even wall
    in another room would also give a tight ratio, so the result must roughly agree with
    the skin (foundation hardly changes the color of the forehead and the mouth area
    compared with a change of light).
    """
    t_face, b_face = target.image[stable_target > 0.5], bare.image[stable_bare > 0.5]
    has_skin = len(t_face) >= 50 and len(b_face) >= 50
    skin_gains = _median_gains(t_face, b_face) if has_skin else None
    t_skin = np.log(np.maximum(np.median(t_face, axis=0), 1e-3)) if has_skin else None
    b_skin = np.log(np.maximum(np.median(b_face, axis=0), 1e-3)) if has_skin else None

    def disagreement(gains: np.ndarray, power: np.ndarray) -> float:
        if not has_skin:
            return 0.0
        return float(np.abs(t_skin - np.log(gains) - power * b_skin).max())

    def from_pairs(source: str, t_mask: np.ndarray, b_mask: np.ndarray) -> _Balance | None:
        both = t_mask & b_mask
        if both.mean() < _MIN_ROOM_SHARE:
            return None
        found = _paired_curve(target.room[both], bare.room[both])
        if found is None:
            return None
        plain, curved, power, spread, plain_spread = found
        if spread > _OTHER_ROOM_SPREAD:
            return None
        one = np.ones(3)
        plain_miss = disagreement(plain, one)
        curved_miss = disagreement(curved, power)
        # The pairs themselves prove a curve when no gain can explain them. Otherwise the
        # skin decides (it may wear foundation, so it does not overrule the pairs).
        proven = plain_spread >= _CURVE_MIN_SPREAD and spread < _CURVE_PROOF * plain_spread
        if proven or curved_miss < plain_miss - _CURVE_MIN_IMPROVEMENT:
            gains, used, miss = curved, power, curved_miss
        else:
            gains, used, miss = plain, one, plain_miss
        if miss > _MAX_ROOM_SKIN_DISAGREEMENT:
            return None
        trust = 1.0 - 0.5 * _ramp(spread, _SAME_ROOM_SPREAD, _OTHER_ROOM_SPREAD)
        return _Balance(gains, source, trust, None if np.all(used == 1.0) else used, miss)

    t_room = target.room_mask > _ROOM_CONFIDENCE
    b_room = bare.room_mask > _ROOM_CONFIDENCE
    same_frame = target.room.shape == bare.room.shape
    if same_frame:
        if target.room_person is not None and bare.room_person is not None:
            found = from_pairs("person", target.room_person > _ROOM_CONFIDENCE, bare.room_person > _ROOM_CONFIDENCE)
            if found is not None:
                return found
        found = from_pairs("room", t_room, b_room)
        if found is not None:
            return found
    overlapping = same_frame and (t_room & b_room).mean() >= _MIN_ROOM_SHARE
    if t_room.mean() >= _MIN_ROOM_SHARE and b_room.mean() >= _MIN_ROOM_SHARE and not overlapping:
        t_px, b_px = _usable(target.room[t_room]), _usable(bare.room[b_room])
        if len(t_px) >= 50 and len(b_px) >= 50:
            t_lab = np.median(blend.bgr_to_lab(t_px.reshape(-1, 1, 3)).reshape(-1, 3), axis=0)
            b_lab = np.median(blend.bgr_to_lab(b_px.reshape(-1, 1, 3)).reshape(-1, 3), axis=0)
            gains = _median_gains(t_px, b_px)
            miss = disagreement(gains, np.ones(3))
            if float(np.linalg.norm(t_lab[1:] - b_lab[1:])) < 12.0 and miss <= _MAX_ROOM_SKIN_DISAGREEMENT:
                return _Balance(gains, "room", 0.5, None, miss)
    t_neck, b_neck = target.image[_neck_zone(target)], bare.image[_neck_zone(bare)]
    if len(t_neck) >= 100 and len(b_neck) >= 100:
        gains = _median_gains(t_neck, b_neck)
        return _Balance(gains, "neck", 0.5, None, disagreement(gains, np.ones(3)))
    if skin_gains is not None:
        return _Balance(skin_gains, "forehead", 0.15)
    return _Balance(np.ones(3, np.float32), "none", 0.15)


def _to_bare_light(image: np.ndarray, balance: _Balance) -> np.ndarray:
    """The makeup photo in the bare photo's light (exactly ``image`` when nothing differs)."""
    if balance.power is None:
        return _apply_gains(image, 1.0 / balance.gains)
    ratio = np.maximum(image, 1e-4) / balance.gains.reshape(1, 1, 3)
    return np.clip(ratio ** (1.0 / balance.power.reshape(1, 1, 3).astype(np.float32)), 0.0, 1.0).astype(np.float32)


def _neck_zone(photo: _Photo) -> np.ndarray:
    """Clearly-skin pixels below the chin, as a bool mask."""
    height, width = photo.image.shape[:2]
    chin_y = float(photo.landmarks[_CHIN_BOTTOM][1])
    ys = np.arange(height, dtype=np.float32)[:, None]
    return (photo.body_skin > 0.7) & (ys > chin_y + 0.05 * photo.scale) & np.ones((height, width), bool)


def gray_world_gains(photo: _Photo) -> tuple[np.ndarray, float | None]:
    """Partial correction that neutralizes a near-neutral room, and the room's chroma.

    (gains 1, chroma) when the room is colorful (a colored wall, a green
    screen); (gains 1, None) when too little room is visible.
    """
    room = photo.room_mask > _ROOM_CONFIDENCE
    if room.mean() < _MIN_ROOM_SHARE:
        return np.ones(3, np.float32), None
    pixels = _usable(photo.room[room])
    if len(pixels) < 50:
        return np.ones(3, np.float32), None
    median = np.median(pixels, axis=0)
    lab = _lab_of_bgr(median)
    chroma = float(np.hypot(lab[1], lab[2]))
    if chroma > _GRAY_WORLD_MAX_CHROMA:
        return np.ones(3, np.float32), chroma
    gains = (float(median.mean()) / np.maximum(median, 1e-3)) ** _GRAY_WORLD_ADAPTATION
    gains = np.clip(gains, 1.0 - _GRAY_WORLD_MAX_GAIN, 1.0 + _GRAY_WORLD_MAX_GAIN).astype(np.float32)
    return gains, chroma


def _clipped(image: np.ndarray) -> np.ndarray:
    """Pixels blown out in any channel (bool). Their real color is unknown."""
    return image.max(axis=2) >= _CLIPPED_LEVEL


def _apply_gains(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    if np.abs(gains - 1.0).max() < _MIN_GAIN_CHANGE:
        return image
    return np.minimum(image * gains.reshape(1, 1, 3), 1.0)


def _tps_map(
    source: np.ndarray, values: np.ndarray, size: tuple[int, int], step: int, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Thin-plate spline from ``source`` to ``values`` points, evaluated for every
    pixel of ``size`` = (w, h): maps (x, y) for ``cv2.remap``, on a coarse grid
    upsampled so it stays cheap. Coordinates are pixel centers."""
    pts = source.astype(np.float64) / scale
    n = len(pts)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    kernel = np.where(dist > 0, dist * dist * np.log(np.maximum(dist, 1e-9)), 0.0) + np.eye(n) * 1e-3
    poly = np.hstack([np.ones((n, 1)), pts])
    system = np.zeros((n + 3, n + 3))
    system[:n, :n] = kernel
    system[:n, n:] = poly
    system[n:, :n] = poly.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = values.astype(np.float64) / scale
    coeffs = np.linalg.solve(system, rhs)
    width, height = size
    grid_w = width // step + 2
    grid_h = height // step + 2
    xs, ys = np.meshgrid(np.arange(grid_w) * float(step), np.arange(grid_h) * float(step))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1) / scale
    d = np.linalg.norm(grid[:, None] - pts[None], axis=2)
    u = np.where(d > 0, d * d * np.log(np.maximum(d, 1e-9)), 0.0)
    mapped = (u @ coeffs[:n] + np.hstack([np.ones((len(grid), 1)), grid]) @ coeffs[n:]) * scale
    map_x = _upsample_grid(mapped[:, 0].reshape(grid_h, grid_w), step, width, height)
    map_y = _upsample_grid(mapped[:, 1].reshape(grid_h, grid_w), step, width, height)
    return map_x, map_y


def _upsample_grid(grid: np.ndarray, step: int, width: int, height: int) -> np.ndarray:
    """Bilinear interpolation of values on grid nodes ``k * step`` at every pixel (float32).
    (cv2.resize aligns pixel centers instead of nodes, which would shift the map.)"""
    xs = np.arange(width) / step
    ys = np.arange(height) / step
    x0 = np.minimum(xs.astype(int), grid.shape[1] - 2)
    y0 = np.minimum(ys.astype(int), grid.shape[0] - 2)
    fx = (xs - x0)[None, :]
    fy = (ys - y0)[:, None]
    top = grid[y0][:, x0] * (1.0 - fx) + grid[y0][:, x0 + 1] * fx
    bottom = grid[y0 + 1][:, x0] * (1.0 - fx) + grid[y0 + 1][:, x0 + 1] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def _in_front_of_face(photo: _Photo) -> np.ndarray:
    """Hair and accessories (glasses, a mask): what hides makeup in this photo."""
    return photo.hair if photo.others is None else np.maximum(photo.hair, photo.others)


@dataclass
class _Aligned:
    image: np.ndarray
    hair: np.ndarray
    face_skin: np.ndarray
    body_skin: np.ndarray
    clipped: np.ndarray  # bool: blown out in the bare photo
    mismatch: float  # landmark distance after a similarity fit, x face scale


def _pose_mismatch(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Median landmark distance after the best similarity transform, x face scale."""
    pa, pb = a[:468].astype(np.float64), b[:468].astype(np.float64)
    ca, cb = pa.mean(axis=0), pb.mean(axis=0)
    pa, pb = pa - ca, pb - cb
    za, zb = pa[:, 0] + 1j * pa[:, 1], pb[:, 0] + 1j * pb[:, 1]
    k = (np.conj(za) @ zb) / max(float(np.real(np.conj(za) @ za)), 1e-9)
    return float(np.median(np.abs(za * k - zb))) / max(scale, 1e-6)


def _align_bare(bare: _Photo, image: np.ndarray, target: _Photo) -> _Aligned:
    """The bare photo warped onto the makeup photo's landmarks."""
    height, width = target.image.shape[:2]
    step = max(4, int(round(target.scale / 25.0)))
    points = np.asarray(_WARP_POINTS)
    map_x, map_y = _tps_map(
        target.landmarks[points] - 0.5, bare.landmarks[points] - 0.5, (width, height), step, target.scale
    )

    def warp(a: np.ndarray, interpolation: int) -> np.ndarray:
        return cv2.remap(a, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)

    warped = np.clip(warp(image, cv2.INTER_CUBIC), 0.0, 1.0)
    return _Aligned(
        warped,
        warp(_in_front_of_face(bare), cv2.INTER_LINEAR),
        warp(bare.face_skin, cv2.INTER_LINEAR),
        warp(bare.body_skin, cv2.INTER_LINEAR),
        warp(_clipped(bare.image).astype(np.float32), cv2.INTER_LINEAR) > 0.1,
        _pose_mismatch(target.landmarks, bare.landmarks, target.scale),
    )


# --------------------------------------------------------------------------
# The shared state of a fit.
# --------------------------------------------------------------------------


@dataclass
class _Scene:
    target: np.ndarray  # the made-up face
    base: np.ndarray  # the same face without makeup (warped bare photo, or approximated)
    landmarks: np.ndarray
    scale: float
    hair: np.ndarray
    face_skin: np.ndarray
    body_skin: np.ndarray
    skin_ok: np.ndarray  # bool: skin in both photos, not blown out
    unclipped: np.ndarray  # bool: not blown out in either photo
    has_bare: bool
    # Whether the bare photo shows the same state: the eyes (opening, brow height) and the
    # rest of the face (expression, pose: the skin and its shading).
    eyes_comparable: bool
    face_comparable: bool
    tone_trust: float
    settings: MakeupSettings
    measurements: dict[str, Any]
    confidence: dict[str, float]
    notes: list[str]
    effects: dict[str, Effect]
    current: np.ndarray = None  # type: ignore[assignment]  # base with the fitted sections so far
    target_lab: np.ndarray = None  # type: ignore[assignment]
    eyes_hidden: bool = False  # something (dark glasses) is in front of the eyes
    lips_blown: bool = False  # the lips are blown out in the makeup photo

    def __post_init__(self) -> None:
        self.current = self.base
        self.target_lab = blend.bgr_to_lab(self.target)

    @property
    def roi(self) -> regions.Roi:
        return regions.Roi(0, 0, self.target.shape[1], self.target.shape[0])

    def context(self) -> FrameContext:
        height, width = self.target.shape[:2]
        return FrameContext(
            height=height,
            width=width,
            landmarks=self.landmarks,
            hair_mask=self.hair,
            face_skin_mask=self.face_skin,
            body_skin_mask=self.body_skin,
        )

    def render(self, name: str, section: Any, image: np.ndarray | None = None) -> np.ndarray:
        """``section`` of effect ``name`` applied to a copy of ``image`` (default: current)."""
        source = self.current if image is None else image
        return self.effects[name].apply(source.copy(), self.context(), section)

    def commit(self, name: str) -> None:
        """Apply the fitted section to the current image."""
        self.current = self.render(name, getattr(self.settings, name))


def _load_effects() -> dict[str, Effect]:
    """One instance of each fitted effect (imported here: importing registers them)."""
    from .effects.blush import BlushEffect
    from .effects.brows import BrowEffect
    from .effects.contour import ContourEffect, TearbagEffect
    from .effects.eyes import EyelinerEffect, EyeshadowEffect
    from .effects.lips import LipEffect
    from .effects.skin import SkinEffect

    return {
        e.name: e
        for e in (
            SkinEffect(),
            ContourEffect(),
            BrowEffect(),
            EyeshadowEffect(),
            TearbagEffect(),
            EyelinerEffect(),
            BlushEffect(),
            LipEffect(),
        )
    }


def _stable_skin_zone(landmarks: np.ndarray, roi: regions.Roi, skin_ok: np.ndarray) -> np.ndarray:
    """Forehead center and the skin between nose and mouth: makeup rarely changes
    them except for the foundation (they are outside blush and most contour).
    Float32 0/1 mask; empty when hair covers all of it."""
    zone = regions.polygon_mask(landmarks[_FOREHEAD_ZONE], roi)
    for polygon in _PERIORAL_ZONES:
        zone = np.maximum(zone, regions.polygon_mask(landmarks[polygon], roi))
    return (zone > 0.5).astype(np.float32) * skin_ok.astype(np.float32)


def _stable_lab(lab: np.ndarray, landmarks: np.ndarray, roi: regions.Roi, skin_ok: np.ndarray) -> np.ndarray | None:
    """Mean over the stable zones of each zone's trimmed median. One median over
    all of them would jump between the bright forehead and the darker mouth area."""
    medians = []
    for polygon in (_FOREHEAD_ZONE, *_PERIORAL_ZONES):
        zone = (regions.polygon_mask(landmarks[polygon], roi) > 0.5) & skin_ok
        if zone.sum() >= 8:
            medians.append(trimmed_median(lab[zone]))
    return np.mean(medians, axis=0) if medians else None


def _zone_lab(lab: np.ndarray, zone: np.ndarray, trim: tuple[float, float] = (15.0, 85.0)) -> np.ndarray | None:
    return trimmed_median(lab[zone > 0.5], trim)


def _noise_factor(scene: _Scene) -> float:
    """How much more of a difference it takes to count as makeup when the two
    photos differ in pose or expression (1 for matching photos)."""
    return 1.0 + _POSE_NOISE * float(scene.measurements["photos"]["pose_mismatch"])


def _visible_share(scene: _Scene, zone: np.ndarray) -> float:
    """Share of a zone not covered by hair or anything else that is not face."""
    total = float(np.sum(zone > 0.5))
    if total <= 0:
        return 0.0
    return float(np.sum((zone > 0.5) & (scene.hair < 0.5) & scene.skin_ok)) / total


def _typical_gain(weight: np.ndarray, base: np.ndarray, delta: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Per-channel ``v`` such that ``delta`` and ``weight * base * v`` have the same mean
    where the mask is substantial inside ``zone`` (a bool mask). A least-squares slope goes
    wrong when the makeup sits a pixel off the landmarks, as real makeup does. The most
    extreme pixels are left out: stray hairs and lashes that moved between the photos."""
    core = zone & (weight > _GAIN_MASK_LEVEL)
    if not core.any():
        return np.zeros(3)
    tinted = weight[core][:, None] * base[core]
    change = delta[core]
    low, high = np.percentile(change / np.maximum(tinted, 1e-3), _GAIN_TRIM, axis=0)
    ratio = change / np.maximum(tinted, 1e-3)
    kept = ((ratio >= low) & (ratio <= high)).astype(np.float64)
    return (change * kept).sum(axis=0) / np.maximum((tinted * kept).sum(axis=0), 1e-6)


def _response_line(
    apply_color: Callable[[np.ndarray, Sequence[float], np.ndarray], np.ndarray], base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For a tint that is linear in the color: relative change per channel at full
    weight with color 0 and with color 1, on a skin of color ``base``."""
    patch = np.asarray(base, np.float32).reshape(1, 1, 3)
    weight = np.ones((1, 1), np.float32)
    at_0 = apply_color(patch, (0.0, 0.0, 0.0), weight)[0, 0] / np.maximum(patch[0, 0], 1e-3) - 1.0
    at_1 = apply_color(patch, (1.0, 1.0, 1.0), weight)[0, 0] / np.maximum(patch[0, 0], 1e-3) - 1.0
    return at_0.astype(np.float64), at_1.astype(np.float64)


def canonical_tint(
    gain: np.ndarray,
    at_0: np.ndarray,
    at_1: np.ndarray,
    criterion: Callable[[np.ndarray], float],
    goal: float,
) -> tuple[float, np.ndarray]:
    """The (opacity, BGR color) of a relative change ``gain`` whose color fits ``goal``.

    ``gain = opacity * (at_0 + color * (at_1 - at_0))`` per channel has one
    solution per opacity, as far as the color is displayable. This returns the
    one that reproduces ``gain`` best and, among equally good ones, has its
    ``criterion(color)`` closest to ``goal``.
    """
    best: tuple[float, float, np.ndarray] | None = None
    span = np.where(np.abs(at_1 - at_0) < 1e-6, 1e-6, at_1 - at_0)
    for opacity in _OPACITY_STEPS:
        color = np.clip((gain / opacity - at_0) / span, 0.0, 1.0)
        miss = float(np.abs(opacity * (at_0 + color * span) - gain).max())
        score = miss * _TINT_MISS_WEIGHT + abs(criterion(color) - goal)
        if best is None or score < best[0]:
            best = (score, float(opacity), color)
    assert best is not None
    return best[1], best[2]


def _chroma_of(color: np.ndarray) -> float:
    lab = _lab_of_bgr(color)
    return float(np.hypot(lab[1], lab[2]))


def _lightness_of(color: np.ndarray) -> float:
    return float(_lab_of_bgr(color)[0])


# --------------------------------------------------------------------------
# Removing the makeup from a single photo.
# --------------------------------------------------------------------------


def natural_lip_lab(skin_lab: Sequence[float]) -> np.ndarray:
    """The natural lip color expected on a skin, from the test faces (see the constants)."""
    lightness, a, b = (float(v) for v in skin_lab)
    slope, pivot, offset = _LIP_A_SLOPE
    return np.array(
        [
            max(lightness + _LIP_FROM_SKIN_L, 8.0),
            a + max(slope * (lightness - pivot) + offset, _LIP_A_MIN_OFFSET),
            b * _LIP_B_RATIO,
        ]
    )


def lipstick_distance(lip_lab: Sequence[float], natural_lab: Sequence[float]) -> float:
    """How far lips are from the natural color, in the terms that tell lipstick."""
    dl, da, db = (float(x) - float(y) for x, y in zip(lip_lab, natural_lab))
    return float(np.linalg.norm([da, db, max(-dl - _NATURAL_LIP_DARK_FREE, 0.0)]))


def _looks_like_lipstick(lip_lab: Sequence[float], natural_lab: Sequence[float]) -> bool:
    """Far enough from the natural lip color, in the direction lipsticks take it."""
    distance = lipstick_distance(lip_lab, natural_lab)
    hue_gap = math.degrees(math.atan2(float(natural_lab[2]), float(natural_lab[1]))) - math.degrees(
        math.atan2(float(lip_lab[2]), float(lip_lab[1]))
    )
    return any(distance >= least and hue_gap >= gap for least, gap in _LIPSTICK_RULES)


def _lid_zones(landmarks: np.ndarray, roi: regions.Roi, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(mobile lid, brow bone) as bool masks, off the brows, lashes and eyes."""
    from .effects import eyes

    lid = np.zeros((roi.h, roi.w), bool)
    bone = np.zeros((roi.h, roi.w), bool)
    for side in regions.SIDES:
        axes = eyes._eye_axes(landmarks, side)
        t = eyes._lid_coordinate(landmarks, side, roi, 1.0)
        u = axes.along_coordinate(roi)
        region = regions.polygon_mask(eyes._lid_outline(landmarks, side, axes, scale), roi) > 0.5
        region &= regions.brow_mask(landmarks, side, roi, thickness=0.5) < 0.5
        region &= regions.eye_opening_mask(landmarks, side, roi, expand_px=0.03 * scale) < 0.5
        span = (u > 0.1) & (u < 0.95)
        lid |= region & span & (t > 0.1) & (t < 0.45)
        bone |= region & span & (t > 0.5) & (t < 0.8)
    return lid, bone


def _cheek_bump(photo: _Photo, image: np.ndarray, skin_ok: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """The blush mask at full strength, and the amplitude of that shape in
    (log(R/G), log(G/B)) over a plane fitted per cheek. Log chromaticity does not
    follow the shading, unlike Lab a*/b*, which fade toward the sides of a dark face."""
    from .effects.blush import BlushEffect, _flush

    landmarks, scale = photo.landmarks, photo.scale
    height, width = image.shape[:2]
    roi = regions.Roi(0, 0, width, height)
    flat = np.full((height, width, 3), 0.5, np.float32)
    full = float(_flush(flat[:1, :1], (0.0, 0.0, 0.0), np.ones((1, 1), np.float32))[0, 0, 0] - 0.5)
    ctx = FrameContext(
        height=height, width=width, landmarks=landmarks, face_skin_mask=photo.face_skin, hair_mask=photo.hair
    )
    trial = type(MakeupSettings().blush)(enabled=True, color="#000000", opacity=1.0)
    bump = (BlushEffect().apply(flat, ctx, trial)[..., 0] - 0.5) / full
    zone = regions.face_oval_mask(landmarks, roi, expand_px=-0.06 * scale) > 0.5
    for side in regions.SIDES:
        zone &= regions.eye_opening_mask(landmarks, side, roi, expand_px=0.15 * scale) < 0.5
    center_points = regions.LIPS_OUTER + regions.NOSE_BRIDGE[2:] + regions.RIGHT_NOSE_WING + regions.LEFT_NOSE_WING
    center = cv2.convexHull(landmarks[center_points].astype(np.float32))
    zone &= regions.polygon_mask(regions.expand_polygon(center.reshape(-1, 2), 0.03 * scale), roi) < 0.5
    zone &= skin_ok
    ys, xs = np.nonzero(zone)
    if len(xs) < 200:
        return None, np.zeros(2)
    sigma = 0.03 * scale
    smooth = _blur(image, sigma)
    log_rg = np.log(np.maximum(smooth[..., 2], 0.02) / np.maximum(smooth[..., 1], 0.02))
    log_gb = np.log(np.maximum(smooth[..., 1], 0.02) / np.maximum(smooth[..., 0], 0.02))
    middle = float(landmarks[regions.NOSE_TIP][0])
    u, v = (xs - middle) / scale, (ys - float(landmarks[regions.NOSE_TIP][1])) / scale
    left = (xs >= middle).astype(np.float64)
    right = 1.0 - left
    bump_samples = _blur(bump, sigma)[ys, xs].astype(np.float64)
    design = np.stack([bump_samples, left, right, u * left, u * right, v * left, v * right], axis=1)
    observed = np.stack([log_rg[ys, xs], log_gb[ys, xs]], axis=1).astype(np.float64)
    solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
    return bump, solution[0]


def _liner_band(landmarks: np.ndarray, roi: regions.Roi, side: regions.Side, scale: float) -> np.ndarray:
    """Where a liner (with its wing) could be, as a bool mask; the eye opening is included."""
    from .effects import eyes

    axes = eyes._eye_axes(landmarks, side)
    s = regions.side_indices(side)
    lash = landmarks[s.eye_upper].astype(np.float32)
    tip = axes.outer + axes.direction(eyes._WING_ANGLE_DEG) * np.float32(0.26 * scale)
    upper = regions.polyline_mask(np.vstack([lash, tip]), 0.12 * scale, roi)
    lower = regions.polyline_mask(landmarks[s.eye_lower].astype(np.float32), 0.07 * scale, roi)
    opening = regions.eye_opening_mask(landmarks, side, roi, expand_px=0.01 * scale)
    return (upper > 0.2) | (lower > 0.2) | (opening > 0.2)


def _approximate_bare(photo: _Photo, skin_ok: np.ndarray, measurements: dict[str, Any]) -> np.ndarray:
    """The makeup photo with its visible makeup taken off (see the module docstring)."""
    image, landmarks, scale = photo.image, photo.landmarks, photo.scale
    roi = regions.Roi(0, 0, image.shape[1], image.shape[0])
    lab = blend.bgr_to_lab(image)
    skin = _stable_lab(lab, landmarks, roi, skin_ok)
    if skin is None:
        skin = _zone_lab(lab, regions.face_oval_mask(landmarks, roi, expand_px=-0.15 * scale) * skin_ok)
    if skin is None:
        return image.copy()
    out = lab.copy()
    info: dict[str, Any] = {"skin_lab": skin}

    # Lips: natural color, keeping the lightness texture.
    core = regions.lips_mask(landmarks, roi, outline=-0.12) > 0.9
    lip = _zone_lab(lab, core.astype(np.float32))
    if lip is not None:
        natural = natural_lip_lab(skin)
        distance = lipstick_distance(lip, natural)
        info["lip_lab"], info["natural_lip_lab"], info["lip_distance"] = lip, natural, distance
        info["lipstick"] = _looks_like_lipstick(lip, natural)
        if info["lipstick"]:
            paint = regions.lips_mask(landmarks, roi, outline=0.05, feather_px=0.012 * scale)
            out[..., 0] += paint * np.float32(natural[0] - lip[0])
            out[..., 1] += paint * (np.float32(natural[1]) - lab[..., 1])
            out[..., 2] += paint * (np.float32(natural[2]) - lab[..., 2])

    # Lids: toward the brow bone (plus the natural lid-to-bone difference).
    lid_zone, bone_zone = _lid_zones(landmarks, roi, scale)
    bone = trimmed_median(lab[bone_zone & skin_ok], (40.0, 90.0))
    lid = trimmed_median(lab[lid_zone & skin_ok])
    if bone is not None and lid is not None:
        natural_lid = bone + np.asarray(_NATURAL_LID)
        distance = float(delta_e00(lid, natural_lid))
        expression = max(
            _ramp(_mouth_shape(landmarks)[0], *_SMILE_WIDTH), 1.0 - _ramp(photo.eyes_open, *_NARROWED_EYES)
        )
        margin = _NATURAL_LID_MARGIN + (_EXPRESSION_LID_MARGIN - _NATURAL_LID_MARGIN) * expression
        info["bone_lab"], info["lid_lab"], info["lid_distance"], info["lid_margin"] = bone, lid, distance, margin
        info["lid_shifted"] = distance > margin
        if distance > margin:
            # Soft threshold: only what is above the margin is believed, so there is no
            # jump when a bare eye comes close to it.
            kept = 1.0 - margin / distance
            shift = (lid - natural_lid) * kept * _ramp(photo.eyes_open, *_SQUINT)
            shift[0] = max(float(shift[0]), -_MAX_LID_DARKENING)
            area = np.zeros((roi.h, roi.w), np.float32)
            for side in regions.SIDES:
                area = np.maximum(area, regions.eyelid_mask(landmarks, side, roi, 0.9, feather_px=0.02 * scale))
            for i in range(3):
                out[..., i] -= area * np.float32(shift[i])
    result = blend.lab_to_bgr(out)

    # Cheeks: a smooth bump of chromaticity on the apples, measured against the rest
    # of each cheek. What goes beyond a natural flush is divided out.
    bump, amplitude = _cheek_bump(photo, image, skin_ok)
    if bump is not None:
        info["cheek_amplitude"] = amplitude
        if float(amplitude[0]) > _NATURAL_CHEEK_MARGIN:
            share = 1.0 - _NATURAL_CHEEK_KEEP / float(amplitude[0])
            rg, gb = float(amplitude[0]) * share, float(amplitude[1]) * share
            # Keep G and move R and B, so the lightness stays about the same.
            result = result * np.exp(bump[..., None] * np.array([gb, 0.0, -rg], np.float32))
            result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # Lash lines: inpainted from the skin around them (the eye opening comes back afterwards).
    canvas = np.clip(result * 255.0 + 0.5, 0, 255).astype(np.uint8)
    for side in regions.SIDES:
        band = _liner_band(landmarks, roi, side, scale)
        ys, xs = np.nonzero(band)
        if len(xs) == 0:
            continue
        pad = 6
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, roi.w)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, roi.h)
        filled = cv2.inpaint(canvas[y0:y1, x0:x1], band[y0:y1, x0:x1].astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        keep = regions.eye_opening_mask(landmarks, side, roi)[y0:y1, x0:x1] > 0.5
        filled[keep] = canvas[y0:y1, x0:x1][keep]
        patch = result[y0:y1, x0:x1]
        mask = band[y0:y1, x0:x1] & ~keep
        patch[mask] = filled[mask].astype(np.float32) * np.float32(1.0 / 255.0)
    measurements["approximate_bare"] = info
    return result


# --------------------------------------------------------------------------
# Skin.
# --------------------------------------------------------------------------


def _retouch_zone(scene: _Scene) -> np.ndarray:
    """Skin away from the features, as a bool mask (for texture statistics)."""
    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    zone = regions.face_oval_mask(landmarks, roi, expand_px=-0.12 * scale) > 0.5
    for side in regions.SIDES:
        zone &= regions.eye_opening_mask(landmarks, side, roi, expand_px=0.14 * scale) < 0.5
        zone &= regions.brow_mask(landmarks, side, roi, thickness=1.2) < 0.5
    zone &= regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_OUTER], 0.08 * scale), roi) < 0.5
    hull = cv2.convexHull(landmarks[regions.RIGHT_NOSE_WING + regions.LEFT_NOSE_WING].astype(np.float32)).reshape(-1, 2)
    zone &= regions.polygon_mask(regions.expand_polygon(hull, 0.03 * scale), roi) < 0.5
    return zone & scene.skin_ok


def _texture_energy(lightness: np.ndarray, zone: np.ndarray, scale: float) -> tuple[float, float, float]:
    """(fine, medium, dark spots) energy of a Lab L channel inside ``zone``."""
    fine_blur = _blur(lightness, 0.015 * scale)
    wide_blur = _blur(lightness, 0.05 * scale)
    fine = (lightness - fine_blur)[zone]
    medium = (fine_blur - wide_blur)[zone]
    spot = _blur(lightness, 0.07 * scale) - _blur(lightness, 0.03 * scale)
    return _robust_std(fine), _robust_std(medium), float(np.percentile(spot[zone], 98)) if zone.any() else 0.0


def _grid_search(values: Sequence[float], cost: Callable[[float], float]) -> tuple[float, float]:
    """(best value, its cost). Ties go to the earliest value."""
    best_value, best_cost = values[0], math.inf
    for value in values:
        c = cost(value)
        if c < best_cost - 1e-9:
            best_value, best_cost = value, c
    return best_value, best_cost


def _fit_skin_texture(scene: _Scene, zone: np.ndarray, section: Any, info: dict[str, Any]) -> None:
    """Smoothing, texture and blemish cover from how much skin detail is left."""
    target = _texture_energy(scene.target_lab[..., 0], zone, scene.scale)
    bare = _texture_energy(blend.bgr_to_lab(scene.base)[..., 0], zone, scene.scale)
    info["texture_target"], info["texture_bare"] = target, bare
    section.smoothing = section.texture = section.blemish = 0.0
    if target[0] >= bare[0] * (1.0 - _MIN_ENERGY_CHANGE) and target[1] >= bare[1] * (1.0 - _MIN_ENERGY_CHANGE):
        return

    def cost() -> float:
        out = scene.render("skin", section, scene.base)
        r = _texture_energy(blend.bgr_to_lab(out)[..., 0], zone, scene.scale)
        return sum(
            math.log(max(r[i], 1e-3) / max(target[i], 1e-3)) ** 2 for i in range(2)
        )

    def set_and_cost(name: str) -> Callable[[float], float]:
        def inner(value: float) -> float:
            setattr(section, name, value)
            return cost()

        return inner

    section.texture = 0.35
    section.smoothing, _ = _grid_search([round(v * 0.1, 1) for v in range(11)], set_and_cost("smoothing"))
    section.texture, _ = _grid_search((0.0, 0.2, 0.35, 0.5, 0.7, 1.0), set_and_cost("texture"))
    section.smoothing, _ = _grid_search([round(v * 0.1, 1) for v in range(11)], set_and_cost("smoothing"))

    def spot_cost(value: float) -> float:
        section.blemish = value
        out = scene.render("skin", section, scene.base)
        return abs(_texture_energy(blend.bgr_to_lab(out)[..., 0], zone, scene.scale)[2] - target[2])

    section.blemish, _ = _grid_search((0.0, 0.25, 0.5, 0.75, 1.0), spot_cost)


def _tone_zone(scene: _Scene) -> np.ndarray:
    """Skin that makeup other than foundation hardly tints: off the features and the
    apples of the cheeks. Bool mask."""
    zone = _retouch_zone(scene)
    for side in regions.SIDES:
        apple = regions.cheek_center(scene.landmarks, side)
        zone &= regions.ellipse_mask(apple, (0.6 * scene.scale, 0.45 * scene.scale), 0.0, scene.roi) < 0.5
    return zone


def _fit_skin_tone(scene: _Scene, stable: np.ndarray, section: Any, info: dict[str, Any]) -> None:
    """Foundation (shade and strength) and brightness.

    skin.py moves each pixel's a*/b* a share ``strength`` of the way to one target,
    so the change depends linearly on the pixel's own a*/b*: the slope gives the
    strength, the mean the target. Likewise brightness lifts by a share of the
    headroom (100 - L), the foundation by a constant.
    """

    forehead = (regions.polygon_mask(scene.landmarks[_FOREHEAD_ZONE], scene.roi) > 0.5) & scene.skin_ok

    def measure(lab: np.ndarray) -> np.ndarray | None:
        # The chroma from the forehead when it is visible: a big blush reaches the mouth area.
        found = _stable_lab(lab, scene.landmarks, scene.roi, scene.skin_ok)
        if found is not None and forehead.sum() >= _MIN_FOREHEAD_PIXELS:
            found[1:] = trimmed_median(lab[forehead])[1:]
        return found

    def rendered_lab() -> np.ndarray | None:
        return measure(blend.bgr_to_lab(scene.render("skin", section, scene.base)))

    target = measure(scene.target_lab)
    if target is None:
        return
    section.tone_strength = 0.0
    section.brightness = 0.0
    start = blend.bgr_to_lab(scene.render("skin", section, scene.base))
    skin = measure(start)
    if skin is None:
        return
    info["tone_delta_lab"] = target - skin
    sigma = 0.03 * scene.scale
    current_blur = _blur(start, sigma)
    target_blur = _blur(scene.target_lab, sigma)
    wide = _tone_zone(scene)
    if wide.sum() < 200:
        wide = stable > 0.5
    x = current_blur[wide][:, 1:].astype(np.float64)
    y = (target_blur - current_blur)[wide][:, 1:].astype(np.float64)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    variance = float(((x - x_mean) ** 2).sum())
    strength = -float(((x - x_mean) * (y - y_mean)).sum()) / max(variance, 1e-6)
    spread = math.sqrt(variance / max(len(x), 1))
    info.update(tone_slope=strength, tone_spread=spread)
    change = float(np.linalg.norm(target[1:] - skin[1:]))
    needed = change / (_TONE_COVERAGE * _MAX_TONE_DISTANCE)
    info["tone_needed"] = needed
    if change >= _MIN_TONE_CHANGE and spread >= _MIN_TONE_SPREAD and strength < _TONE_EXPLAINED * needed:
        _reject_tone(scene, section, info)
        return
    strength = _clip(strength, 0.0, 1.0) if spread >= _MIN_TONE_SPREAD else 0.0
    if change >= _MIN_TONE_CHANGE or (strength >= _MIN_TONE_STRENGTH and change >= _MIN_LIGHTNESS_CHANGE):
        # A foundation far from the skin at a low strength would pull a live face
        # too hard: keep the implied shade plausible by raising the strength.
        strength = max(strength, _MIN_TONE_STRENGTH, change / (_TONE_COVERAGE * _MAX_TONE_DISTANCE))
        strength = _clip(strength, 0.0, _MAX_TONE_STRENGTH)
        tone = np.array([min(float(target[0]) + _TONE_ABOVE_SKIN, 92.0), 0.0, 0.0])
        tone[1:] = skin[1:] + (target[1:] - skin[1:]) / (_TONE_COVERAGE * strength)
        section.tone_strength = strength
        section.tone_color = _hex_of_lab(tone)
        for _ in range(2):
            rendered = rendered_lab()
            if rendered is None:
                break
            miss = target - rendered
            if float(np.hypot(miss[1], miss[2])) < 0.3:
                break
            tone[1:] += miss[1:] / (_TONE_COVERAGE * strength)
            section.tone_color = _hex_of_lab(tone)
        hue = math.degrees(math.atan2(float(tone[2]), float(tone[1])))
        info["tone_lab"] = tone
        if not (_TONE_HUE_RANGE[0] <= hue <= _TONE_HUE_RANGE[1]) or float(np.hypot(tone[1], tone[2])) > _TONE_MAX_CHROMA:
            _reject_tone(scene, section, info)
            return
    # Lightness: what is left on the stable zones (contour stays out of them).
    miss_l = 0.0
    for _ in range(3):
        rendered = rendered_lab()
        if rendered is None:
            return
        miss_l = float(target[0] - rendered[0])
        if abs(miss_l) < _MIN_LIGHTNESS_CHANGE:
            miss_l = 0.0
            break
        room = (100.0 - float(rendered[0])) if miss_l > 0 else float(rendered[0])
        brightness = section.brightness + miss_l / max(_BRIGHTNESS_GAIN * room, 1.0)
        section.brightness = _clip(brightness, -_MAX_BRIGHTNESS, _MAX_BRIGHTNESS)
    rendered = rendered_lab()
    if rendered is not None and abs(float(target[0] - rendered[0])) > 2.0 * _MIN_LIGHTNESS_CHANGE:
        # More than the largest brightness: the light differs.
        _reject_tone(scene, section, info)


def _reject_tone(scene: _Scene, section: Any, info: dict[str, Any]) -> None:
    """The skin's color differs between the photos in a way no foundation does."""
    section.tone_strength = 0.0
    section.brightness = 0.0
    scene.tone_trust = min(scene.tone_trust, 0.15)
    info["tone_rejected"] = True
    scene.notes.append(
        "2枚の写真で肌の色が違いますが、ファンデの色の変化には見えません（光の違いの可能性があります）。ファンデは元のプリセットのままです。"
    )


def _fit_skin_bare(scene: _Scene, stable: np.ndarray, section: Any, info: dict[str, Any]) -> None:
    """All skin sliders, by rendering the skin effect on the bare face."""
    zone = _retouch_zone(scene)
    section.redness = 0.0
    section.under_eye = 0.0
    section.tone_strength = 0.0
    section.brightness = 0.0
    if zone.sum() >= 200:
        _fit_skin_texture(scene, zone, section, info)
    if scene.tone_trust >= _MIN_TONE_TRUST:
        _fit_skin_tone(scene, stable, section, info)
    if scene.tone_trust < _MIN_TONE_TRUST:
        section.tone_strength = 0.0
        section.brightness = 0.0
        info["tone_skipped"] = True

    # Redness, on the skin blush does not reach.
    red_zone = stable > 0.5
    if red_zone.sum() >= 50:
        sigma = 0.03 * scene.scale

        def redness_of(lab: np.ndarray) -> float:
            a = _blur(lab[..., 1], sigma)[red_zone]
            return float(np.percentile(a, 90) - np.median(a))

        goal = redness_of(scene.target_lab)

        def red_cost(value: float) -> float:
            section.redness = value
            return abs(redness_of(blend.bgr_to_lab(scene.render("skin", section, scene.base))) - goal) + 0.1 * value

        section.redness, _ = _grid_search((0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0), red_cost)

    # Concealer: the lower part of the under-eye area (the tear bag sits above it), on the
    # nose side (a cheekbone highlight reaches the other side).
    area = np.zeros(scene.target.shape[:2], np.float32)
    from .effects import eyes

    for side in regions.SIDES:
        low = regions.under_eye_mask(scene.landmarks, side, scene.roi, depth=0.75)
        high = regions.under_eye_mask(scene.landmarks, side, scene.roi, depth=0.25)
        inner = eyes._eye_axes(scene.landmarks, side).along_coordinate(scene.roi) < _CONCEALER_ZONE_END
        area = np.maximum(area, low * (1.0 - high) * inner.astype(np.float32))
    area *= scene.skin_ok.astype(np.float32)
    if (area > 0.5).sum() >= 30:
        goal_lab = _zone_lab(scene.target_lab, area)
        if scene.tone_trust < _MIN_TONE_TRUST:
            # In another light, what counts is the change beyond the rest of the skin's.
            shift_to = _stable_lab(scene.target_lab, scene.landmarks, scene.roi, scene.skin_ok)
            shift_from = _stable_lab(blend.bgr_to_lab(scene.base), scene.landmarks, scene.roi, scene.skin_ok)
            if shift_to is not None and shift_from is not None:
                goal_lab = goal_lab - (shift_to - shift_from)

        def concealer_cost(value: float) -> float:
            section.under_eye = value
            lab = _zone_lab(blend.bgr_to_lab(scene.render("skin", section, scene.base)), area)
            return float(np.abs(lab - goal_lab)[[0, 2]].sum()) + 0.4 * value

        section.under_eye, _ = _grid_search((0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0), concealer_cost)


def _fit_skin(scene: _Scene, gray_world_applied: bool) -> None:
    section = scene.settings.skin
    kept_tone = (section.tone_color, section.tone_strength, section.brightness)
    info: dict[str, Any] = {}
    scene.measurements["skin"] = info
    stable = _stable_skin_zone(scene.landmarks, scene.roi, scene.skin_ok)
    target = _stable_lab(scene.target_lab, scene.landmarks, scene.roi, scene.skin_ok)
    if target is None:
        scene.notes.append("おでこと口のまわりが隠れているため、肌は元のプリセットのままです。")
        scene.confidence["skin"] = 0.0
        if scene.face_comparable:
            scene.commit("skin")
        return
    info["target_lab"] = target
    if scene.face_comparable:
        _fit_skin_bare(scene, stable, section, info)
        section.enabled = True
        scene.commit("skin")
        lab = blend.bgr_to_lab(scene.current)
        rendered = _stable_lab(lab, scene.landmarks, scene.roi, scene.skin_ok)
        if scene.tone_trust < _MIN_TONE_TRUST:
            # The preset keeps the base's foundation, but it was not rendered: the later
            # fits compare with this face, and must not take the unexplained skin tone
            # for their own makeup either.
            section.tone_color, section.tone_strength, section.brightness = kept_tone
            if rendered is not None:
                scene.current = blend.lab_to_bgr(lab + np.float32(target - rendered))
                rendered = target
        if rendered is None:
            rendered = target
        residual = float(delta_e00(rendered, target))
        info["residual_de"] = residual
        scene.confidence["skin"] = (0.9 if scene.tone_trust >= 0.75 else 0.55) * (1.0 - _ramp(residual, 2.0, 8.0) * 0.6)
        return
    # One photo: the face's color is the foundation's shade as far as anyone can
    # tell. The strength and the retouch sliders stay as in the base preset.
    hue = math.degrees(math.atan2(float(target[2]), float(target[1])))
    usable = (
        _TONE_HUE_RANGE[0] <= hue <= _TONE_HUE_RANGE[1]
        and float(np.hypot(target[1], target[2])) <= _TONE_MAX_CHROMA
        and float(scene.measurements["photos"].get("clipped_skin_share", 0.0)) <= _MAX_CLIPPED_SKIN
    )
    if usable:
        section.tone_color = _hex_of_lab(target)
    section.enabled = True
    scene.confidence["skin"] = 0.3 if gray_world_applied else 0.25


# --------------------------------------------------------------------------
# Contour and tear bags (bare photo only).
# --------------------------------------------------------------------------

# BGR -> lightness and two opponent chroma axes, weighted (see _FaceTones).
_OPPONENT = np.array(
    [
        [0.114 * 0.4, 0.587 * 0.4, 0.299 * 0.4],
        [-0.75, 0.0, 0.75],
        [0.375, -0.75, 0.375],
    ]
)
# Shrinks areas with a weak response (a highlight on skin near white) instead of
# letting a tiny change there become a strong slider.
_CONTOUR_RIDGE = 0.03
_MAX_CONTOUR = 0.6  # stronger is never natural; it means a different light between the photos
_LUMA_BGR = np.array([0.114, 0.587, 0.299])

_SHADE_AREAS = ("jawline", "cheek_hollow", "nose_shadow", "forehead")
_LIGHT_AREAS = ("nose_bridge", "cheekbone", "chin", "cupids_bow")


def _contour_zone(scene: _Scene) -> np.ndarray:
    """Face skin away from the eyes, brows, lips and nostrils, as a bool mask."""
    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    zone = regions.face_oval_mask(landmarks, roi, expand_px=-0.02 * scale) > 0.5
    for side in regions.SIDES:
        zone &= regions.eye_opening_mask(landmarks, side, roi, expand_px=0.05 * scale) < 0.5
        zone &= regions.brow_mask(landmarks, side, roi, thickness=0.7) < 0.5
    zone &= regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_OUTER], 0.015 * scale), roi) < 0.5
    return zone & scene.skin_ok


def _contour_responses(scene: _Scene, zone: np.ndarray, sigma: float, step: int) -> np.ndarray:
    """Rendered change per contour area at full slider: (samples, 3, areas), blurred."""
    section = scene.settings.contour
    base = _blur(scene.current, sigma)[zone][::step]
    columns = []
    for area in _SHADE_AREAS + _LIGHT_AREAS:
        trial = type(section)(**{**section.__dict__})
        for name in _SHADE_AREAS + _LIGHT_AREAS:
            setattr(trial, name, 1.0 if name == area else 0.0)
        trial.shading = 1.0 if area in _SHADE_AREAS else 0.0
        trial.highlight = 1.0 if area in _LIGHT_AREAS else 0.0
        columns.append(_blur(scene.render("contour", trial), sigma)[zone][::step] - base)
    return np.stack(columns, axis=2)


def _set_contour(scene: _Scene, weights: np.ndarray, info: dict[str, Any]) -> None:
    """Sliders from one weight per area: the strongest area is at 100 % of its tone."""
    section = scene.settings.contour
    named = dict(zip(_SHADE_AREAS + _LIGHT_AREAS, (float(v) for v in weights)))
    info["area_weights"] = named
    mismatch = scene.measurements["photos"]["pose_mismatch"]
    floor = _MIN_CONTOUR_WEIGHT * _noise_factor(scene)
    info["floor"] = floor
    if info.get("explained", 1.0) < _MIN_CONTOUR_EXPLAINED:
        # The areas do not explain the change: it is noise and light, not contour.
        floor = math.inf
    shade = {k: named[k] if named[k] >= floor else 0.0 for k in _SHADE_AREAS}
    light = {k: named[k] if named[k] >= floor else 0.0 for k in _LIGHT_AREAS}
    # Differences of pose come out as shading and highlight: believe them less.
    trust = 1.0 - 0.5 * _ramp(mismatch, 0.01, 0.04)
    section.shading = _clip(max(shade.values()) * trust, 0.0, _MAX_CONTOUR)
    section.highlight = _clip(max(light.values()) * trust, 0.0, _MAX_CONTOUR)
    if section.shading > 0.0:
        for k, v in shade.items():
            setattr(section, k, _clip(v / max(shade.values()), 0.0, 1.0))
    if section.highlight > 0.0:
        for k, v in light.items():
            setattr(section, k, _clip(v / max(light.values()), 0.0, 1.0))
    section.enabled = True
    scene.confidence["contour"] = 0.75 * (1.0 - _ramp(mismatch, 0.01, 0.05) * 0.7)


def _fit_tearbag(scene: _Scene) -> None:
    section = scene.settings.tearbag
    info: dict[str, Any] = {}
    scene.measurements["tearbag"] = info
    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    zone = np.zeros(scene.target.shape[:2], np.float32)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        line = landmarks[s.eye_lower].astype(np.float32) + np.float32([0.0, 0.05 * scale])
        band = regions.polyline_mask(line, 0.11 * scale, roi)
        zone = np.maximum(zone, band * (1.0 - regions.eye_opening_mask(landmarks, side, roi, expand_px=0.02 * scale)))
    mask = (zone > 0.5) & scene.skin_ok
    if mask.sum() < 40:
        scene.notes.append("目の下が隠れているため、涙袋は元のプリセットのままです。")
        scene.confidence["tearbag"] = 0.0
        return
    sigma = 0.006 * scale
    goal = _blur(scene.target_lab, sigma)[mask]

    def cost_of(values: dict[str, float]) -> float:
        trial = type(section)(**{**section.__dict__, **values})
        lab = _blur(blend.bgr_to_lab(scene.render("tearbag", trial)), sigma)[mask]
        return float(np.mean((lab - goal) ** 2 * np.array([1.0, 0.5, 0.5])))

    base_cost = cost_of({"strength": 0.0})
    best = {"strength": 0.0, "width": section.width, "shadow": section.shadow}
    best_cost = base_cost
    for _ in range(2):
        for key, grid in (
            ("strength", (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)),
            ("width", (0.3, 0.5, 0.7, 0.85, 1.0)),
            ("shadow", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        ):
            for value in grid:
                c = cost_of({**best, key: value})
                if c < best_cost - 1e-6:
                    best_cost, best[key] = c, value
    info["cost_without"], info["cost_with"] = base_cost, best_cost
    if best_cost > 0.9 * base_cost:
        best["strength"] = 0.0
    section.strength = best["strength"]
    if section.strength > 0.0:
        section.width = best["width"]
        section.shadow = best["shadow"]
    section.enabled = True
    scene.confidence["tearbag"] = 0.7 * (1.0 - _ramp(scene.measurements["photos"]["pose_mismatch"], 0.01, 0.05) * 0.7)


# --------------------------------------------------------------------------
# Eyebrows.
# --------------------------------------------------------------------------


def _brow_zone(scene: _Scene) -> np.ndarray:
    area = np.zeros(scene.target.shape[:2], np.float32)
    for side in regions.SIDES:
        brow = regions.brow_mask(scene.landmarks, side, scene.roi, thickness=0.6, feather_px=0.01 * scene.scale)
        area = np.maximum(area, brow)
    return (area > 0.5) & scene.unclipped


def _fit_brows_from_contrast(scene: _Scene, zone: np.ndarray) -> None:
    """One photo: darkness and hue of the brow against the skin around it. Her
    natural brow cannot be told from the fill, so this stays a modest estimate."""
    section = scene.settings.brows
    info: dict[str, Any] = {}
    scene.measurements["brows"] = info
    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    core = np.zeros(zone.shape, bool)
    ring = np.zeros(zone.shape, bool)
    for side in regions.SIDES:
        core |= regions.brow_mask(landmarks, side, roi, thickness=-0.2) > 0.5
        wide = regions.brow_mask(landmarks, side, roi, thickness=1.6) > 0.5
        ring |= wide & (regions.brow_mask(landmarks, side, roi, thickness=0.7) < 0.5)
    for side in regions.SIDES:
        ring &= regions.eye_opening_mask(landmarks, side, roi, expand_px=0.05 * scale) < 0.5
    ring &= scene.hair < 0.5
    if core.sum() < 15 or ring.sum() < 15:
        scene.confidence["brows"] = 0.0
        return
    luma = blend.luminance(scene.target)
    skin_luma = float(np.percentile(luma[ring], 60))
    brow_luma = float(np.percentile(luma[core], 40))
    contrast = 1.0 - brow_luma / max(skin_luma, 1e-3)
    info["contrast"] = contrast
    opacity = _NO_BARE_BROW_MAX * _ramp(contrast, *_BROW_CONTRAST)
    dark = scene.target_lab[core & (luma <= np.percentile(luma[core], 30))]
    hair = np.median(dark, axis=0) if len(dark) else None
    section.opacity = opacity
    if hair is not None:
        info["hair_lab"] = hair
        lab = np.array([_clip(float(hair[0]), 15.0, 55.0), hair[1], hair[2]])
        section.color = _hex_of_lab(_within_product_range(lab, _BROW_HUE_RANGE, _BROW_MAX_CHROMA))
    section.enabled = True
    scene.confidence["brows"] = 0.3


def _brow_hidden(scene: _Scene, info: dict[str, Any]) -> None:
    scene.notes.append("眉が前髪などで隠れているため、眉は元のプリセットのままです。")
    scene.confidence["brows"] = 0.0
    info["hidden"] = True


def _band_above_brows(scene: _Scene) -> np.ndarray:
    """Skin just above the brows, as a bool mask."""
    band = np.zeros(scene.target.shape[:2], bool)
    lifted = scene.landmarks - np.float32([0.0, _BROW_RING_SHIFT * scene.scale])
    for side in regions.SIDES:
        band |= regions.brow_mask(lifted, side, scene.roi, thickness=0.3) > 0.5
    for side in regions.SIDES:
        band &= regions.brow_mask(scene.landmarks, side, scene.roi, thickness=0.8) < 0.5
    return band & scene.unclipped


def _dark_above_brows(scene: _Scene) -> bool:
    """Hair just above the brows that the segmentation did not see (works on one photo)."""
    band = _band_above_brows(scene)
    skin = regions.face_oval_mask(scene.landmarks, scene.roi, expand_px=-0.15 * scene.scale) > 0.5
    skin &= scene.skin_ok
    if not band.any() or not skin.any():
        return False
    luma = blend.luminance(scene.target)
    return float(np.median(luma[band])) < _BROW_RING_LUMA * float(np.median(luma[skin]))


def _brow_color(gain: np.ndarray) -> tuple[float, np.ndarray]:
    """(opacity, BGR color) of a brow's relative change. Brow products are browns and grays:
    a change that takes more red than blue is noise on the darkening, so only the darkening
    counts then, and what is left is held inside the range of real products."""
    if float(gain[2]) < float(gain[0]):
        gain = np.full(3, float(np.dot(gain, _LUMA_BGR)))
    # brows.py is plain multiply: gain = opacity * (color - 1).
    opacity, color = canonical_tint(gain, -np.ones(3), np.zeros(3), _lightness_of, _BROW_LIGHTNESS)
    lab = _within_product_range(_lab_of_bgr(color), _BROW_HUE_RANGE, _BROW_MAX_CHROMA)
    return opacity, np.asarray(hex_to_bgr(_hex_of_lab(lab)), np.float64)


def _fit_brows(scene: _Scene) -> None:
    section = scene.settings.brows
    zone = _brow_zone(scene)
    if _visible_brow_share(scene, zone) < 0.5 or scene.eyes_hidden or _dark_above_brows(scene):
        scene.measurements["brows"] = {}
        _brow_hidden(scene, scene.measurements["brows"])
        return
    if not scene.has_bare:
        _fit_brows_from_contrast(scene, zone)
        if scene.confidence.get("brows", 0.0) > 0.0:
            scene.commit("brows")
        return
    info: dict[str, Any] = {}
    scene.measurements["brows"] = info
    sigma = 0.006 * scene.scale
    t = _blur(scene.target, sigma)
    b = _blur(scene.current, sigma)
    delta = t - b
    best: tuple[float, float, float, np.ndarray] | None = None
    empty = float(np.sum(delta[zone] ** 2))
    for thickness in (-0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5):
        for softness in (0.0, 0.25, 0.5, 0.75, 1.0):
            trial = type(section)(enabled=True, color="#000000", opacity=1.0, thickness=thickness, softness=softness)
            out = scene.render("brows", trial)
            weight = _blur(1.0 - out[..., 1] / np.maximum(scene.current[..., 1], 1e-3), sigma)
            gain = _typical_gain(weight, b, delta, zone)
            residual = delta[zone] - weight[zone][:, None] * b[zone] * gain
            cost = float(np.sum(residual**2))
            if best is None or cost < best[0]:
                best = (cost, thickness, softness, gain)
    assert best is not None
    cost, thickness, softness, gain = best
    darkening = -float(np.dot(gain, _LUMA_BGR))
    band = _band_above_brows(scene)
    band_darkening = _relative_darkening(t, b, band)
    info.update(darkening=darkening, residual_share=cost / max(empty, 1e-9), gain=gain, band_darkening=band_darkening)
    if darkening >= _MIN_BROW_CHANGE and (darkening > _MAX_BROW_DARKENING or band_darkening > _BROW_RING_DARKENING):
        # Nothing drawn on a brow is this dark or reaches this far: bangs or glasses the
        # segmentation did not see, in this photo only.
        _brow_hidden(scene, info)
        return
    if not scene.face_comparable and darkening > _MAX_BROW_DARKENING_APART:
        # Between different expressions the brows may simply not line up.
        scene.notes.append("2枚の写真で眉の位置が合わないため、眉は元のプリセットのままです。")
        scene.confidence["brows"] = 0.0
        return
    unexplained = cost > (1.0 - _MIN_BROW_EXPLAINED) * empty
    if darkening >= _MIN_BROW_CHANGE * _noise_factor(scene) and unexplained and not scene.face_comparable:
        # Darker, but not in the shape of the brow: between different expressions the brows
        # may simply not line up, which says nothing about the pencil.
        scene.notes.append("2枚の写真で眉の位置が合わないため、眉は元のプリセットのままです。")
        scene.confidence["brows"] = 0.0
        return
    if darkening < _MIN_BROW_CHANGE * _noise_factor(scene) or unexplained:
        section.opacity = 0.0
        section.enabled = True
        scene.confidence["brows"] = 0.8
        return
    opacity, color = _brow_color(gain)
    section.color = _hex_of_bgr(color)
    section.opacity = opacity
    section.thickness = thickness
    section.softness = softness
    section.enabled = True
    scene.commit("brows")
    scene.confidence["brows"] = 0.8 * (1.0 - _ramp(scene.measurements["photos"]["pose_mismatch"], 0.015, 0.06) * 0.5)


def _visible_brow_share(scene: _Scene, zone: np.ndarray) -> float:
    total = float(zone.sum())
    return float((zone & (scene.hair < 0.75)).sum()) / total if total else 0.0


# --------------------------------------------------------------------------
# Eyeshadow.
# --------------------------------------------------------------------------


def _shadow_zone(scene: _Scene) -> np.ndarray:
    """Where a shadow can be and be measured (off the lashes and the brows), bool."""
    from .effects import eyes

    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    zone = np.zeros(scene.target.shape[:2], bool)
    for side in regions.SIDES:
        axes = eyes._eye_axes(landmarks, side)
        area = regions.polygon_mask(eyes._lid_outline(landmarks, side, axes, scale), roi) > 0.5
        area &= regions.brow_mask(landmarks, side, roi, thickness=0.4) < 0.5
        s = regions.side_indices(side)
        lash_band = regions.polyline_mask(landmarks[s.eye_upper].astype(np.float32), 0.14 * scale, roi)
        area &= lash_band < 0.5
        area &= regions.eye_opening_mask(landmarks, side, roi, expand_px=0.03 * scale) < 0.5
        zone |= area
    return zone & (scene.hair < 0.5) & scene.unclipped


def _fit_eyeshadow(scene: _Scene) -> None:
    from .effects import eyes

    section = scene.settings.eyeshadow
    info: dict[str, Any] = {}
    scene.measurements["eyeshadow"] = info
    zone = _shadow_zone(scene)
    if zone.sum() < 60 or scene.eyes_hidden:
        _keep_base_eyeshadow(scene, "まぶたが隠れているか小さすぎるため、アイシャドウは元のプリセットのままです。")
        return
    sigma = 0.012 * scene.scale
    t = _blur(scene.target, sigma)
    b = _blur(scene.current, sigma)
    delta = t - b
    flat = np.full_like(scene.target, 0.5)
    full = eyes._tint(flat[:1, :1], (0.0, 0.0, 0.0), np.ones((1, 1), np.float32))[0, 0, 0] - 0.5
    empty = float(np.sum(delta[zone] ** 2))
    best: tuple[float, float, float, np.ndarray] | None = None
    for height in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for softness in (0.2, 0.4, 0.6, 0.8, 1.0):
            trial = type(section)(
                enabled=True, color="#000000", opacity=1.0, height=height, softness=softness, shimmer=0.0
            )
            weight = (scene.render("eyeshadow", trial, flat)[..., 0] - 0.5) / float(full)
            weight = _blur(weight, sigma)
            gain = _typical_gain(weight, b, delta, zone)
            residual = delta[zone] - weight[zone][:, None] * b[zone] * gain
            cost = float(np.sum(residual**2))
            if best is None or cost < best[0]:
                best = (cost, height, softness, gain)
    assert best is not None
    cost, height, softness, gain = best
    gain = _limit_by_lash_band(scene, flat, full, section, height, softness, gain, t, b, sigma)
    lid = b[zone].mean(axis=0)
    change = float(np.linalg.norm(_lab_of_bgr(lid * (1.0 + gain)) - _lab_of_bgr(lid)))
    info.update(change=change, gain=gain, height=height, softness=softness, residual_share=cost / max(empty, 1e-9))
    section.enabled = True
    lighter = float(np.dot(gain, _LUMA_BGR)) > _MAX_SHADOW_LIGHTENING
    unexplained = scene.eyes_comparable and cost > (1.0 - _MIN_EXPLAINED) * empty
    # On one photo only a lid that was taken off counts: the rest differs from the
    # estimate by the inpainted lash line, not by makeup.
    no_estimate = not scene.eyes_comparable and not scene.measurements.get("approximate_bare", {}).get("lid_shifted")
    if change < _MIN_SHADOW_CHANGE * _noise_factor(scene) or lighter or unexplained or no_estimate:
        section.opacity = 0.0
        section.shimmer = 0.0
        scene.confidence["eyeshadow"] = 0.8 if scene.eyes_comparable else 0.4
        return
    at_0, at_1 = _response_line(eyes._tint, lid)
    gain = gain - max(float(gain.max()), 0.0)  # a shadow tints and darkens, see the blush
    opacity, color = canonical_tint(gain, at_0, at_1, _lightness_of, _SHADOW_LIGHTNESS)
    lab = _lab_of_bgr(color)
    info.update(raw_opacity=opacity, raw_lab=lab)
    if float(lab[0]) < _SHADOW_MIN_LIGHTNESS and opacity >= 0.9:
        # Black at full strength: something dark in front of the lid, not a shadow.
        _keep_base_eyeshadow(scene, "まぶたの変化がアイシャドウらしくないため、アイシャドウは元のプリセットのままです。")
        return
    section.color = _hex_of_lab(_within_product_range(lab, _SHADOW_HUE_RANGE, _SHADOW_MAX_CHROMA))
    section.opacity = opacity
    section.height = height
    section.softness = softness
    section.shimmer = 0.0
    if scene.eyes_comparable:
        _fit_shimmer(scene, section, zone, sigma)
    scene.commit("eyeshadow")
    mismatch = scene.measurements["photos"]["pose_mismatch"]
    measured = 0.85 * (1.0 - _ramp(mismatch, 0.015, 0.06) * 0.5)
    scene.confidence["eyeshadow"] = measured if scene.eyes_comparable else 0.45


def _keep_base_eyeshadow(scene: _Scene, note: str) -> None:
    scene.notes.append(note)
    scene.confidence["eyeshadow"] = 0.0
    scene.commit("eyeshadow")


def _limit_by_lash_band(
    scene: _Scene,
    flat: np.ndarray,
    full: float,
    section: Any,
    height: float,
    softness: float,
    gain: np.ndarray,
    t: np.ndarray,
    b: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """The fit leaves out the lash line, where a liner may be, but there the shadow is at its
    deepest: it must not come out darker than the photo, or no room is left for the liner."""
    band = np.zeros(scene.target.shape[:2], bool)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        band |= regions.polyline_mask(scene.landmarks[s.eye_upper].astype(np.float32), 0.14 * scene.scale, scene.roi) > 0.5
        band &= regions.eye_opening_mask(scene.landmarks, side, scene.roi, expand_px=0.5) < 0.2
    band &= scene.unclipped
    if not band.any():
        return gain
    trial = type(section)(enabled=True, color="#000000", opacity=1.0, height=height, softness=softness, shimmer=0.0)
    weight = _blur((scene.render("eyeshadow", trial, flat)[..., 0] - 0.5) / float(full), sigma)
    predicted = float(-np.mean((weight[band][:, None] * b[band] * gain) @ _LUMA_BGR))
    available = float(np.mean((b[band] - t[band]) @ _LUMA_BGR))
    if predicted <= 0.0 or available >= predicted:
        return gain
    return gain * max(available / predicted, _MIN_SHADOW_SCALE)


def _fit_shimmer(scene: _Scene, section: Any, zone: np.ndarray, sigma: float) -> None:
    goal = _blur(scene.target_lab, sigma)[zone]

    def cost(value: float) -> float:
        section.shimmer = value
        lab = _blur(blend.bgr_to_lab(scene.render("eyeshadow", section)), sigma)[zone]
        return float(np.mean((lab - goal) ** 2))

    none = cost(0.0)
    value, best = _grid_search((0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0), cost)
    section.shimmer = value if best < 0.97 * none else 0.0


# --------------------------------------------------------------------------
# Eyeliner.
# --------------------------------------------------------------------------


def _liner_zones(scene: _Scene) -> tuple[np.ndarray, np.ndarray]:
    """(upper band with the wing, lower band), off the eye opening, as bool masks."""
    from .effects import eyes

    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    upper = np.zeros(scene.target.shape[:2], bool)
    lower = np.zeros(scene.target.shape[:2], bool)
    for side in regions.SIDES:
        axes = eyes._eye_axes(landmarks, side)
        s = regions.side_indices(side)
        lash = landmarks[s.eye_upper].astype(np.float32)
        tip = axes.outer + axes.direction(eyes._WING_ANGLE_DEG) * np.float32(0.26 * scale)
        band = regions.polyline_mask(np.vstack([lash, tip]), 0.14 * scale, roi) > 0.5
        opening = regions.eye_opening_mask(landmarks, side, roi, expand_px=0.5) > 0.2
        upper |= band & ~opening
        under = regions.polyline_mask(landmarks[s.eye_lower].astype(np.float32), 0.06 * scale, roi) > 0.5
        lower |= under & ~opening
    lower &= ~upper
    return upper & scene.unclipped, lower & scene.unclipped


def _liner_weight(
    scene: _Scene, flat: np.ndarray, section: Any, thickness: float, wing: float, softness: float, lower: float
) -> np.ndarray:
    """The liner's mask at full opacity: rendered in black on a flat gray image."""
    trial = type(section)(
        enabled=True, color="#000000", opacity=1.0, thickness=thickness, wing=wing, lower=lower, softness=softness
    )
    return 1.0 - scene.render("eyeliner", trial, flat)[..., 0] * np.float32(2.0)


def _relative_darkening(target: np.ndarray, base: np.ndarray, zone: np.ndarray) -> float:
    if not zone.any():
        return 0.0
    return float(np.mean(base[zone] - target[zone]) / max(float(np.mean(base[zone])), 1e-3))


def _inked(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    """(N, 3) pixels fully covered by the liner's ink of ``color``."""
    from .effects import eyes

    weight = np.ones((len(pixels), 1), np.float32)
    return eyes._ink(pixels.reshape(-1, 1, 3), tuple(color), weight).reshape(-1, 3)


def _wing_depth(scene: _Scene, flat: np.ndarray) -> float:
    """Darkness of the photo where a wing would be, relative to the skin around it."""
    section = scene.settings.eyeliner
    with_wing = _liner_weight(scene, flat, section, 0.3, 0.6, 0.2, 0.0)
    no_wing = _liner_weight(scene, flat, section, 0.3, 0.0, 0.2, 0.0)
    wing = (with_wing - no_wing) > 0.5
    if not wing.any():
        return 0.0
    reach = _liner_weight(scene, flat, section, 0.3, 1.0, 0.2, 0.0) > 0.05
    size = max(3, int(0.14 * scene.scale)) | 1
    ring = cv2.dilate(wing.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
    ring &= ~reach
    for side in regions.SIDES:
        ring &= regions.eye_opening_mask(scene.landmarks, side, scene.roi, expand_px=0.03 * scene.scale) < 0.5
    if not ring.any():
        return 0.0
    luma = blend.luminance(_blur(scene.target, 0.004 * scene.scale))
    return 1.0 - float(np.median(luma[wing])) / max(float(np.percentile(luma[ring], 70)), 1e-3)


def _fit_eyeliner(scene: _Scene) -> None:
    from .effects import eyes

    section = scene.settings.eyeliner
    info: dict[str, Any] = {}
    scene.measurements["eyeliner"] = info
    upper, lower_zone = _liner_zones(scene)
    if upper.sum() < 30 or scene.eyes_hidden:
        scene.notes.append("目元が隠れているか小さすぎるため、アイライナーは元のプリセットのままです。")
        scene.confidence["eyeliner"] = 0.0
        scene.commit("eyeliner")
        return
    sigma = max(0.7, 0.008 * scene.scale)
    t = _blur(scene.target, sigma)
    b = _blur(scene.current, sigma)
    flat = np.full_like(scene.target, 0.5)
    color = np.asarray(hex_to_bgr(section.color), np.float64)
    empty = float(((t[upper] - b[upper]) ** 2).sum())

    def predict(weight: np.ndarray, ink_color: np.ndarray) -> tuple[float, float]:
        """(opacity, squared error over the upper band) for a liner mask."""
        ink = _inked(b[upper], ink_color)
        direction = (ink - b[upper]) * weight[upper][:, None]
        change = t[upper] - b[upper]
        opacity = _clip(float((direction * change).sum() / max(float((direction * direction).sum()), 1e-9)), 0.0, 1.0)
        if opacity < _MIN_LINER_OPACITY:  # a broad faint stroke is a leftover of the shadow, not a liner
            return 0.0, empty
        return opacity, float(((change - opacity * direction) ** 2).sum())

    thickness, wing, softness = 0.3, 0.0, 0.4
    best = empty

    def blurred(th: float, wi: float, so: float) -> np.ndarray:
        return _blur(_liner_weight(scene, flat, section, th, wi, so, 0.0), sigma)

    for _ in range(2):
        for th in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
            for so in (0.0, 0.3, 0.6, 1.0):
                _, c = predict(blurred(th, wing, so), color)
                if c < best:
                    best, thickness, softness = c, th, so
        for wi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
            _, c = predict(blurred(thickness, wi, softness), color)
            if c < best:
                best, wing = c, wi
    weight = blurred(thickness, wing, softness)
    opacity, best = predict(weight, color)

    # Color of the ink, where the stroke is solid.
    solid = upper & (weight > 0.6)
    if opacity > 0.2 and solid.sum() >= 25:
        share = opacity * weight[solid][:, None]
        ink = (t[solid] - (1.0 - share) * b[solid]) / np.maximum(share, 1e-3)
        estimate = np.median(ink / np.maximum(0.7 + 0.3 * b[solid], 1e-3), axis=0)
        estimate = np.clip(estimate, 0.02, 1.0)
        if _lightness_of(estimate) <= 50.0:
            color = estimate
            opacity, best = predict(weight, color)
    width = eyes._LINER_WIDTH_BASE + eyes._LINER_WIDTH_RANGE * thickness
    ink_amount = opacity * width
    # Half of the mask's peak: a thin stroke on a small face never reaches 0.5 after the blur.
    stroke = upper & (weight > 0.5 * max(float(weight[upper].max()), 1e-3))
    darkening = _relative_darkening(t, b, stroke)
    size = max(3, int(0.2 * scene.scale)) | 1
    around = cv2.dilate(stroke.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
    around &= upper & (weight < 0.15)
    darkening_around = _relative_darkening(t, b, around)
    info["darkening_around"] = darkening_around
    info.update(
        thickness=thickness,
        wing=wing,
        softness=softness,
        opacity=opacity,
        ink=ink_amount,
        darkening=darkening,
        residual_share=best / max(empty, 1e-9),
    )

    # Lower line.
    lower_value = 0.0
    if lower_zone.sum() >= 15:
        change = t[lower_zone] - b[lower_zone]
        ink_low = _inked(b[lower_zone], color)
        empty_low = float((change**2).sum())
        best_low = empty_low
        for value in (0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0):
            w = _blur(_liner_weight(scene, flat, section, thickness, wing, softness, value), sigma)[lower_zone]
            direction = (ink_low - b[lower_zone]) * w[:, None] * max(opacity, 0.5)
            c = float(((change - direction) ** 2).sum())
            if c < best_low:
                best_low, lower_value = c, value
        info["lower_cost_share"] = best_low / max(empty_low, 1e-9)
        if best_low > 0.93 * empty_low:
            lower_value = 0.0
    info["lower"] = lower_value

    section.enabled = True
    if scene.eyes_comparable:
        stands_out = darkening - darkening_around >= _MIN_LINER_CONTRAST
        if (
            best > (1.0 - _MIN_LINER_EXPLAINED) * empty
            or opacity < _MIN_LINER_OPACITY
            or not stands_out
            or ink_amount < _MIN_LINER_INK
        ):
            section.opacity = 0.0
            section.wing = 0.0
            section.lower = 0.0
            scene.confidence["eyeliner"] = 0.75
            return
    else:
        depth = _wing_depth(scene, flat)
        info["wing_depth"] = depth
        if depth < _LINER_WING_DEPTH[0]:
            section.opacity = 0.0
            section.wing = 0.0
            section.lower = 0.0
            scene.confidence["eyeliner"] = 0.3
            return
        opacity = _NO_BARE_LINER_OPACITY
        thickness = _clip(thickness - _NATURAL_LINER_THICKNESS, _NO_BARE_LINER_THICKNESS[0], _NO_BARE_LINER_THICKNESS[1])
        wing = wing * _ramp(depth, *_LINER_WING_DEPTH)
        lower_value = 0.0
    if thickness < _MIN_LINER_THICKNESS:
        # The same ink as a line the slider can work with: full opacity on a hair-thin line
        # would turn into a heavy line as soon as the thickness is moved.
        wider = eyes._LINER_WIDTH_BASE + eyes._LINER_WIDTH_RANGE * _MIN_LINER_THICKNESS
        opacity = min(ink_amount / wider, opacity)
        thickness = _MIN_LINER_THICKNESS
    section.color = _hex_of_bgr(color)
    section.opacity = opacity
    section.thickness = thickness
    section.wing = wing
    section.lower = lower_value
    section.softness = softness
    scene.commit("eyeliner")
    resolution = _ramp(scene.scale, 40.0, 90.0)
    if scene.eyes_comparable:
        mismatch = scene.measurements["photos"]["pose_mismatch"]
        scene.confidence["eyeliner"] = 0.8 * (0.5 + 0.5 * resolution) * (1.0 - _ramp(mismatch, 0.01, 0.05) * 0.6)
    else:
        scene.confidence["eyeliner"] = 0.45 * (0.5 + 0.5 * resolution)


# --------------------------------------------------------------------------
# Blush.
# --------------------------------------------------------------------------


def _blush_unit(
    scene: _Scene, flat: np.ndarray, full: float, section: Any, shape: str, size: float, ox: float, oy: float, softness: float
) -> np.ndarray:
    """The blush's mask at full opacity: rendered in black on a flat gray image."""
    trial = type(section)(
        enabled=True, color="#000000", opacity=1.0, shape=shape, size=size, offset_x=ox, offset_y=oy, softness=softness
    )
    return (scene.render("blush", trial, flat)[..., 0] - 0.5) / full


def _solve_bounded(gram: np.ndarray, rhs: np.ndarray, free: int, sweeps: int = 60) -> np.ndarray:
    """Coordinate descent on normal equations: the first ``free`` unknowns are
    unconstrained, the others must not go negative."""
    x = np.zeros(len(rhs))
    diagonal = np.maximum(np.diag(gram), 1e-12)
    for _ in range(sweeps):
        for i in range(len(x)):
            value = x[i] + (rhs[i] - float(gram[i] @ x)) / diagonal[i]
            x[i] = value if i < free else max(value, 0.0)
    return x


class _FaceTones:
    """Least-squares split of the color change over the face between the blush (three
    free per-channel gains on its mask) and the contour areas (non-negative weights).

    Rows are weighted in an opponent color space: lightness also changes with the
    light, chroma much less, so chroma counts more. With a bare photo, a plane per row
    across the face is projected out of everything first: a head turned a little or a
    lamp moved changes the light that way, and neither blush nor contour looks like it.
    """

    def __init__(self, scene: _Scene, zone: np.ndarray, sigma: float, step: int, measure_contour: bool) -> None:
        self.zone, self.sigma, self.step = zone, sigma, step
        self.base = _blur(scene.current, sigma)[zone][::step].astype(np.float64)
        change = _blur(scene.target, sigma)[zone][::step].astype(np.float64) - self.base
        self.light: np.ndarray | None = None
        if measure_contour:
            ys, xs = np.nonzero(zone)
            nose = scene.landmarks[regions.NOSE_TIP]
            plane = np.stack(
                [np.ones(len(xs)), (xs - float(nose[0])) / scene.scale, (ys - float(nose[1])) / scene.scale], axis=1
            )
            self.light = np.linalg.qr(plane[::step])[0]
        self.y = self._without_light(change @ _OPPONENT.T)
        self.yty = float((self.y * self.y).sum())
        self.columns = 0
        if measure_contour:
            responses = _contour_responses(scene, zone, sigma, step)  # (n, 3, areas)
            self.contour = self._without_light(np.einsum("ij,nja->nia", _OPPONENT, responses.astype(np.float64)))
            self.columns = self.contour.shape[2]
            self.contour_gram = np.einsum("nia,nib->ab", self.contour, self.contour)
            self.contour_rhs = np.einsum("nia,ni->a", self.contour, self.y)
            self.ridge = _CONTOUR_RIDGE * float(np.max(np.diag(self.contour_gram), initial=0.0))

    def _without_light(self, values: np.ndarray) -> np.ndarray:
        """``values`` (samples first) minus their projection on the light planes."""
        if self.light is None:
            return values
        return values - np.tensordot(self.light, np.tensordot(self.light.T, values, axes=1), axes=1)

    def solve(self, mask: np.ndarray | None) -> tuple[float, np.ndarray, np.ndarray]:
        """(squared error, blush gains, contour weights) for a blush mask (None: no blush)."""
        free = 0
        if mask is None:
            gram = np.zeros((self.columns, self.columns))
            rhs = np.zeros(self.columns)
        else:
            free = 3
            weight = _blur(mask, self.sigma)[self.zone][:: self.step].astype(np.float64)
            tint = self._without_light(weight[:, None, None] * self.base[:, None, :] * _OPPONENT[None, :, :])  # (n, 3, 3)
            gram = np.zeros((3 + self.columns, 3 + self.columns))
            rhs = np.zeros(3 + self.columns)
            gram[:3, :3] = np.einsum("nia,nib->ab", tint, tint)
            rhs[:3] = np.einsum("nia,ni->a", tint, self.y)
            if self.columns:
                cross = np.einsum("nia,nib->ab", tint, self.contour)
                gram[:3, 3:] = cross
                gram[3:, :3] = cross.T
        if self.columns:
            gram[free:, free:] = self.contour_gram + np.eye(self.columns) * self.ridge
            rhs[free:] = self.contour_rhs
        if len(rhs) == 0:
            return self.yty, np.zeros(3), np.zeros(0)
        x = _solve_bounded(gram, rhs, free)
        cost = self.yty - 2.0 * float(x @ rhs) + float(x @ gram @ x)
        gains = x[:3] if free else np.zeros(3)
        return cost, gains, x[free:]


def _limited_hue(lab: np.ndarray, hue_range: tuple[float, float]) -> np.ndarray:
    """The Lab color with its hue angle moved into ``hue_range`` (degrees, -180..180)."""
    chroma = float(np.hypot(lab[1], lab[2]))
    hue = _clip(math.degrees(math.atan2(float(lab[2]), float(lab[1]))), *hue_range)
    return np.array([lab[0], chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))])


def _within_product_range(lab: np.ndarray, hue_range: tuple[float, float], max_chroma: float) -> np.ndarray:
    """The Lab color moved into a product family's hue range and chroma limit."""
    limited = _limited_hue(np.asarray(lab, np.float64), hue_range)
    chroma = float(np.hypot(limited[1], limited[2]))
    if chroma > max_chroma:
        limited[1:] *= max_chroma / chroma
    return limited


def _blush_amplitude(cheek: np.ndarray, gains: np.ndarray) -> float:
    """Lab a/b change that gains make on a cheek color at full mask."""
    before = _lab_of_bgr(cheek)
    after = _lab_of_bgr(np.clip(cheek * (1.0 + gains), 0.0, 1.0))
    return float(np.hypot(after[1] - before[1], after[2] - before[2]))


def _fit_cheeks_and_contour(scene: _Scene) -> None:
    """Blush and (with a bare photo) contour together; see ``_FaceTones``.

    Only the contour is committed here (it comes first in the pipeline); the
    blush's sliders are stored and ``_fit_all`` commits them in their turn.
    """
    from .effects import blush as blush_module

    blush = scene.settings.blush
    base_opacity = blush.opacity
    info: dict[str, Any] = {}
    scene.measurements["blush"] = info
    flat = np.full_like(scene.target, 0.5)
    full = float(blush_module._flush(flat[:1, :1], (0.0, 0.0, 0.0), np.ones((1, 1), np.float32))[0, 0, 0] - 0.5)
    cheeks = np.zeros(scene.target.shape[:2], bool)
    for ox, oy in ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)):
        cheeks |= _blush_unit(scene, flat, full, blush, "oval", 1.6, ox, oy, 1.0) > 0.03
    for side in regions.SIDES:
        cheeks &= regions.eye_opening_mask(scene.landmarks, side, scene.roi, expand_px=0.12 * scene.scale) < 0.5
    visible = cheeks & scene.skin_ok & (scene.hair < 0.5)
    visible_share = float(visible.sum()) / max(float(cheeks.sum()), 1.0)
    info["visible_share"] = visible_share
    cheeks = visible
    can_measure_blush = cheeks.sum() >= 200 and visible_share >= _MIN_VISIBLE_CHEEKS
    if not can_measure_blush:
        scene.notes.append("頬が隠れているか白飛びしているため、チークは元のプリセットのままです。")
        scene.confidence["blush"] = 0.0
    sigma = 0.03 * scene.scale
    step = max(1, int(scene.scale // 20))
    measure_contour = False
    zone = cheeks
    if scene.has_bare:
        contour_zone = _contour_zone(scene)
        if not scene.face_comparable:
            scene.confidence["contour"] = 0.0
        elif contour_zone.sum() >= 400:
            measure_contour, zone = True, contour_zone
        else:
            scene.notes.append("顔が隠れているため、シェーディング・ハイライトは元のプリセットのままです。")
            scene.confidence["contour"] = 0.0
            scene.commit("contour")
    if zone.sum() < 50:
        return
    tones = _FaceTones(scene, zone, sigma, step, measure_contour)
    cheek = _blur(scene.current, sigma)[cheeks].mean(axis=0) if cheeks.any() else np.full(3, 0.5)

    def mask_of(state: dict[str, Any]) -> np.ndarray:
        return _blush_unit(
            scene, flat, full, blush, state["shape"], state["size"], state["ox"], state["oy"], state["softness"]
        )

    state: dict[str, Any] = {"shape": "oval", "size": 1.0, "ox": 0.0, "oy": 0.0, "softness": 0.7}
    gains = np.zeros(3)
    weights = np.zeros(tones.columns)
    has_blush = False
    if can_measure_blush:
        _, gains, weights = tones.solve(mask_of(state))
        amplitude = _blush_amplitude(cheek, gains)
        info["first_amplitude"] = amplitude
        has_blush = amplitude >= 0.6 * _MIN_BLUSH_AMPLITUDE * _noise_factor(scene)
    if has_blush:
        best_cost = math.inf
        best_state = state
        sizes = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6)
        offsets = (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)
        softs = (0.3, 0.5, 0.7, 0.9)
        for shape in ("oval", "round", "lifted"):
            trial = {**state, "shape": shape, "size": 1.0, "ox": 0.0, "oy": 0.0, "softness": 0.7}
            cost = tones.solve(mask_of(trial))[0]
            for _ in range(2):
                for key, grid in (("size", sizes), ("ox", offsets), ("oy", offsets), ("softness", softs)):
                    for value in grid:
                        candidate = {**trial, key: value}
                        c = tones.solve(mask_of(candidate))[0]
                        if c < cost - 1e-9:
                            cost, trial = c, candidate
            if cost < best_cost:
                best_cost, best_state = cost, trial
        state = best_state
        _, gains, weights = tones.solve(mask_of(state))
        amplitude = _blush_amplitude(cheek, gains)
        info.update(amplitude=amplitude, gains=gains, geometry=dict(state))
        has_blush = amplitude >= _MIN_BLUSH_AMPLITUDE * _noise_factor(scene)
    if not has_blush and measure_contour:
        _, gains, weights = tones.solve(None)
    if measure_contour:
        contour_info: dict[str, Any] = {"energy": tones.yty / max(len(tones.y), 1)}
        scene.measurements["contour"] = contour_info
        contour_info["explained"] = 1.0 - tones.solve(mask_of(state) if has_blush else None)[0] / max(tones.yty, 1e-9)
        _set_contour(scene, weights, contour_info)
        scene.commit("contour")
    if not can_measure_blush:
        return
    blush.enabled = True
    if not has_blush:
        blush.opacity = 0.0
        scene.confidence["blush"] = 0.8 if scene.face_comparable else 0.4
        return
    at_0, at_1 = _response_line(blush_module._flush, cheek)
    # Blush tints and darkens. A lighter cheek than bare is a change of light or a
    # smile: only the tint is kept.
    gains = gains - max(float(gains.max()), 0.0)
    opacity, color = canonical_tint(gains, at_0, at_1, _chroma_of, _BLUSH_CHROMA)
    lab = _limited_hue(_lab_of_bgr(color), _BLUSH_HUE_RANGE)
    darkest_red = float(at_0[2] + _BLUSH_MIN_RED * (at_1[2] - at_0[2]))
    info.update(raw_opacity=opacity, raw_lab=lab, darkest_red=darkest_red)
    if (
        float(gains[2]) < darkest_red
        or float(lab[0]) < _BLUSH_MIN_LIGHTNESS
        or float(np.hypot(lab[1], lab[2])) < _BLUSH_MIN_CHROMA
    ):
        scene.notes.append("頬の変化がチークらしくないため、チークは元のプリセットのままです（影や照明の違いかもしれません）。")
        scene.confidence["blush"] = 0.0
        blush.opacity = base_opacity
        return
    blush.color = _hex_of_lab(lab)
    blush.opacity = opacity
    blush.shape = state["shape"]
    blush.size = state["size"]
    blush.offset_x = state["ox"]
    blush.offset_y = state["oy"]
    blush.softness = state["softness"]
    if scene.face_comparable:
        mismatch = scene.measurements["photos"]["pose_mismatch"]
        scene.confidence["blush"] = 0.85 * (1.0 - _ramp(mismatch, 0.015, 0.06) * 0.5)
    else:
        scene.confidence["blush"] = 0.5


# --------------------------------------------------------------------------
# Lips.
# --------------------------------------------------------------------------


def _lip_core(scene: _Scene) -> np.ndarray:
    """Lip pixels well inside both lip lines, as a float32 0/1 mask."""
    core = regions.lips_mask(scene.landmarks, scene.roi, outline=-0.12) > 0.9
    grown = regions.expand_polygon(scene.landmarks[regions.LIPS_INNER], 0.015 * scene.scale)
    core &= regions.polygon_mask(grown, scene.roi) < 0.01
    return (core & scene.unclipped).astype(np.float32)


def _solve_lip_color(
    scene: _Scene, section: Any, core: np.ndarray, goal: np.ndarray, start: np.ndarray, sweeps: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Color (Lab) with which the rendered lips' trimmed median reaches ``goal``;
    also returns the final render. The opacity and finish stay as in ``section``."""
    color = start.copy()
    rendered = scene.current
    gains = np.array([_LIP_LIGHTNESS_FOLLOW, 1.0, 1.0]) * max(section.opacity, 0.05)
    for _ in range(sweeps):
        section.color = _hex_of_lab(color)
        rendered = scene.render("lips", section)
        lab = _zone_lab(blend.bgr_to_lab(rendered), core)
        if lab is None:
            break
        miss = goal - lab
        if float(np.abs(miss).max()) < 0.3:
            break
        color = color + miss / gains
        color[0] = _clip(float(color[0]), 5.0, 95.0)
        color[1:] = np.clip(color[1:], -60.0, 90.0)
    return color, rendered


def _lip_spread(lab: np.ndarray, core: np.ndarray, sigma: float) -> np.ndarray:
    blurred = _blur(lab[..., 1:], sigma)
    return np.array([_robust_std(blurred[..., i][core > 0.5]) for i in range(2)])


def _highlight_profile(lab: np.ndarray, core: np.ndarray, sigma: float) -> np.ndarray:
    values = _blur(lab[..., 0], sigma)[core > 0.5]
    p = np.percentile(values, (50, 75, 90, 97))
    return p[1:] - p[0]


def _fit_lips(scene: _Scene) -> None:
    section = scene.settings.lips
    info: dict[str, Any] = {}
    scene.measurements["lips"] = info
    core = _lip_core(scene)
    if core.sum() < 25 or _visible_share(scene, core) < 0.5 or scene.lips_blown:
        scene.notes.append("唇が隠れているか小さすぎる、または白飛びしているため、リップは元のプリセットのままです。")
        scene.confidence["lips"] = 0.0
        return
    goal = _zone_lab(scene.target_lab, core)
    current_lab = blend.bgr_to_lab(scene.current)
    bare = _zone_lab(current_lab, core)
    assert goal is not None and bare is not None
    change = float(delta_e00(goal, bare))
    chroma_change = float(np.hypot(*(goal[1:] - bare[1:])))
    lightness_change = abs(float(goal[0] - bare[0]))
    info.update(target_lab=goal, bare_lab=bare, change=change, chroma_change=chroma_change)
    section.enabled = True
    mouth_change = float(scene.measurements["photos"].get("mouth_state_change", 0.0))
    noise = _noise_factor(scene) + _MOUTH_NOISE * max(mouth_change - _MOUTH_FREE, 0.0)
    # Lips also get lighter or darker when the mouth moves; lipstick shows in the chroma,
    # a clear gloss in the highlights.
    fine = 0.006 * scene.scale
    shine = float((_highlight_profile(scene.target_lab, core, fine) - _highlight_profile(current_lab, core, fine))[2])
    info.update(shine_change=shine, lightness_change=lightness_change, noise=noise)
    chroma_needed = min(_MIN_LIP_CHROMA * noise, _ALWAYS_LIPSTICK_CHROMA)
    if chroma_change < chroma_needed and change < _MIN_LIP_CHANGE * noise and shine < _MIN_LIP_SHINE * noise:
        section.opacity = 0.0
        scene.confidence["lips"] = 0.85 if scene.has_bare else 0.45
        return
    sigma = 0.012 * scene.scale
    # JPEG chroma bleeds a couple of pixels over the lip line: the spread is measured inside it.
    inner = cv2.erode(core, np.ones((5, 5), np.uint8))
    spread_zone = inner if (inner > 0.5).sum() >= _MIN_SPREAD_PIXELS else core
    spread_goal = _lip_spread(scene.target_lab, spread_zone, sigma)
    spread_bare = _lip_spread(current_lab, spread_zone, sigma)
    info.update(spread_target=spread_goal, spread_bare=spread_bare)

    section.finish = "satin"
    section.gloss = 0.3
    section.outline = 0.0
    section.feather = 0.4
    mid = _LIP_LIGHTNESS_FOLLOW

    def start_for(opacity: float) -> np.ndarray:
        return bare + (goal - bare) / (np.array([mid, 1.0, 1.0]) * opacity)

    if scene.has_bare and float(spread_bare.max()) >= _MIN_LIP_SPREAD:
        goal_blur = _blur(scene.target_lab, sigma)
        best_cost, best_opacity = math.inf, _DEFAULT_LIP_OPACITY
        costs: dict[float, float] = {}
        for opacity in _LIP_OPACITIES:
            section.opacity = opacity
            _, rendered = _solve_lip_color(scene, section, core, goal, start_for(opacity), sweeps=3)
            lab = blend.bgr_to_lab(rendered)
            spread = _lip_spread(lab, spread_zone, sigma)
            pixels = _blur(lab, sigma)[core > 0.5] - goal_blur[core > 0.5]
            cost = float(np.sqrt(np.mean(pixels[:, 1:] ** 2))) + float(np.abs(spread - spread_goal).sum())
            cost += _LIP_OPACITY_PRIOR * abs(opacity - _DEFAULT_LIP_OPACITY)
            # A deep lipstick: at a low opacity no color gets dark enough.
            reached = _zone_lab(lab, core)
            if reached is not None:
                cost += abs(float(reached[0]) - float(goal[0]))
            costs[opacity] = cost
            if cost < best_cost:
                best_cost, best_opacity = cost, opacity
        info["opacity_costs"] = costs
        opacity = best_opacity
    elif scene.has_bare:
        opacity = _DEFAULT_LIP_OPACITY
    else:
        distance = float(scene.measurements["approximate_bare"].get("lip_distance", 0.0))
        # Not lower than this: the color would have to make up for it and turn extreme.
        opacity = max(_NO_BARE_LIP_OPACITY * _ramp(distance, _NATURAL_LIP_MARGIN, _NATURAL_LIP_FULL), _MIN_NO_BARE_LIP_OPACITY)
    section.opacity = opacity
    color, rendered = _solve_lip_color(scene, section, core, goal, start_for(opacity))

    # Finish: how the highlights compare.
    profile_goal = _highlight_profile(scene.target_lab, core, 0.006 * scene.scale)
    if scene.has_bare:
        best_finish, best_gloss, best_cost = "satin", section.gloss, math.inf
        for finish, gloss in (
            ("matte", 0.0),
            *(("satin", g) for g in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)),
            *(("gloss", g) for g in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0)),
        ):
            section.finish, section.gloss = finish, gloss
            lab = blend.bgr_to_lab(scene.render("lips", section))
            cost = float(np.abs(_highlight_profile(lab, core, 0.006 * scene.scale) - profile_goal).sum())
            fine = 0.006 * scene.scale
            diff = _blur(lab[..., 0], fine)[core > 0.5] - _blur(scene.target_lab[..., 0], fine)[core > 0.5]
            cost += 0.5 * float(np.sqrt(np.mean((diff - np.median(diff)) ** 2)))
            if cost < best_cost:
                best_finish, best_gloss, best_cost = finish, gloss, cost
        section.finish, section.gloss = best_finish, best_gloss
    else:
        highlight = float(profile_goal[2])
        if highlight < _MATTE_HIGHLIGHT:
            section.finish, section.gloss = "matte", 0.0
        elif highlight < _GLOSS_HIGHLIGHT:
            section.finish, section.gloss = "satin", 0.3
        else:
            section.finish, section.gloss = "gloss", 0.5
    info["highlight_goal"] = profile_goal
    color, rendered = _solve_lip_color(scene, section, core, goal, color)

    if scene.face_comparable:
        _fit_lip_edge(scene, section)
        color, rendered = _solve_lip_color(scene, section, core, goal, color, sweeps=2)
    section.color = _hex_of_lab(color)
    lab = _zone_lab(blend.bgr_to_lab(rendered), core)
    residual = float(delta_e00(lab, goal)) if lab is not None else math.inf
    info["residual_de"] = residual
    quality = 1.0 - _ramp(residual, 2.0, 8.0) * 0.6
    if scene.has_bare:
        mismatch = scene.measurements["photos"]["pose_mismatch"]
        moved = 1.0 - _ramp(mouth_change, 0.05, 0.25) * 0.5
        scene.confidence["lips"] = 0.9 * quality * moved * (1.0 - _ramp(mismatch, 0.015, 0.06) * 0.4)
    else:
        scene.confidence["lips"] = 0.55 * quality
    scene.current = rendered


def _fit_lip_edge(scene: _Scene, section: Any) -> None:
    """Outline and feather from the band around the lip line."""
    landmarks, scale, roi = scene.landmarks, scene.scale, scene.roi
    outer = regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_OUTER], 0.13 * scale), roi)
    inner = regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_OUTER], -0.04 * scale), roi)
    band = (outer > 0.5) & (inner < 0.5) & scene.skin_ok
    if band.sum() < 40:
        return
    sigma = 0.004 * scale
    goal = _blur(scene.target_lab, sigma)[band]

    def cost() -> float:
        lab = _blur(blend.bgr_to_lab(scene.render("lips", section)), sigma)[band]
        return float(np.mean((lab - goal) ** 2 * np.array([0.5, 1.0, 1.0])))

    def outline_cost(value: float) -> float:
        section.outline = value
        return cost()

    def feather_cost(value: float) -> float:
        section.feather = value
        return cost()

    outlines = (0.0, -0.05, 0.05, -0.1, 0.1, -0.15, 0.15, -0.2, 0.2)
    section.outline, _ = _grid_search(outlines, outline_cost)
    section.feather, _ = _grid_search((0.4, 0.0, 0.2, 0.6, 0.8, 1.0), feather_cost)
    section.outline, _ = _grid_search(outlines, outline_cost)


# --------------------------------------------------------------------------
# Putting it together.
# --------------------------------------------------------------------------


def _default_base() -> MakeupSettings:
    try:
        return load_preset(DEFAULT_PRESET_NAME)
    except PresetError:
        return MakeupSettings()


def _round_settings(settings: MakeupSettings) -> MakeupSettings:
    """Values rounded for the sliders, and clamped through the settings' own validation."""
    data = settings.to_dict()
    for section in data.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, float):
                    section[key] = round(value, 3)
    return MakeupSettings.from_dict(data)


def _photo_notes(photo: _Photo, label: str, notes: list[str]) -> None:
    if photo.source_scale < _LOW_RES_SCALE:
        notes.append(f"{label}の顔が小さめのため、アイライナーなどの細かい部分は推定です。")
    if photo.yaw > _WARN_YAW:
        notes.append(f"{label}の顔が斜めを向いているため、精度が下がります。正面の写真のほうが正確です。")
    if photo.eyes_open < 0.12:
        notes.append(f"{label}で目を閉じているため、アイライナーの測定は精度が下がります。")


def _lid_depth_ratio(photo: _Photo) -> float:
    """Upper lash line to brow, x face scale, both eyes on average."""
    from .effects import eyes

    return (eyes._lid_depth(photo.landmarks, "left") + eyes._lid_depth(photo.landmarks, "right")) / (2.0 * photo.scale)


def _eye_state_change(a: _Photo, b: _Photo) -> float:
    """Relative difference of the eye opening and of the lid height between two photos."""
    opening = abs(a.eyes_open - b.eyes_open) / max(a.eyes_open, b.eyes_open, 1e-3)
    depth_a, depth_b = _lid_depth_ratio(a), _lid_depth_ratio(b)
    return max(opening, abs(depth_a - depth_b) / max(depth_a, depth_b, 1e-3) * 2.0)


def _mouth_state_change(a: _Photo, b: _Photo) -> float:
    """Relative difference of the mouth width (a smile widens it) between two photos."""

    def width(photo: _Photo) -> float:
        corners = photo.landmarks[[regions.MOUTH_LEFT_CORNER, regions.MOUTH_RIGHT_CORNER]]
        return float(np.linalg.norm(corners[0] - corners[1])) / photo.scale

    return abs(width(a) - width(b)) / max(width(a), width(b), 1e-3)


def _brow_area_mask(landmarks: np.ndarray, roi: regions.Roi, scale: float) -> np.ndarray:
    area = np.zeros((roi.h, roi.w), np.float32)
    for side in regions.SIDES:
        area = np.maximum(area, regions.brow_mask(landmarks, side, roi, thickness=0.8, feather_px=0.012 * scale))
    return area


def _eye_area_mask(landmarks: np.ndarray, roi: regions.Roi, scale: float) -> np.ndarray:
    """Eyes, lids and brows (soft): where a single-photo estimate replaces the bare photo."""
    area = np.zeros((roi.h, roi.w), np.float32)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        hull = cv2.convexHull(landmarks[s.eye + s.brow].astype(np.float32)).reshape(-1, 2)
        grown = regions.expand_polygon(hull, 0.12 * scale)
        area = np.maximum(area, regions.polygon_mask(grown, roi, feather_px=0.04 * scale))
    return area


def analyze_makeup(
    makeup_bgr: np.ndarray,
    bare_bgr: np.ndarray | None = None,
    *,
    base: MakeupSettings | None = None,
    models_dir: str | Path | None = None,
    models: StillModels | None = None,
) -> AnalysisResult:
    """Measure the makeup in a photo and return settings that reproduce it.

    ``makeup_bgr``: uint8 BGR photo of the made-up face. ``bare_bgr``: optional
    bare-face photo in the same setup; with it the measurements are much more
    exact, and contour and tear bags are measured at all. ``base``: preset
    used for everything that is not measured (default: the built-in natural).
    ``models``: inject already loaded models (else they are loaded from
    ``models_dir``).

    Raises ``AnalysisError`` (Japanese message) for a photo without a usable
    face and ``ModelDownloadError`` when the models cannot be obtained.
    """
    settings = base.copy() if base is not None else _default_base()
    own_models = models is None
    still_models: StillModels = MediaPipeStillModels(models_dir) if models is None else models
    try:
        target = _prepare(makeup_bgr, still_models, "メイクの写真")
        bare = None if bare_bgr is None else _prepare(bare_bgr, still_models, "すっぴんの写真")
    finally:
        if own_models:
            still_models.close()
    notes: list[str] = []
    measurements: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    _photo_notes(target, "メイクの写真", notes)
    if bare is not None:
        _photo_notes(bare, "すっぴんの写真", notes)
    # The measured values are full strength: a lower overall intensity on top would thin them.
    if settings.general.intensity != 1.0:
        settings.general.intensity = 1.0
        notes.append("写真の濃さをそのまま再現するため、全体のメイクの強さは 100% にしました。")
    measurements["photos"] = {
        "source_scale": target.source_scale,
        "work_scale": target.scale,
        "yaw": target.yaw,
        "eyes_open": target.eyes_open,
        "mouth_shape": _mouth_shape(target.landmarks),
        "pose_mismatch": 0.0,
    }
    problems = [
        problem
        for problem in (
            _skin_color_problem(target, "メイクの写真"),
            None if bare is None else _skin_color_problem(bare, "すっぴんの写真"),
        )
        if problem
    ]
    if problems:
        notes.extend(problems)
        notes.append("写真からは何も測らず、すべて元のプリセットの値のままです。")
        return AnalysisResult(_round_settings(settings), _jsonable(measurements), dict.fromkeys(MEASURED_SECTIONS, 0.0), notes)
    # The preset will be applied to a bare live face, so the bare photo's light is
    # the reference: the makeup photo is brought to it, then both are neutralized.
    gray_gains, room_chroma = gray_world_gains(bare if bare is not None else target)
    gray_applied = room_chroma is not None and room_chroma <= _GRAY_WORLD_MAX_CHROMA
    measurements["photos"].update(room_chroma=room_chroma, gray_world_gains=gray_gains)
    if room_chroma is None:
        notes.append("背景がほとんど写っていないため、写真の色かぶりは補正していません。")
    elif not gray_applied:
        notes.append("背景の色が濃いため、写真の色かぶりは補正していません。ファンデの色は参考程度です。")
    if bare is None:
        inputs = _inputs_alone(target, gray_gains, measurements, notes)
    else:
        inputs = _inputs_with_bare(target, bare, gray_gains, measurements, notes)
    scene = _Scene(
        landmarks=target.landmarks,
        scale=target.scale,
        settings=settings,
        measurements=measurements,
        confidence=confidence,
        notes=notes,
        effects=_load_effects(),
        **inputs,
    )
    _fit_all(scene, gray_applied)
    if bare is None:
        sections = (("チーク", settings.blush), ("アイシャドウ", settings.eyeshadow), ("アイライナー", settings.eyeliner))
        found_nothing = [label for label, section in sections if section.opacity <= 0.0]
        if found_nothing:
            notes.append(f"{'・'.join(found_nothing)}は写真からは見つかりませんでした（薄いものは、すっぴん写真がないと見分けられません）。")
    notes.append("歯・顔の形・髪・髪型は写真からは測らず、元のプリセットの値のままです。")
    return AnalysisResult(_round_settings(settings), _jsonable(measurements), confidence, notes)


def _eyes_look_hidden(photo: _Photo) -> bool:
    """Glasses the segmentation did not see. Dark ones: open eyes whose brightest part is
    hardly brighter than deep shadow (the landmarks still find the eyes behind them). Clear
    ones: a dark rim in the band under the eyes. Nothing around the eyes can be measured then."""
    landmarks, scale = photo.landmarks, photo.scale
    roi = regions.Roi(0, 0, photo.image.shape[1], photo.image.shape[0])
    opening = np.zeros(photo.image.shape[:2], bool)
    for side in regions.SIDES:
        opening |= regions.eye_opening_mask(landmarks, side, roi) > 0.5
    around = regions.face_oval_mask(landmarks, roi, expand_px=-0.15 * scale) > 0.5
    around &= (photo.face_skin > 0.5) & ~opening
    if around.sum() < 50:
        return False
    luma = blend.luminance(photo.image)
    skin = float(np.median(luma[around]))
    if photo.eyes_open >= _SQUINT[1] and opening.sum() >= _MIN_EYE_OPENING_PIXELS:
        if float(np.percentile(luma[opening], 90)) < _MIN_SCLERA_CONTRAST * skin:
            return True
    band = np.zeros(photo.image.shape[:2], bool)
    for side in regions.SIDES:
        s = regions.side_indices(side)
        wide = regions.polyline_mask(landmarks[s.eye_lower].astype(np.float32) + np.float32([0.0, 0.35 * scale]), 0.25 * scale, roi)
        band |= (wide > 0.5) & (regions.eye_opening_mask(landmarks, side, roi, expand_px=0.1 * scale) < 0.5)
    band &= regions.face_oval_mask(landmarks, roi, expand_px=-0.05 * scale) > 0.5
    if band.sum() < 30:
        return False
    return float(np.mean(luma[band] < _RIM_DARKNESS * skin)) > _RIM_SHARE


def _skin_color_problem(photo: _Photo, label: str) -> str | None:
    """Why the colors of this photo cannot be measured at all (a Japanese note), or None."""
    roi = regions.Roi(0, 0, photo.image.shape[1], photo.image.shape[0])
    lab = _stable_lab(blend.bgr_to_lab(photo.image), photo.landmarks, roi, photo.face_skin > 0.5)
    if lab is None:
        return None
    if float(np.hypot(lab[1], lab[2])) < _MIN_SKIN_CHROMA:
        if float(lab[0]) > _WHITE_SKIN_LIGHTNESS:
            return f"{label}の顔が白飛びしているため、色を測れません。少し暗めに撮ってください。"
        return f"{label}が白黒に見えるため、色を測れません。カラーの写真を使ってください。"
    hue = math.degrees(math.atan2(float(lab[2]), float(lab[1])))
    if not (_SKIN_HUE_RANGE[0] <= hue <= _SKIN_HUE_RANGE[1]):
        return f"{label}の肌の色が肌らしくないため（色のフィルタなど）、色を測れません。"
    return None


def _clipped_lip_share(landmarks: np.ndarray, clipped: np.ndarray) -> float:
    roi = regions.Roi(0, 0, clipped.shape[1], clipped.shape[0])
    lips = regions.lips_mask(landmarks, roi, outline=-0.12) > 0.9
    return float(clipped[lips].mean()) if lips.any() else 0.0


def _inputs_alone(target: _Photo, gray_gains: np.ndarray, measurements: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """``_Scene`` fields for one photo: the bare image is an approximation."""
    roi = regions.Roi(0, 0, target.image.shape[1], target.image.shape[0])
    image = _apply_gains(target.image, gray_gains)
    unclipped = ~_clipped(target.image)
    skin_ok = (target.face_skin > 0.5) & unclipped
    if float((target.face_skin > 0.5).mean()) < 0.02:
        skin_ok = (regions.face_oval_mask(target.landmarks, roi) > 0.5) & unclipped
    stable = _stable_skin_zone(target.landmarks, roi, target.face_skin > 0.5) > 0.5
    clipped_share = float((~unclipped)[stable].mean()) if stable.any() else 0.0
    measurements["photos"]["clipped_skin_share"] = clipped_share
    if clipped_share > _MAX_CLIPPED_SKIN:
        notes.append("顔が白飛びしているため、肌の色やチークはうまく測れません。少し暗めに撮るとより正確です。")
    notes.append("すっぴん写真がないため、リップ・チーク・アイシャドウの濃さは推定です。同じ場所で撮ったすっぴん写真も指定すると正確になります。")
    notes.append("シェーディング・ハイライトと涙袋はすっぴん写真がないと測れないため、元のプリセットの値のままです。")
    return {
        "target": image,
        "base": _approximate_bare(_Photo(**{**target.__dict__, "image": image}), skin_ok, measurements),
        "hair": _in_front_of_face(target),
        "face_skin": target.face_skin,
        "body_skin": target.body_skin,
        "skin_ok": skin_ok,
        "unclipped": unclipped,
        "has_bare": False,
        "eyes_comparable": False,
        "face_comparable": False,
        "tone_trust": 0.0,
        "eyes_hidden": _eyes_look_hidden(target),
        "lips_blown": _clipped_lip_share(target.landmarks, ~unclipped) > _MAX_CLIPPED_LIPS,
    }


def _inputs_with_bare(
    target: _Photo, bare: _Photo, gray_gains: np.ndarray, measurements: dict[str, Any], notes: list[str]
) -> dict[str, Any]:
    """``_Scene`` fields for two photos: both in the bare photo's light and the makeup photo's geometry."""
    photos = measurements["photos"]
    roi = regions.Roi(0, 0, target.image.shape[1], target.image.shape[0])
    stable_t = _stable_skin_zone(target.landmarks, roi, target.face_skin > 0.5)
    bare_roi = regions.Roi(0, 0, bare.image.shape[1], bare.image.shape[0])
    stable_b = _stable_skin_zone(bare.landmarks, bare_roi, bare.face_skin > 0.5)
    balance = balance_gains(target, bare, stable_t, stable_b)
    tone_trust = balance.trust
    photos.update(
        balance_source=balance.source,
        balance_gains=balance.gains,
        balance_power=balance.power if balance.power is not None else np.ones(3),
        balance_trust=balance.trust,
        skin_disagreement=balance.skin_disagreement,
    )
    if balance.source == "neck":
        notes.append("背景が比べられないため、首の色で2枚の写真の色を合わせました。肌の色は推定です。")
    elif balance.source in ("forehead", "none"):
        notes.append("2枚の写真の明るさや色を背景で合わせられませんでした。ファンデの色は測らず、元のプリセットのままです。")
    if balance.gains.min() < _LARGE_GAIN[0] or balance.gains.max() > _LARGE_GAIN[1]:
        notes.append("2枚の写真の明るさが大きく違うため、色の精度が下がります。同じ明るさで撮るとより正確です。")
        tone_trust = min(tone_trust, 0.3)
    image = _apply_gains(_to_bare_light(target.image, balance), gray_gains)
    aligned = _align_bare(bare, _apply_gains(bare.image, gray_gains), target)
    photos["pose_mismatch"] = aligned.mismatch
    if aligned.mismatch > _LARGE_MISMATCH:
        notes.append("2枚の写真で顔の向きや表情がかなり違うため、精度が下がります。同じ向き・同じ表情で撮るとより正確です。")
    eye_change = _eye_state_change(target, bare)
    mouth_change = _mouth_state_change(target, bare)
    photos.update(eye_state_change=eye_change, mouth_state_change=mouth_change)
    eyes_comparable = eye_change <= _MAX_EYE_STATE_CHANGE
    face_comparable = mouth_change <= _MAX_MOUTH_STATE_CHANGE and aligned.mismatch <= _MAX_FACE_MISMATCH
    clipped = _clipped(target.image) | aligned.clipped
    stable_clipped = float(clipped[stable_t > 0.5].mean()) if (stable_t > 0.5).any() else 0.0
    photos["clipped_skin_share"] = stable_clipped
    if stable_clipped > _MAX_CLIPPED_SKIN:
        notes.append("顔の一部が白飛びしているため、ファンデの色は測らず、元のプリセットのままです。少し暗めに撮るとより正確です。")
        tone_trust = min(tone_trust, 0.15)
    skin_ok = (target.face_skin > 0.5) & (aligned.face_skin > 0.5)
    if float(skin_ok.mean()) < 0.02:
        skin_ok = regions.face_oval_mask(target.landmarks, roi) > 0.5
    skin_ok &= ~clipped
    base_image = aligned.image
    if not (eyes_comparable and face_comparable):
        # Squinting or raised brows change the lids, a smile or a turned head the
        # light on the skin, more than makeup does. Those areas are measured as on
        # a single photo instead; the lips and brows can still be compared.
        single = _approximate_bare(_Photo(**{**target.__dict__, "image": image}), skin_ok, measurements)
        brows = _brow_area_mask(target.landmarks, roi, target.scale)
        if not face_comparable:
            notes.append("2枚の写真で表情や顔の向きが違うため、すっぴん写真と比べたのは唇と眉だけです。ほかはメイクの写真からの推定です。同じ向き・同じ表情で撮るとより正確です。")
            eyes_comparable = False
            lips = regions.lips_mask(target.landmarks, roi, outline=0.1, feather_px=0.012 * target.scale)
            area = np.maximum(lips, brows)[..., None]
            base_image = single * (1.0 - area) + base_image * area
        else:
            notes.append("2枚の写真で目の開き方や眉の位置が違うため、まぶたとアイラインはメイクの写真からの推定です。同じ表情で撮るとより正確です。")
            area = (_eye_area_mask(target.landmarks, roi, target.scale) * (1.0 - brows))[..., None]
            base_image = base_image * (1.0 - area) + single * area
    return {
        "target": image,
        "base": base_image,
        # Bangs or glasses in either photo hide what is under them.
        "hair": np.maximum(_in_front_of_face(target), aligned.hair),
        "face_skin": aligned.face_skin,
        "body_skin": aligned.body_skin,
        "skin_ok": skin_ok,
        "unclipped": ~clipped,
        "has_bare": True,
        "eyes_comparable": eyes_comparable,
        "face_comparable": face_comparable,
        "tone_trust": tone_trust,
        "eyes_hidden": _eyes_look_hidden(target) or _eyes_look_hidden(bare),
        "lips_blown": _clipped_lip_share(target.landmarks, _clipped(target.image))
        - _clipped_lip_share(target.landmarks, aligned.clipped)
        > _MAX_CLIPPED_LIPS,
    }


def _target_without_blush(scene: _Scene) -> tuple[np.ndarray, np.ndarray]:
    """The photo with the measured blush divided out. The blush is committed after the eye
    sections (pipeline order), but it reaches their zones: its tint must not count as theirs."""
    if scene.settings.blush.opacity <= 0.0 or scene.confidence.get("blush", 0.0) <= 0.0:
        return scene.target, scene.target_lab
    ratio = scene.current / np.maximum(scene.render("blush", scene.settings.blush), 1e-3)
    target = np.clip(scene.target * ratio, 0.0, 1.0).astype(np.float32)
    return target, blend.bgr_to_lab(target)


def _fit_all(scene: _Scene, gray_applied: bool) -> None:
    """Every section, in pipeline order (see ``_fit_cheeks_and_contour`` for the exception)."""
    if scene.eyes_hidden:
        scene.notes.append("メガネのようなものが写っているため、眉と目元は測れません。メガネを外して撮るとより正確です。")
    _fit_skin(scene, gray_applied)
    _fit_cheeks_and_contour(scene)
    scene.confidence.setdefault("contour", 0.0)
    photo = scene.target, scene.target_lab
    scene.target, scene.target_lab = _target_without_blush(scene)
    _fit_brows(scene)
    _fit_eyeshadow(scene)
    if scene.eyes_comparable and scene.face_comparable and not scene.eyes_hidden:
        _fit_tearbag(scene)
        scene.commit("tearbag")
    else:
        scene.confidence["tearbag"] = 0.0
    _fit_eyeliner(scene)
    scene.target, scene.target_lab = photo
    scene.commit("blush")
    _fit_lips(scene)


def save_as_preset(result: AnalysisResult, name: str) -> PresetInfo:
    """Store the result as a user preset (``PresetError`` with a Japanese message on failure).

    A built-in preset's name is refused: the new preset would silently replace it, and
    with ``natural`` also what every later analysis starts from.
    """
    safe_name = sanitize_preset_name(name)
    if (BUILTIN_PRESETS_DIR / f"{safe_name}{PRESET_SUFFIX}").is_file():
        raise PresetError(f"『{safe_name}』は標準プリセットの名前なので使えません。別の名前を指定してください。")
    return save_preset(safe_name, result.settings)  # refuses a built-in preset's display name too


# --------------------------------------------------------------------------
# Command line: python -m makeup_cam.analyze MAKEUP.jpg --name いつものメイク
# --------------------------------------------------------------------------


def read_photo(path: Path) -> np.ndarray:
    """Read a photo as uint8 BGR, honoring EXIF rotation (non-ASCII paths are fine)."""
    if path.suffix.lower() in (".heic", ".heif"):
        raise AnalysisError(f"HEIC 形式の写真は読み込めません。JPEG か PNG に変換してください: {path}")
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise AnalysisError(f"写真を読み込めませんでした: {path}\n（詳細: {exc}）") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise AnalysisError(f"写真を読み込めませんでした（画像ファイルではないか、壊れています）: {path}")
    return image


class _HelpInJapanese(argparse.HelpFormatter):
    def add_usage(self, usage: Any, actions: Any, groups: Any, prefix: str | None = None) -> None:
        super().add_usage(usage, actions, groups, "使い方: ")


class _ParserInJapanese(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        self.exit(2, "エラー: 指定の仕方が正しくありません。メイクの写真と --name は必須です（--help で使い方を表示します）。\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = _ParserInJapanese(
        prog="python -m makeup_cam.analyze",
        description="メイクをした写真から、同じ見た目になるプリセットを作ります。",
        formatter_class=_HelpInJapanese,
        add_help=False,
    )
    parser._positionals.title = "引数"
    parser._optionals.title = "オプション"
    parser.add_argument("-h", "--help", action="help", help="この説明を表示して終了します")
    parser.add_argument("makeup", type=Path, help="メイクをした顔の写真")
    parser.add_argument("--bare", type=Path, help="同じ場所・同じ照明で撮ったすっぴんの写真（あると正確になります）")
    parser.add_argument("--base", default=DEFAULT_PRESET_NAME, help="測らない項目に使うプリセット（既定: natural）")
    parser.add_argument("--name", required=True, help="保存するプリセットの名前")
    parser.add_argument("--json", type=Path, help="測定結果を書き出す JSON ファイル（確認用）")
    return parser


def _report(result: AnalysisResult, out: Any) -> None:
    print("項目ごとの信頼度（1 に近いほど確か、0 は測っていません）:", file=out)
    for section in MEASURED_SECTIONS:
        if section in result.confidence:
            print(f"  {SECTION_LABELS.get(section, section)}: {result.confidence[section]:.2f}", file=out)
    if result.notes:
        print("メモ:", file=out)
        for note in result.notes:
            print(f"  ・{note}", file=out)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        makeup = read_photo(args.makeup)
        bare = read_photo(args.bare) if args.bare is not None else None
        base = load_preset(args.base)
        print("写真を解析しています…", flush=True)
        result = analyze_makeup(makeup, bare, base=base)
        # The file first: when it cannot be written, nothing was saved at all.
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        info = save_as_preset(result, args.name)
    except (AnalysisError, PresetError, ModelDownloadError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"エラー: ファイルを保存できませんでした: {exc}", file=sys.stderr)
        return 1
    _report(result, sys.stdout)
    print(f"プリセット『{info.name}』を保存しました: {info.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

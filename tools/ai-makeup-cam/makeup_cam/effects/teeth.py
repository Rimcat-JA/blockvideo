"""Teeth: natural whitening, yellow reduction and evenness ("clean teeth").

Where the teeth are: inside the mouth opening (the inner lip line,
``regions.LIPS_INNER``, feathered by the ``feather`` slider), weighted per
pixel by how tooth-like it looks in Lab:

* bright relative to the teeth's own lightness (the dark mouth cavity and
  deep shadows fade out; teeth in the shade of the lips are kept),
* not in a narrow dark valley: lightness close to the "gap-filled" lightness
  (a morphological closing), so the gaps between teeth are not treated as
  teeth,
* not red, by two tests that must both pass: a* and the hue-aware redness
  a* - 0.6 b* must each stay below a threshold derived from the teeth and the
  skin. Lips, lipstick, gums and the tongue are at least about as red as the
  skin; teeth are less red, and teeth in the shade (warmer and yellower than
  the lit ones, or the lit ones carry a greenish cast) stay in because they
  are yellow rather than pink. Chroma is smoothed first because webcams
  subsample color, which would otherwise make the mask blocky,
* not skin: in a warm light skin passes both redness tests, so a pixel with
  the chroma of the skin around the mouth is out (a hand passing in front).

References. The teeth's lightness and a* are measured on the mouth interior
every frame (the brightest, least red pixels) and folded into a
confidence-weighted running mean: a measurement counts less when the mouth
is only a slit (it then mostly shows the inner lip edge) and when it looks
redder or darker than the current state (a half-closed mouth, a hand
passing), so the mask stays steady while talking; one that looks clearly less
red or brighter replaces the state quickly (the state was measured on lips or
the tongue). The skin around the mouth
(sampled on two rings of points outside the lips) gives the scene's light
level and color cast; it is smoothed with a plain EMA. ``reset`` clears both.

Gating. Nothing happens while the mouth is closed (the frame is skipped
before any image work); the effect fades in with the mouth opening. A
"presence" check fades it out when nothing tooth-like is visible (an open
mouth showing only the tongue or the dark cavity): visible teeth are clearly
less red than the skin and not much darker than it. The presence changes by
at most ``_PRESENCE_STEP`` per frame, so it never pops. When this frame's teeth
measurement has the color of this frame's skin ring, a hand covers the mouth:
the frame is left alone at once and the references do not follow it, so the
teeth come back as they were (``_SAME_AS_SKIN``).

What it does:

* ``whitening`` lifts the teeth's lightness with a gain (shading and texture
  scale with it instead of flattening). The gain fades out smoothly as the
  teeth reference gets brighter (``_WHITENING_ROOM``), which keeps the
  reference below about L 89 and, unlike a hard ceiling, does not turn the
  reference's frame-to-frame jitter into flicker. The result is softly
  compressed toward ``_L_CAP`` above ``_SHOULDER_KNEE``. The curve works on the
  tooth surface (the lightness with its narrow dips filled along the rows) and
  its lift is added, so a gap keeps its depth where the shoulder compresses.
* ``yellow_reduction`` pulls b* (and a little of a*) toward a slightly warm
  neutral that follows the scene's light, never past it, and fades out on
  dim teeth: teeth never turn blue or gray.
* ``evenness`` evens the color (a*, b*) between neighbouring teeth, lifts
  small dark blotches toward an edge-preserving (self-guided) smoothing of
  the lightness (highlights are kept, so the teeth do not turn matte), and
  partially lifts narrow dark gaps and overlap shadows (a morphological
  black-hat along the rows of the lightness). The lift closes at most
  ``_GAP_LIFT`` of a gap's depth (the blotch lift stays out of the gaps), so the
  separations stay visible and the teeth never merge into one white block.

Support. The made-up image is computed at full strength and its change is
blended in with one mask: the feathered mouth opening, multiplied by a window
at the frame's resolution that reaches exactly 0 at ``_FEATHER_REACH`` sigmas
(Euclidean) from the inner lip line. Outside that distance the frame is
bit-identical. The window is needed because a float32 ``cv2.GaussianBlur``
has a square kernel of +-4 sigmas, so the plain feather is still non-zero
more than 5 sigmas away along the diagonals.

Close-up faces (face scale above ``_MAX_WORK_SCALE``) are processed on a
downscaled copy of the mouth ROI (the change and the feather are upsampled),
and the window still applies at full resolution, which keeps the cost bounded
without widening the support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import blend, regions
from ..config import TeethSettings
from .base import Effect, FrameContext, register_effect

_Lab = tuple[float, float, float]

# All lengths below are fractions of the face scale (inter-ocular distance)
# unless noted otherwise.

# ---- geometry ---------------------------------------------------------------
# Mouth opening (largest gap between the inner lip lines, middle pairs):
# below the first value the frame is skipped, the effect is full at the second.
_OPEN_RAMP = (0.012, 0.05)
_OPEN_PAIRS = slice(2, -2)  # pairs of LIPS_UPPER_INNER / LIPS_LOWER_INNER away from the corners
# Mouth-opening feather (sigma): base + feather slider x range.
_FEATHER_BASE = 0.004
_FEATHER_RANGE = 0.02
_FEATHER_REACH = 3.0  # nothing changes farther than this many sigmas from the inner lip line
_MIN_REACH_PX = 1.5  # keeps the polygon's anti-aliased edge under very small feathers
_FILL_SHIFT = 4  # fractional bits of the polygon vertices in cv2.fillPoly
# Statistics use the interior shrunk by this (clear of the lip line).
_STATS_INSET = 0.012
_MIN_FACE_SCALE = 12.0  # px; smaller faces are skipped
# Largest believable mouth opening (bounding box, x face scale; real ones reach
# about 1.15) and face scale (x the frame's longer side): beyond these the
# landmarks are broken.
_MAX_MOUTH_SIZE = 2.5
_MAX_FACE_SCALE_IN_FRAME = 2.0
# Larger faces (face scale in px) are processed at a reduced resolution.
_MAX_WORK_SCALE = 150.0

# ---- teeth-likeness ---------------------------------------------------------
_CHROMA_SIGMA = 0.015  # chroma smoothing before the redness test
# Lightness relative to the teeth reference: weight 0 at the first ratio, 1 at the second.
_L_RATIO = (0.3, 0.55)
# Lightness relative to the gap-filled lightness (closing over _GAP_WIDTH):
# weight 0 at the first ratio, 1 at the second.
_GAP_RATIO = (0.55, 0.85)
# Redness, two tests (both must pass). a*: the weight starts to fade at
# max(teeth a* + _RED_MARGIN, skin a* - _RED_SKIN_MARGIN) and is gone
# _RED_WIDTH above that. Hue: the same on r = a* - _HUE_B_WEIGHT x b* with
# the _HUE_* values (separates pink tongue / gums from yellow teeth).
_RED_MARGIN = 5.0
_RED_SKIN_MARGIN = 6.0
_RED_WIDTH = 6.0
_HUE_B_WEIGHT = 0.6
_HUE_MARGIN = 6.0
_HUE_SKIN_MARGIN = 4.0
_HUE_WIDTH = 5.0
# Teeth reference: the interior pixels scoring above this percentile of
# L - _SCORE_REDNESS x a* (bright and not red: the teeth whenever they cover
# more than about half that share of the mouth opening); their median L and
# a* are the teeth's lightness and a*.
_TEETH_PERCENTILE = 88.0
_SCORE_REDNESS = 1.5
_MIN_INTERIOR_PIXELS = 12
# Presence: teeth a* minus skin a* (full at or below the first, none at the
# second), and teeth L / skin L (none at the first, full at the second).
_PRESENCE_RED = (-4.5, -0.5)
_PRESENCE_LIGHT = (0.6, 0.85)
_PRESENCE_STEP = 0.2  # largest change of the presence per frame

# ---- skin reference (around the mouth) ----------------------------------------
_SKIN_RINGS = (0.08, 0.14)  # outer lip line grown by these -> sample points
_MIN_SKIN_SAMPLES = 8
_REFERENCE_SKIN = (65.0, 11.0, 18.0)  # typical skin L, a*, b* in neutral light
_SKIN_EMA = 0.3
# Something skin-colored in front of the mouth (a hand): a pixel counts as skin
# when its chroma is this close to the skin's (fully at the first, not at the
# second; a*, b* units) and closer to it than to the teeth's: at most these
# fractions of the teeth <-> skin distance. When those two are nearer than
# _SKIN_APART (no test at the first value, full at the second), color cannot
# tell them apart.
_SKIN_CLOSE = (2.5, 5.0)
_SKIN_CLOSE_SHARE = (0.25, 0.5)
_SKIN_APART = (4.0, 8.0)
_SKIN_SAME = 1.5  # the current skin chroma counts as new when this far from the smoothed one
# The same cue for the whole mouth: this frame's teeth measurement (a* and b*)
# has the color of this frame's skin ring when something skin-colored covers
# the opening. No effect and no measurement at the first distance, no
# reaction at the second. Measured at full cover, also on shaded, creased and
# noisy simulated hands: at most 1.5. On real teeth: median 14, below 3 in
# 0.7 % of the open-mouth frames.
_SAME_AS_SKIN = (1.5, 3.0)

# ---- temporal smoothing of the teeth references ------------------------------
# A measurement's confidence (0..1) grows with the mouth opening and with the
# interior's pixel count. The state is a confidence-weighted running mean
# whose memory is capped at _STATS_MEMORY measurements of full confidence: a
# new full-confidence measurement then weighs 1 / (1 + _STATS_MEMORY) (about
# 0.3), and an early, unsure measurement is quickly replaced by better ones.
_CONFIDENCE_OPENING = (0.03, 0.1)
_FULL_CONFIDENCE_AREA = 0.03  # interior pixels (x scale^2) for full confidence
_STATS_MEMORY = 2.3
# Measurements redder / darker than the state are outliers: the confidence is
# divided by 1 + r^2, r being that excess in these units (a*, L); lasting
# changes (the light) still get through, just more gradually. Less red or
# brighter ones mean the state was measured on lips or the tongue: the same
# formula divides the state's memory instead, so the teeth showing up are
# adopted within a frame or two.
_OUTLIER_SCALE = (6.0, 15.0)

# ---- whitening ----------------------------------------------------------------
_MAX_GAIN = 0.16  # lightness gain at whitening 1.0
# The tone curve works on the tooth surface: the lightness with dark features
# narrower than this (about a tooth) filled along the rows.
_SURFACE_WIDTH = 0.12
# The gain is full up to the first teeth L and gone at the second. Wide on
# purpose: the lift's sensitivity to the teeth reference is proportional to
# the slope of this ramp.
_WHITENING_ROOM = (74.0, 90.0)
_SHOULDER_KNEE = 82.0  # above this the lift is compressed ...
_L_CAP = 92.0  # ... toward this L (never reached)

# ---- yellow reduction ---------------------------------------------------------
_NEUTRAL_AB = (1.0, 5.0)  # slightly warm off-white (a*, b*) in neutral light
_NEUTRAL_FLOOR = (0.0, 3.0)  # the target never gets greener / bluer than this
_CAST_ADAPTATION = 0.8  # share of the skin's color cast the neutral follows
_MAX_CAST = 10.0
_MAX_YELLOW_CUT = 0.65  # share of the b* excess over the neutral removed at 1.0
_A_PULL = 0.4  # a* is pulled toward the neutral by this share of the b* cut
# Teeth reference L: no reduction at the first value, full at the second.
# Dim teeth (a dark room) with less yellow just read as gray.
_YELLOW_LIGHT = (35.0, 65.0)

# ---- evenness -----------------------------------------------------------------
_AB_EVEN_SIGMA = 0.05  # color evening reach (about half a front tooth)
_AB_EVEN = 0.7  # share of the color difference removed at 1.0
_MIN_BLUR_SIGMA = 3.0  # px: wider blurs run at a reduced resolution
_L_EVEN_RADIUS = 0.015  # edge-preserving lightness smoothing: box radius ...
_L_EVEN_RANGE = 0.08  # ... and edge threshold as a fraction of the teeth lightness
_L_EVEN = 0.6  # share of the smoothing applied at 1.0
_GAP_WIDTH = 0.05  # dark features narrower than this are gaps
_MAX_GAP_KERNEL = 15  # px
_GAP_LIFT = 0.35  # share of a gap's depth lifted at 1.0 (keeps the separations)
_VALLEY_DEPTH = (0.5, 2.5)  # closing minus lightness (L): blotch lift full below, none above


@dataclass(frozen=True)
class _Levels:
    """Slider values times the master intensity."""

    whitening: float
    yellow: float
    evenness: float

    @classmethod
    def from_settings(cls, settings: TeethSettings, intensity: float) -> _Levels:
        k = max(float(intensity), 0.0)
        return cls(
            whitening=_clamp01(settings.whitening) * k,
            yellow=_clamp01(settings.yellow_reduction) * k,
            evenness=_clamp01(settings.evenness) * k,
        )

    @property
    def active(self) -> bool:
        return self.whitening > 0.0 or self.yellow > 0.0 or self.evenness > 0.0


@dataclass(frozen=True)
class _Stats:
    """The smoothed references (Lab units)."""

    teeth_l: float  # lightness of the teeth (bright reference)
    teeth_a: float  # a* of the teeth
    teeth_b: float  # b* of the teeth
    skin: tuple[float, float, float] | None  # L, a*, b* of the lit skin around the mouth

    @property
    def skin_lab(self) -> tuple[float, float, float]:
        return _REFERENCE_SKIN if self.skin is None else self.skin

    def presence(self) -> float:
        """0..1: how sure we are that teeth are visible (see module docstring)."""
        red = 1.0 - _smoothstep_scalar(self.teeth_a - self.skin_lab[1], *_PRESENCE_RED)
        if self.skin is None:
            return red
        light = _smoothstep_scalar(self.teeth_l / max(self.skin[0], 1.0), *_PRESENCE_LIGHT)
        return red * light

    def red_threshold(self) -> float:
        """a* where the not-red weight starts to fade (see _RED_MARGIN)."""
        return max(self.teeth_a + _RED_MARGIN, self.skin_lab[1] - _RED_SKIN_MARGIN)

    def hue_threshold(self) -> float:
        """a* - _HUE_B_WEIGHT x b* where the not-red weight starts to fade."""
        teeth = self.teeth_a - _HUE_B_WEIGHT * self.teeth_b
        skin = self.skin_lab[1] - _HUE_B_WEIGHT * self.skin_lab[2]
        return max(teeth + _HUE_MARGIN, skin - _HUE_SKIN_MARGIN)

    def neutral(self) -> tuple[float, float]:
        """Target (a*, b*) of the yellow reduction: a warm neutral in the scene's light."""
        skin = self.skin_lab
        cast_a = (skin[1] - _REFERENCE_SKIN[1]) * _CAST_ADAPTATION
        cast_b = (skin[2] - _REFERENCE_SKIN[2]) * _CAST_ADAPTATION
        length = math.hypot(cast_a, cast_b)
        if length > _MAX_CAST:
            cast_a, cast_b = cast_a * _MAX_CAST / length, cast_b * _MAX_CAST / length
        return (
            max(_NEUTRAL_AB[0] + cast_a, _NEUTRAL_FLOOR[0]),
            max(_NEUTRAL_AB[1] + cast_b, _NEUTRAL_FLOOR[1]),
        )


@dataclass(frozen=True)
class _Work:
    """The mouth ROI as processed: the image (possibly downscaled) and the
    landmarks and face scale in its pixel coordinates."""

    image: np.ndarray
    landmarks: np.ndarray
    roi: regions.Roi  # the work image's own frame: Roi(0, 0, w, h) or the frame ROI
    scale: float
    downscaled: bool


@register_effect
class TeethEffect(Effect):
    """Whitening, yellow reduction and evenness of the visible teeth."""

    name = "teeth"
    order = 75

    def __init__(self) -> None:
        self._teeth: tuple[float, float, float] | None = None  # smoothed teeth L, a*, b*
        self._weight = 0.0  # accumulated confidence behind self._teeth
        self._skin: tuple[float, float, float] | None = None  # smoothed skin Lab
        self._presence: float | None = None  # rate-limited presence

    def reset(self) -> None:
        self._teeth = None
        self._weight = 0.0
        self._skin = None
        self._presence = None

    @property
    def stats(self) -> _Stats | None:
        """The current smoothed references (None before teeth were measured)."""
        if self._teeth is None:
            return None
        return _Stats(*self._teeth, self._skin)

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: TeethSettings) -> np.ndarray:
        levels = _Levels.from_settings(settings, ctx.intensity)
        if not levels.active or ctx.landmarks is None:
            return frame
        landmarks = ctx.landmarks
        scale = regions.face_scale(landmarks)
        if not (math.isfinite(scale) and scale >= _MIN_FACE_SCALE):
            return frame
        opening = _opening(landmarks) / scale
        if not opening > _OPEN_RAMP[0]:
            return frame  # closed mouth (or broken landmarks): nothing to do
        if self._teeth is None and opening <= _CONFIDENCE_OPENING[0]:
            return frame  # a slit, and no references yet: nothing to measure or use
        if not _is_believable_mouth(landmarks[regions.LIPS_INNER], scale, frame.shape):
            return frame
        sigma = _feather(settings) * scale
        margin = _support_reach(sigma) + _gap_kernel_size(scale) / 2.0 + 2.0
        roi = regions.roi_around(landmarks[regions.LIPS_INNER], frame.shape, 0.0, margin)
        if roi is None or roi.w < 4 or roi.h < 4:
            return frame
        view = frame[roi.slices]
        work = _work_copy(view, landmarks, roi, scale)
        lab = blend.bgr_to_lab(work.image)
        chroma_sigma = max(_CHROMA_SIGMA * work.scale, 0.5)
        chroma = cv2.GaussianBlur(lab[..., 1:], (0, 0), chroma_sigma, borderType=cv2.BORDER_REPLICATE)
        skin_now = _skin_reference(frame, landmarks, scale)
        self._update_skin(skin_now)
        visible = self._update_teeth(lab[..., 0], chroma, work, opening, skin_now)
        stats = self.stats
        if stats is None:
            return frame
        gate = self._step_presence(stats.presence()) * _smoothstep_scalar(opening, *_OPEN_RAMP) * visible
        if gate <= 0.0:
            return frame
        made_up = _make_up(lab, chroma, stats, levels, work.scale, skin_now)
        delta = blend.lab_to_bgr(made_up)
        delta -= blend.lab_to_bgr(lab)
        area = regions.polygon_mask(work.landmarks[regions.LIPS_INNER], work.roi, sigma * work.scale / scale)
        if work.downscaled:
            delta = cv2.resize(delta, (roi.w, roi.h), interpolation=cv2.INTER_LINEAR)
            area = cv2.resize(area, (roi.w, roi.h), interpolation=cv2.INTER_LINEAR)
        area *= _support_window(landmarks[regions.LIPS_INNER], roi, sigma)
        delta *= area[..., None] * np.float32(gate)
        view += delta
        np.clip(view, 0.0, 1.0, out=view)
        return frame

    # ---- temporal state -------------------------------------------------------

    def _update_skin(self, measured: _Lab | None) -> None:
        if measured is None:
            return
        if self._skin is None:
            self._skin = measured
        else:
            self._skin = tuple(old + _SKIN_EMA * (new - old) for old, new in zip(self._skin, measured))

    def _update_teeth(
        self, lightness: np.ndarray, chroma: np.ndarray, work: _Work, opening: float, skin_now: _Lab | None
    ) -> float:
        """Measure the teeth references and fold them into the smoothed state.

        Returns 0..1, how much this frame's mouth interior can be something
        other than skin (see ``_SAME_AS_SKIN``). What is not is not measured,
        and the caller does not touch it.
        """
        confidence = _smoothstep_scalar(opening, *_CONFIDENCE_OPENING)
        if confidence <= 0.0:
            return 1.0
        interior = _interior(work)
        count = int(np.count_nonzero(interior))
        if count < _MIN_INTERIOR_PIXELS:
            return 1.0
        measured = _teeth_reference(lightness[interior], chroma[interior])
        visible = _unlike_skin(measured, skin_now)
        confidence *= min(count / (_FULL_CONFIDENCE_AREA * work.scale * work.scale), 1.0) * visible
        if self._teeth is None:
            self._teeth, self._weight = measured, confidence
            return visible
        if confidence <= 0.0:
            return visible
        redder = max(measured[1] - self._teeth[1], 0.0) / _OUTLIER_SCALE[0]
        darker = max(self._teeth[0] - measured[0], 0.0) / _OUTLIER_SCALE[1]
        confidence /= 1.0 + redder * redder + darker * darker
        less_red = max(self._teeth[1] - measured[1], 0.0) / _OUTLIER_SCALE[0]
        brighter = max(measured[0] - self._teeth[0], 0.0) / _OUTLIER_SCALE[1]
        memory = self._weight / (1.0 + less_red * less_red + brighter * brighter)
        share = confidence / (memory + confidence)
        self._teeth = tuple(old + share * (new - old) for old, new in zip(self._teeth, measured))
        self._weight = min(memory + confidence, _STATS_MEMORY)
        return visible

    def _step_presence(self, target: float) -> float:
        """Move the presence toward ``target`` by at most ``_PRESENCE_STEP``."""
        if self._presence is None:
            self._presence = target
        else:
            self._presence += min(max(target - self._presence, -_PRESENCE_STEP), _PRESENCE_STEP)
        return self._presence


# ---- measurements -----------------------------------------------------------


def _opening(landmarks: np.ndarray) -> float:
    """Largest distance (px) between paired points of the inner lip lines."""
    upper = landmarks[regions.LIPS_UPPER_INNER[_OPEN_PAIRS]]
    lower = landmarks[regions.LIPS_LOWER_INNER[_OPEN_PAIRS]]
    return float(np.max(np.linalg.norm(upper - lower, axis=1)))


def _is_believable_mouth(polygon: np.ndarray, scale: float, shape: tuple[int, ...]) -> bool:
    """False for lip points no real face gives: not finite (``_opening`` only looks
    at the middle pairs) or absurdly far apart. Rasterizing those stalls ``cv2``
    for up to seconds per frame (the coordinates overflow), so they stop here."""
    if not np.isfinite(polygon).all():
        return False
    size = float((polygon.max(axis=0) - polygon.min(axis=0)).max())
    return size <= _MAX_MOUTH_SIZE * scale and scale <= _MAX_FACE_SCALE_IN_FRAME * max(shape[0], shape[1])


def _work_copy(view: np.ndarray, landmarks: np.ndarray, roi: regions.Roi, scale: float) -> _Work:
    """The ROI to process: ``view`` itself, or for close-up faces a copy
    downscaled to a face scale of about ``_MAX_WORK_SCALE``."""
    if scale <= _MAX_WORK_SCALE:
        return _Work(view, landmarks, roi, scale, False)
    size = (max(1, round(roi.w * _MAX_WORK_SCALE / scale)), max(1, round(roi.h * _MAX_WORK_SCALE / scale)))
    ratio = np.array([size[0] / roi.w, size[1] / roi.h], np.float32)
    small = cv2.resize(view, size, interpolation=cv2.INTER_AREA)
    local = (landmarks - np.asarray(roi.offset, np.float32)) * ratio
    return _Work(small, local, regions.Roi(0, 0, *size), scale * float(ratio[0]), True)


def _interior(work: _Work) -> np.ndarray:
    """Boolean mask of the mouth opening shrunk by ``_STATS_INSET``, clear of the lip line."""
    opening = regions.polygon_mask(work.landmarks[regions.LIPS_INNER], work.roi) > 0.5
    size = 2 * math.ceil(_STATS_INSET * work.scale) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(opening.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def _teeth_reference(lightness: np.ndarray, chroma: np.ndarray) -> _Lab:
    """(L, a*, b*) of the teeth from the mouth interior's pixels (chroma: (N, 2)).

    The teeth are the brightest, least red pixels of the mouth opening. When
    no teeth are visible the same pixels are the lit tongue or lip edge,
    which the presence check then recognizes by their redness.
    """
    score = lightness - np.float32(_SCORE_REDNESS) * chroma[:, 0]
    teeth = score >= np.percentile(score, _TEETH_PERCENTILE)
    a, b = np.median(chroma[teeth], axis=0)
    return max(float(np.median(lightness[teeth])), 1.0), float(a), float(b)


def _unlike_skin(teeth: _Lab, skin: _Lab | None) -> float:
    """0..1: 0 when the color measured as teeth is the color of the skin around the mouth."""
    if skin is None:
        return 1.0
    return _smoothstep_scalar(math.hypot(teeth[1] - skin[1], teeth[2] - skin[2]), *_SAME_AS_SKIN)


_PATCH = np.arange(-1, 2)


def _skin_reference(frame: np.ndarray, landmarks: np.ndarray, scale: float) -> _Lab | None:
    """Median Lab of the lit skin on two rings around the lips (3x3 patches), or None."""
    outer = landmarks[regions.LIPS_OUTER]
    points = np.vstack([regions.expand_polygon(outer, ring * scale) for ring in _SKIN_RINGS])
    points = points[np.isfinite(points).all(axis=1)]
    xs = np.round(points[:, 0]).astype(np.int64)
    ys = np.round(points[:, 1]).astype(np.int64)
    height, width = frame.shape[:2]
    inside = (xs >= 1) & (xs < width - 1) & (ys >= 1) & (ys < height - 1)
    if np.count_nonzero(inside) < _MIN_SKIN_SAMPLES:
        return None
    xs, ys = xs[inside], ys[inside]
    patches = frame[ys[:, None, None] + _PATCH[None, :, None], xs[:, None, None] + _PATCH[None, None, :]]
    colors = patches.reshape(len(xs), -1, 3).mean(axis=1, dtype=np.float32)
    lab = blend.bgr_to_lab(colors.reshape(-1, 1, 3)).reshape(-1, 3)
    lit = lab[lab[:, 0] >= np.median(lab[:, 0])]  # skip shadows (folds, under the lip)
    median = np.median(lit, axis=0)
    return float(median[0]), float(median[1]), float(median[2])


# ---- the makeup ---------------------------------------------------------------


def _make_up(
    lab: np.ndarray, chroma: np.ndarray, stats: _Stats, levels: _Levels, scale: float, skin_now: _Lab | None
) -> np.ndarray:
    """The Lab image with the teeth fully made up; the caller blends the change in."""
    lightness = lab[..., 0]
    kernel = _gap_kernel(scale)
    filled = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    surface = _row_closing(lightness, _odd_size(_SURFACE_WIDTH * scale))
    red, hue = stats.red_threshold(), stats.hue_threshold()
    color = 1.0 - _smoothstep(chroma[..., 0], red, red + _RED_WIDTH)
    color *= 1.0 - _smoothstep(chroma[..., 0] - np.float32(_HUE_B_WEIGHT) * chroma[..., 1], hue, hue + _HUE_WIDTH)
    for skin in _skin_colors(stats.skin, skin_now):
        _exclude_skin(color, chroma, stats, skin)
    likeness = _smoothstep(lightness * np.float32(1.0 / stats.teeth_l), *_L_RATIO)
    likeness *= _smoothstep(lightness / np.maximum(filled, np.float32(1.0)), *_GAP_RATIO)
    likeness *= color
    out = lab.copy()
    if levels.evenness > 0.0:
        between = cv2.morphologyEx(likeness, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
        _even_out(out, lab, likeness, surface, between * color, stats.teeth_l, levels.evenness, scale)
    if levels.whitening > 0.0:
        _whiten(out[..., 0], surface, likeness, stats.teeth_l, levels.whitening)
    if levels.yellow > 0.0:
        amount = levels.yellow * _smoothstep_scalar(stats.teeth_l, *_YELLOW_LIGHT)
        _reduce_yellow(out, likeness, stats.neutral(), amount)
    return out


def _skin_colors(smoothed: _Lab | None, current: _Lab | None) -> tuple[_Lab, ...]:
    """The skin colors to exclude: the smoothed one, and this frame's while it
    differs (a hand arriving changes it at once, the smoothed one lags)."""
    if smoothed is None or current is None:
        return ()
    if math.hypot(smoothed[1] - current[1], smoothed[2] - current[2]) < _SKIN_SAME:
        return (smoothed,)
    return (smoothed, current)


def _exclude_skin(weight: np.ndarray, chroma: np.ndarray, stats: _Stats, skin: _Lab) -> None:
    """In place on ``weight``: fade out what has the color of ``skin`` (see ``_SKIN_CLOSE``)."""
    apart = math.hypot(stats.teeth_a - skin[1], stats.teeth_b - skin[2])
    strength = _smoothstep_scalar(apart, *_SKIN_APART)
    if strength <= 0.0:
        return
    near = min(_SKIN_CLOSE[0], _SKIN_CLOSE_SHARE[0] * apart)
    far = min(_SKIN_CLOSE[1], _SKIN_CLOSE_SHARE[1] * apart)
    distance = (chroma[..., 0] - np.float32(skin[1])) ** 2 + (chroma[..., 1] - np.float32(skin[2])) ** 2
    close = _smoothstep(distance, near * near, far * far)
    weight *= 1.0 - np.float32(strength) * (1.0 - close)


def _even_out(
    out: np.ndarray,
    lab: np.ndarray,
    likeness: np.ndarray,
    surface: np.ndarray,
    gap_area: np.ndarray,
    teeth_l: float,
    amount: float,
    scale: float,
) -> None:
    """In place on ``out``: color evening, lightness smoothing and gap lift.

    ``surface``: the lightness with its dips filled along the rows;
    ``gap_area``: the teeth area with its gaps filled, where gaps may be lifted.
    """
    lightness = lab[..., 0]
    # Color: blend a*/b* toward their average over nearby teeth (normalized
    # convolution: lips, gums and gaps do not take part).
    sigma = _AB_EVEN_SIGMA * scale
    weighted = np.empty(lab.shape, np.float32)
    np.multiply(lab[..., 1:], likeness[..., None], out=weighted[..., 1:])
    weighted[..., 0] = likeness
    weighted = _blur(weighted, sigma)
    average = weighted[..., 1:] / np.maximum(weighted[..., :1], np.float32(1e-3))
    share = likeness * np.float32(amount * _AB_EVEN)
    out[..., 1:] += (average - lab[..., 1:]) * share[..., None]
    # Lightness: lift small dark blotches toward an edge-preserving smoothing
    # (one-sided: the highlights, the enamel's natural gloss, stay). Not in
    # the dips of the surface: on top of the gap lift it would close them
    # further than _GAP_LIFT allows, and blurred webcam gaps are shallow enough
    # for the filter to smooth.
    radius = max(int(round(_L_EVEN_RADIUS * scale)), 1)
    smooth = _self_guided_filter(lightness, radius, (_L_EVEN_RANGE * teeth_l) ** 2)
    blotches = np.maximum(smooth - lightness, 0.0) * (1.0 - _smoothstep(surface - lightness, *_VALLEY_DEPTH))
    out[..., 0] += blotches * (likeness * np.float32(amount * _L_EVEN))
    # Gaps: lift narrow dark valleys between teeth part of the way. Measured
    # along the rows, so a valley's depth is what separates it from the teeth
    # beside it: the brighter tooth below a lip's shadow does not count.
    depth = _row_closing(lightness, _gap_kernel_size(scale)) - lightness
    out[..., 0] += depth * (gap_area * np.float32(amount * _GAP_LIFT))


def _whiten(lightness: np.ndarray, surface: np.ndarray, teeth: np.ndarray, teeth_l: float, amount: float) -> None:
    """In place on Lab L: gain with a soft shoulder (see module docstring), on the
    tooth surface. The lift is added, so gaps and texture keep their depth."""
    gain = 1.0 + amount * _MAX_GAIN * (1.0 - _smoothstep_scalar(teeth_l, *_WHITENING_ROOM))
    if gain <= 1.0:
        return
    lifted = surface * np.float32(gain)
    knee, room = np.float32(_SHOULDER_KNEE), np.float32(_L_CAP - _SHOULDER_KNEE)
    shoulder = knee + room * np.tanh((lifted - knee) / room)
    lifted = np.where(lifted > knee, shoulder, lifted)
    lift = np.maximum(lifted - surface, 0.0)  # never darken (bright speculars stay)
    lightness += lift * teeth


def _reduce_yellow(out: np.ndarray, teeth: np.ndarray, neutral: tuple[float, float], amount: float) -> None:
    """In place on Lab a*/b*: move b* part of the way to the warm neutral, never past it."""
    cut = teeth * np.float32(amount * _MAX_YELLOW_CUT)
    a, b = out[..., 1], out[..., 2]
    b -= np.maximum(b - np.float32(neutral[1]), 0.0) * cut
    a -= (a - np.float32(neutral[0])) * (cut * np.float32(_A_PULL))


# ---- helpers ------------------------------------------------------------------


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide ones at a reduced resolution (same look, less work)."""
    height, width = image.shape[:2]
    factor = int(min(max(sigma // _MIN_BLUR_SIGMA, 1), 8))
    if factor == 1:
        return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma * size[0] / width, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _self_guided_filter(image: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.) of a single-channel image guided by itself.

    Smooths variations well below ``sqrt(eps)`` within a ``2 * radius + 1``
    box and keeps edges well above it. Box filters only: cost independent of
    the radius.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(values: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(values, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mean = box(image)
    variance = np.maximum(box(image * image) - mean * mean, 0.0)
    gain = variance / (variance + np.float32(eps))
    offset = mean - gain * mean
    return box(gain) * image + box(offset)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _feather(settings: TeethSettings) -> float:
    """Mouth-opening feather sigma, x face scale."""
    return _FEATHER_BASE + _FEATHER_RANGE * _clamp01(settings.feather)


def _support_reach(sigma: float) -> float:
    """Largest distance (px) from the inner lip line at which anything changes."""
    return max(_FEATHER_REACH * sigma, _MIN_REACH_PX)


def _support_window(polygon: np.ndarray, roi: regions.Roi, sigma: float) -> np.ndarray:
    """ROI-sized factor for the feather: exactly 0 beyond ``_support_reach(sigma)``
    from ``polygon`` (see the module docstring), give or take a pixel diagonal of
    rasterization, and exactly 1 up to one sigma before that.

    It only touches the feather's last sigma, where the feather is already
    below about 2 %, so the result looks like the plain Gaussian feather.
    """
    outside = np.ones((roi.h, roi.w), np.uint8)
    corners = (np.asarray(polygon, np.float64) - roi.offset - 0.5) * (1 << _FILL_SHIFT)
    cv2.fillPoly(outside, [np.round(corners).astype(np.int32)], 0, cv2.LINE_8, _FILL_SHIFT)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    reach = _support_reach(sigma)
    return 1.0 - _smoothstep(distance, reach - sigma, reach)


def _gap_kernel_size(scale: float) -> int:
    """Odd structuring-element size (px) for the gap search."""
    return min(_odd_size(_GAP_WIDTH * scale), _MAX_GAP_KERNEL)


def _odd_size(length: float) -> int:
    """An odd kernel size (px, at least 3) close to ``length``."""
    return max(int(round(length)) | 1, 3)


def _gap_kernel(scale: float) -> np.ndarray:
    size = _gap_kernel_size(scale)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _row_closing(image: np.ndarray, size: int) -> np.ndarray:
    """Grayscale closing along the rows: dips narrower than ``size`` px are filled
    up to the lower of their two rims."""
    kernel = np.ones((1, size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - np.float32(lo)) * np.float32(1.0 / (hi - lo)), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def _smoothstep_scalar(x: float, lo: float, hi: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

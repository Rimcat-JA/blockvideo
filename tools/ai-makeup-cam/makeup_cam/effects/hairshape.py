"""Hair shape: removes flyaways (アホ毛) and gently changes the hair's silhouette.

Everything is anchored to the head: the work area, the zones and the warp come
from the face landmarks and a silhouette (hair mask + face oval) on a coarse
grid of about ``_GRID_PER_SCALE`` cells per face scale. The hair mask is only
good for "roughly where the hair is": it is low resolution, lags behind a moving
head and often includes the wisps, so what counts as a strand is decided per
frame on the image itself, and the main hair mass is protected by the image too.
Strands are searched on a working copy with a face scale of about
``_WORK_SCALE`` pixels; only the final replacement touches full-resolution pixels.

Flyaways
--------
Strands are looked for in a band around the silhouette, above the chin and away
from the skin and from where the mask is sure of hair (real strands have a low
confidence, while a highlight may look just like the wall). Both methods compare
the picture with a *reference*, what the background should look like, and judge
each pixel against how different pure hair would be from the reference at that
very place (dark hair hardly differs from a dark shelf board). Pixels that differ
at least half as much as that are hair-like; a morphological opening of those
leaves the mass (whatever is thick), and nothing that belongs to the mass is ever
replaced. Where hair covers most of the surroundings outside the mass, it is a
soft, wispy edge rather than a few loose strands, and removal fades out too (it
would leave a cut-out look).

* With a background plate (``ctx.background``) the reference is the plate,
  matched to the live picture: its exposure (a gain measured on a ring of pure
  background farther out) and slow changes of the light across the room (a
  smooth map, measured on all the background around the head). What differs
  from it and is not the mass is a strand, and is replaced by the plate. This
  works on any background, including busy ones. Trust in the plate is measured
  all the time: the typical difference on the ring, the worst part of the ring
  (a poster or a door that changed covers only some of it) and whether the
  plate's edges are still where the picture has them (a camera moved by a pixel
  changes nothing else). If the plate stops matching, the effect falls back to
  the method below instead of showing a wrong background.
* Without a plate the reference is a heavily smoothed picture of the background
  around the head, compared by luminance. It can only say "this looks like the
  plain background" (hair as bright as the wall is simply left alone), so
  strands are found differently: as thin structures (morphological black-hat
  for dark strands, top-hat for bright ones, only the polarity that matches hair
  vs. background at that place). They must start at the silhouette and end near
  it: a door frame or a picture edge runs far out of the band and is left alone.
  The threshold rises with how strong thin structures are on the background
  itself (measured on the same ring), and a busy background fades the effect out
  rather than erasing things. Strand pixels get the color of the smoothed
  background at the brightness of the pixels around them that look like
  background, so this suits plain walls and fine strands.

The strand mask has a short memory (fast rise, slower decay, shifted along with
the head), which keeps a strand from flickering in and out.

Silhouette
----------
``volume_top`` and ``width_sides`` move the hair radially from the middle of the
head: 0 up to the end of the skin (the face never moves), full at the silhouette,
back to 0 a little outside it. The amount is limited by how much hair there is
in that direction, so without hair there is no warp, and the field is smooth
enough never to fold. Only the person moves: where the picture shows the room,
it stays as it is, so thin lines near the hair do not bend. With a plate,
shrinking uncovers the real background: not the plate's colors but the picture
next to it, corrected by how the plate changes between the two places. So a pixel
that does not move can never change, whatever the plate or the person map say.
Without a plate nothing can be uncovered, so shrinking stays very small
(``_SHRINK_WITHOUT_PLATE``), and on a busy background, where thick things cannot
be told from the person, the room bends along. After a warp ``ctx.hair_mask`` is
warped along, for the effects that follow.

The pipeline's fade handles a face that is lost (a bow): this effect needs a
face and multiplies all its strengths by ``ctx.intensity``. Slider values of 0
return the frame untouched, bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import regions
from ..config import HairShapeSettings
from .base import Effect, FrameContext, register_effect

# Lengths are fractions of the face scale (inter-ocular distance) unless noted.

# ---- work area -----------------------------------------------------------------
_ROI_SIDES = 1.5  # beyond the face oval, left and right
_ROI_TOP = 1.8  # above the face oval
_ROI_BOTTOM = 0.2  # below the chin
_AREA_FADE = 0.3  # everything fades out toward the border of the work area
_GRID_PER_SCALE = 16.0  # coarse grid cells per face scale
_MIN_STRIDE = 2  # frame px
_WORK_SCALE = 70.0  # strands are searched on a copy with about this face scale (px)
_FACTOR_HYSTERESIS = 0.15

# ---- silhouette ----------------------------------------------------------------
_HAIR_THRESHOLD = 0.5
_HAIR_NEAR_THRESHOLD = 0.3
_SILHOUETTE_CLOSE = 0.12  # bridges dents of the mask
_SILHOUETTE_OPEN = 0.1  # removes thin spurs of the mask
_FACE_GROW = 0.05  # the face oval is part of the silhouette, grown by this

# ---- zones (signed distance from the silhouette; + is outside) --------------------
_BAND_INNER = (-0.5, -0.35)  # the mask often includes wisps, and the image tells them apart
_REACH = (0.35, 0.9)  # how far out strands are removed, at flyaway 0 and 1
_REACH_FADE = 0.25  # (a long strand fades out toward its tip rather than stopping)
_RING = (0.7, 1.4)  # pure background farther out, for measurements
_HAIR_NEAR = 0.8  # the band only exists this close to actual hair
_CONFIDENT_HAIR = (0.6, 0.8)  # hair confidence where nothing is a flyaway (a highlight as bright as the wall)
_WINDOW_FULL = 0.25  # strands are searched down to this fraction of nose base -> chin ...
_WINDOW_END = 0.95  # ... and not below this one (earrings, collars, shoulders)
_SKIN_GUARD = (0.2, 0.5)  # skin confidence -> no strands here
_FACE_SKIN_GROW = 0.15  # never this close to the face's skin (hairline, baby hair)
_OVAL_GUARD_GROW = 0.1
_OVAL_GUARD_FEATHER = 0.06

# ---- reference and mass ------------------------------------------------------------
_LEVEL_SIGMA = 0.6  # smoothing of the local hair and background levels
_HAIR_CORE = (0.7, -0.1)  # the hair's color: confidence above this, inside by this
_MIN_HAIR_CELLS = 8
_BACKGROUND_FROM = 0.25  # the background's color: this far outside the silhouette
_RING_PERCENTILE = 95.0  # with a plate: everything on the ring should match it
_RING_PERCENTILE_NO_PLATE = 80.0  # a few door frames on the ring say nothing about the noise
_MIN_RING_SAMPLES = 30
_NOISE_FLOOR = 0.012  # difference below this is noise, plus the measured noise ...
_PERSON_RANGE = 0.05  # ... and it is a full person this much above that
_MASS_SEED = 0.5  # a pixel counts toward the mass when it differs this share as much as hair would
_MASS_CLOSE = 0.015  # tiny holes (not the gaps between strands)
_MASS_OPEN = (0.025, 0.05)  # structures thinner than twice this are strands, at flyaway 0 and 1
_MASS_GRADES = (0.5, 1.0)  # shares of that radius, averaged: a narrowing lock fades out, no stump
_MASS_GROW = 0.015  # the mass keeps its own soft edge
_CORE_INSET = (0.3, 0.45)  # the silhouette this far inside is always the mass
_DENSE_START = (0.2, 0.35)  # hair coverage around a strand where removal starts to fade, at flyaway 0 and 1
_DENSE_FULL = (0.35, 0.55)  # ... and where it has stopped
_STATS_EMA = 0.2

# ---- plate ---------------------------------------------------------------------
_PLATE_GAIN_RANGE = (0.5, 2.0)
_PLATE_TRUST_RESIDUAL = (0.03, 0.08)  # typical ring difference after the gain -> trust 1 .. 0
_PLATE_TRUST_OUTLIERS = (0.06, 0.14)  # the same at the ring's high percentile: part of the room changed
_PLATE_TRUST_RISE = 0.08  # share of the gap closed per frame: trust comes back slowly ...
_PLATE_TRUST_FALL = 0.5  # ... and goes quickly
_PLATE_SWITCH = 0.5  # trust at which the method changes (each fades out before that)
_PLATE_FULL = 0.8
_PLATE_FALLBACK_FULL = 0.2  # trust below which the method without a plate is at full strength
_LIGHT_RANGE = (0.7, 1.4)  # how much the light may have changed locally
_LIGHT_KNOWN = (0.02, 0.2)  # share of visible background needed to measure it
_LIGHT_OUTLIER = 0.1  # differs this much from the first estimate: a shoulder, not the light

# ---- plate alignment ----------------------------------------------------------------
_ALIGN_EVERY = 8  # frames between the checks (the camera seldom moves)
_ALIGN_FROM = 0.35  # the check uses what is this far outside the silhouette
_ALIGN_BOX = 5  # frame px: detail is the difference from a box blur of this size
_ALIGN_EDGE = 0.03  # detail of the plate that is surely more than sensor noise
_ALIGN_MIN_SAMPLES = 30  # fewer edges than this: a plain wall, where a shift cannot show
_ALIGN_MAX_SAMPLES = 3000
_ALIGN_REACH = 2  # frame px searched around the expected place
_ALIGN_SURE = 0.97  # correlation that needs no search
_ALIGN_MATCH = (0.4, 0.65)  # correlation at the expected place -> 0 .. 1
_ALIGN_BETTER = (0.04, 0.12)  # how much better a shifted plate fits -> 1 .. 0

# ---- strands without a plate ------------------------------------------------------
_STRAND_RADIUS = (0.014, 0.026)  # half-width of the thickest strand, at flyaway 0 and 1
_MIN_RADIUS = 1  # working-copy px
_POLARITY = (0.02, 0.06)  # background - hair luminance -> that polarity counts
_ABS_FLOOR = 0.008  # weakest response that can be a strand
_FLOOR_PER_LEVEL = 1.0  # threshold above the ring's own thin-structure level
_RELATIVE_FULL = 0.12  # full strand at this share of the hair/background contrast
_MIN_FULL_RANGE = 0.012
_BUSY_RATIO = (0.25, 0.6)  # the ring's level / hair contrast -> the effect fades out
_ATTACH = 0.15  # a strand starts at the silhouette: what stays farther away is a mark on the wall
_LONG_STRUCTURE = 0.3  # what goes this far beyond the band is part of the background
_BACKGROUND_LIKE = 0.15  # differs less than this share of what hair would: background
_FILL_SIGMA = 0.08  # reach of the local background average
_MIN_FILL_LUMA = 0.02  # below this the coarse background is not a usable reference for brightness
_FILL_LOCAL_SHARE = (0.3, 0.7)  # share of visible background around a pixel -> its average can be used
_FILL_KNOWN = (0.05, 0.2)  # nothing that looks like background around a pixel: it is not a flyaway
_LEAVES_LENGTH = 0.08  # a structure going this far out of the search zone runs on: it is the background
_LEAVES_WIDTH = 3.0  # working-copy px: how wide the thinnest bridged structure is
_LEAVES_HOLD = 2.5  # a strand being removed must go this much farther before that changes
_HELD_SHARE = 0.3  # share of a structure under the last strand mask that makes it such a strand

# ---- strand mask ---------------------------------------------------------------
_CANDIDATE = 0.15  # alpha that counts as a strand for the connectivity test
_MIN_AREA = 0.0025  # x scale^2, shorter ones are noise (in the bridged, working-copy mask)
_ALPHA_BOOST = 0.3  # alpha reaching full removal: replacing a bit of background is harmless
_MEMORY_RISE = 0.6  # share of the gap closed per frame when a strand appears
_MEMORY_DECAY = 0.75  # kept per frame when it is no longer seen
_MAX_MEMORY_SHIFT = 0.5  # larger head moves in one frame: forget (x scale)
_MIN_MEMORY_SHIFT = 0.25  # working-copy px
_MIN_MASK = 0.05  # weaker than this only ever lands on background, which hardly changes
_VISIBLE_SHARE = 0.5  # of the noise floor: a smaller difference from the reference needs no replacing

# ---- warp ----------------------------------------------------------------------
_RAY_COUNT = 72
_RAY_SAMPLES = 64
_RAY_MAX = 3.0  # how far silhouette rays go
_RAY_SMOOTH = 2.0  # circular blur of the rays' lengths (sigma in rays)
_RAY_EMA = 0.35
_SKIN_RAY_LEVEL = 0.5
_FACE_MARGIN = 0.08  # the warp starts this far outside the face oval, at the least
_GROW_TOP = 0.13  # silhouette movement at +1
_GROW_SIDES = 0.1
_SHRINK_WITH_PLATE = 0.1  # at -1
_SHRINK_WITHOUT_PLATE = 0.025
_MAX_ZONE_SHARE = 0.33  # movement limit as a share of the hair's thickness (no folds)
_WARP_FALLOFF = 0.8  # back to 0 this far outside the silhouette
_TOP_WINDOW = (20.0, 75.0)  # degrees from up: full, none
_SIDE_WINDOW = ((35.0, 60.0), (105.0, 130.0))  # degrees from up: rise, fall
_MIN_DISPLACEMENT_PX = 0.04
_CLEAR_COVER = 0.02  # a person's coverage this close to 0 or 1 needs no blending
_COVER_OPEN = 0.06  # without a plate, what is thinner than twice this stays with the room
_ROOM_BUSY_PERCENTILE = 70.0  # how different most of the ring is from the smooth background
_ROOM_BUSY_RATIO = (0.1, 0.3)  # that / how different hair would be -> the room bends along
_ROOM_SAMPLE_COLS = 4096  # one remap at most this wide (OpenCV limits a map's width)
_FRAME_EDGE_FADE = 0.15  # the field fades out toward the frame's border

_FOREHEAD, _CHIN = 10, 152
_LUMA_WEIGHTS = np.array([0.114, 0.587, 0.299], np.float32)  # B, G, R
_RECT_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_RECT_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
_NO_ZERO_DISTANCE = 1e6  # farther than anything on a grid (see _distance_to_zero)


@dataclass(frozen=True)
class _Head:
    """Where the head is: the work area, its grids and the head's axes."""

    roi: regions.Roi  # starts at a multiple of ``factor``; its size is a multiple of ``factor`` and ``stride``
    stride: int  # frame px per coarse grid cell
    factor: int  # frame px per working-copy px
    scale: float  # face scale, frame px
    center: np.ndarray  # middle of the eyes, frame px
    up: np.ndarray  # unit vector chin -> forehead
    right: np.ndarray  # unit vector, ``up`` turned 90 degrees clockwise in the image

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.roi.w // self.stride, self.roi.h // self.stride

    @property
    def work_size(self) -> tuple[int, int]:
        return self.roi.w // self.factor, self.roi.h // self.factor

    @property
    def work_slices(self) -> tuple[slice, slice]:
        """The work area on a copy of the whole frame downscaled by ``factor``."""
        f = self.factor
        return slice(self.roi.y0 // f, self.roi.y1 // f), slice(self.roi.x0 // f, self.roi.x1 // f)

    @property
    def work_scale(self) -> float:
        return self.scale / self.factor

    @property
    def cells_per_scale(self) -> float:
        return self.scale / self.stride

    def to_grid_coords(self, points: np.ndarray) -> np.ndarray:
        """Frame coordinates -> coordinates for drawing on or sampling the grid."""
        return (points - np.asarray(self.roi.offset, np.float32)) / np.float32(self.stride) - 0.5

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Frame coordinates of the grid's cell centers, as broadcastable x and y."""
        w, h = self.grid_size
        xs = (np.arange(w, dtype=np.float32) + 0.5) * self.stride + self.roi.x0
        ys = (np.arange(h, dtype=np.float32) + 0.5) * self.stride + self.roi.y0
        return xs[None, :], ys[:, None]

    def shrink(self, image: np.ndarray) -> np.ndarray:
        """ROI-sized image -> working copy (the same array if not downscaled)."""
        if self.factor == 1:
            return image
        return cv2.resize(image, self.work_size, interpolation=cv2.INTER_AREA)

    def to_grid(self, image: np.ndarray) -> np.ndarray:
        """Image of any size in the work area -> grid cells."""
        return cv2.resize(image, self.grid_size, interpolation=cv2.INTER_AREA)

    def sample(self, smooth_mask: np.ndarray) -> np.ndarray:
        """Grid cells of a full-frame mask too smooth to need filtering."""
        half = self.stride // 2
        w, h = self.grid_size
        rows = slice(self.roi.y0 + half, self.roi.y0 + half + h * self.stride, self.stride)
        cols = slice(self.roi.x0 + half, self.roi.x0 + half + w * self.stride, self.stride)
        return np.ascontiguousarray(smooth_mask[rows, cols])

    def grid_to_work(self, grid: np.ndarray) -> np.ndarray:
        return cv2.resize(grid, self.work_size, interpolation=cv2.INTER_LINEAR)

    def grid_to_work_coords(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Working-copy pixels at the centers of the given grid cells."""
        cell = self.stride / self.factor
        return ((rows + 0.5) * cell).astype(np.intp), ((cols + 0.5) * cell).astype(np.intp)


@dataclass(frozen=True)
class _Silhouette:
    """Coarse maps of the head's outline (grid resolution)."""

    distance: np.ndarray  # signed distance from the silhouette, face scales; + outside
    hair: np.ndarray  # hair confidence
    near_hair: np.ndarray  # 1 close to actual hair
    skin: np.ndarray  # 1 on and next to the face's skin, on body skin, and on the face oval
    height: np.ndarray  # 1 above the nose base, fading out toward the chin and the area's border
    ring: np.ndarray  # pure background farther out, for measurements


@dataclass(frozen=True)
class _PlateFit:
    """How the plate relates to the live picture."""

    gain: np.ndarray  # per-channel exposure match
    light: np.ndarray  # smooth change of the light across the room (grid): plate * gain * light ~ live
    trust: float  # 0..1, how well the matched plate fits

    def at(self, rows: np.ndarray, cols: np.ndarray, cell: float) -> np.ndarray:
        """Per-pixel factors for plate colors at the given pixels (``cell`` grid cells per pixel)."""
        light = self.light[(rows * cell).astype(np.intp), (cols * cell).astype(np.intp)]
        return self.gain * light[:, None]

    @property
    def level(self) -> float:
        """Strength of the plate method: fades out toward the switching point."""
        return _ramp(self.trust, _PLATE_SWITCH, _PLATE_FULL)

    @property
    def fallback_level(self) -> float:
        """Strength of the method without a plate while this plate is not trusted."""
        return _ramp(self.trust, _PLATE_SWITCH, _PLATE_FALLBACK_FULL)


@dataclass(frozen=True)
class _RoomLayer:
    """What a warp needs to leave the room alone."""

    cover: np.ndarray  # how much each pixel shows the person (working copy)
    fit: _PlateFit | None  # how to use the plate; None without one
    strength: float  # 0..1: at 0 the room simply bends along with the person


@dataclass(frozen=True)
class _Scene:
    """The picture compared with the reference (working copy, float32 in [0, 1])."""

    difference: np.ndarray  # how much each pixel differs from the reference
    pure_hair: np.ndarray  # how much pure hair would differ from it at that place
    floor: float  # differences below this are noise
    mass: np.ndarray  # the thick part of the person, with a soft edge
    sparse: np.ndarray  # 1 where what is not the mass is sparse: flyaways; 0 where dense: soft hair

    def person(self) -> np.ndarray:
        """Everything clearly different from the reference."""
        return _clip01((self.difference - np.float32(self.floor)) * np.float32(1.0 / _PERSON_RANGE))

    def where_needed(self, mask: np.ndarray) -> np.ndarray:
        """A strand mask without the mass of this very frame (the head may have moved over
        remembered strands) and without pixels that are background already, which keep
        their live noise."""
        needed = 1.0 - self.mass
        needed *= mask
        needed *= _above(self.difference, self.floor * np.float32(_VISIBLE_SHARE))
        return needed

    def background_like(self) -> np.ndarray:
        """What looks like the reference (1) and what does not (0)."""
        limit = self.pure_hair * np.float32(_BACKGROUND_LIKE)
        limit = cv2.max(limit, self.floor, dst=limit)
        return (self.difference < limit).astype(np.float32)


@register_effect
class HairShapeEffect(Effect):
    """Flyaway removal plus small changes of the hair's volume and width."""

    name = "hairshape"
    order = 8
    needs_face = True
    needs_hair_mask = True

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._memory = _StrandMemory()
        self._factor: int | None = None
        self._plate_copy: tuple[np.ndarray, int, np.ndarray] | None = None  # (plate, factor, downscaled plate)
        self._plate_detail: tuple[np.ndarray, np.ndarray] | None = None  # (plate, its fine detail)
        self._alignment = 1.0  # 1: the plate's edges are where the picture has them
        self._align_wait = 0  # frames until the next check
        self._gain: np.ndarray | None = None
        self._light: np.ndarray | None = None  # smooth light change since the plate was taken (grid)
        self._trust: float | None = None
        self._used_plate = False
        self._noise: float | None = None  # how much the ring differs from the reference anyway
        self._room_busy: float | None = None  # 0: plain background, 1: thick things everywhere
        self._ring_level: float | None = None  # thin structures of the background itself
        self._hair_color: np.ndarray | None = None  # mean BGR of the hair
        self._background_color: np.ndarray | None = None  # coarse BGR of the background
        self._rays: np.ndarray | None = None  # silhouette and skin distance per ray, face scales

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: HairShapeSettings) -> np.ndarray:
        intensity = max(float(ctx.intensity), 0.0)
        slider = _clamp(settings.flyaway, 0.0, 1.0)
        flyaway = slider * intensity
        volume = _clamp(settings.volume_top, -1.0, 1.0) * intensity
        width = _clamp(settings.width_sides, -1.0, 1.0) * intensity
        if not (flyaway or volume or width) or ctx.landmarks is None or ctx.hair_mask is None:
            return frame
        if ctx.hair_mask.shape != frame.shape[:2]:
            return frame
        head = _locate_head(ctx.landmarks, frame.shape, self._factor)
        if head is None:
            return frame
        self._factor = head.factor
        if not frame.flags.c_contiguous:  # pixels are addressed by a single index below
            frame = frame.copy()
        reach = _lerp(_REACH, slider)
        warping = bool(volume or width)
        outline = _silhouette(head, ctx)
        colors = head.sample(frame)  # enough for statistics, which are all heavily smoothed
        self._update_hair_color(colors, outline)
        flyaway = flyaway if self._hair_color is not None else 0.0
        plate = _usable_plate(ctx, settings, frame.shape)
        fit = None if plate is None else self._fit_plate(frame, plate, colors, head.sample(plate), outline, head)
        use_plate = fit is not None and fit.level > 0.0 and self._hair_color is not None
        scene = None
        room_busy = 0.0
        if use_plate:
            work = head.shrink(frame[head.roi.slices])
            reference = self._downscaled_plate(plate, head) * fit.gain * head.grid_to_work(fit.light)[..., None]
            hair = tuple(float(v) for v in self._hair_color) + (0.0,)
            pure_hair = _difference(reference, hair)
            scene = self._compare(_difference(work, reference), pure_hair, outline, head, slider, True)
            if flyaway:
                band = head.grid_to_work(_band(outline, reach))
                band *= scene.sparse
                mask = self._memory.update(_strands_from_scene(scene, band, head), head)
                _replace_by_plate(frame, scene.where_needed(mask), _opacity(flyaway) * fit.level, head, plate, fit)
        elif (flyaway or warping) and self._hair_color is not None and self._update_background(colors, outline, head):
            luma = head.shrink(cv2.cvtColor(frame[head.roi.slices], cv2.COLOR_BGR2GRAY))
            background = self._background_color
            background_luma = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            hair_luma = float(np.dot(_LUMA_WEIGHTS, self._hair_color))
            pure_hair = head.grid_to_work(np.abs(background_luma - np.float32(hair_luma)))
            difference = cv2.absdiff(luma, head.grid_to_work(background_luma))
            scene = self._compare(difference, pure_hair, outline, head, slider, False)
            room_busy = self._measure_room_busy(difference, scene, outline, head) if warping else 0.0
            if flyaway:
                band = head.grid_to_work(_band(outline, reach))
                band *= scene.sparse
                contrast = background_luma - np.float32(hair_luma)
                alpha, busy = self._strands_from_image(luma, contrast, band, scene, outline, head, reach, slider)
                mask = scene.where_needed(self._memory.update(alpha, head))
                amount = _opacity(flyaway) * (1.0 - busy) * (1.0 if fit is None else fit.fallback_level)
                _replace_by_surroundings(frame, luma, mask, amount, head, scene, background, background_luma)
        if warping:
            field = self._warp_field(ctx.landmarks, frame.shape, head, outline, volume, width, fit)
            layer = None
            if scene is not None:
                cover = _person_cover(scene, outline, head, include_thin=use_plate)
                layer = _RoomLayer(cover, fit if use_plate else None, fit.level if use_plate else 1.0 - room_busy)
            _warp(frame, ctx, head, field, layer)
        return frame

    # ---- comparing with a reference ---------------------------------------------

    def _downscaled_plate(self, plate: np.ndarray, head: _Head) -> np.ndarray:
        """The plate on the working copy's grid; it never changes, so this is done once."""
        cached = self._plate_copy
        if cached is None or cached[0] is not plate or cached[1] != head.factor:
            f = head.factor
            h, w = plate.shape[:2]
            small = plate
            if f > 1:
                small = cv2.resize(plate[: h - h % f, : w - w % f], (w // f, h // f), interpolation=cv2.INTER_AREA)
            cached = self._plate_copy = (plate, f, small)
        return cached[2][head.work_slices]

    def _fit_plate(
        self,
        frame: np.ndarray,
        plate: np.ndarray,
        colors: np.ndarray,
        plate_colors: np.ndarray,
        outline: _Silhouette,
        head: _Head,
    ) -> _PlateFit | None:
        """Match the plate to the live picture on the background: exposure on the ring, slow
        changes of the light on everything around the head, then see how well it fits."""
        where = outline.ring > 0.5
        if int(np.count_nonzero(where)) >= _MIN_RING_SAMPLES:
            live, known = colors[where], plate_colors[where]
            ratio = np.median(live / np.maximum(known, 1e-3), axis=0)
            gain = np.clip(ratio, *_PLATE_GAIN_RANGE).astype(np.float32)
            light = _light_change(colors, plate_colors * gain, outline, head)
            residuals = np.abs(live - known * gain * light[where][:, None]).max(axis=1)
            rank = int(len(residuals) * _RING_PERCENTILE / 100.0)
            residuals = np.partition(residuals, (len(residuals) // 2, rank))
            trust = (1.0 - _ramp(float(residuals[len(residuals) // 2]), *_PLATE_TRUST_RESIDUAL)) * (
                1.0 - _ramp(float(residuals[rank]), *_PLATE_TRUST_OUTLIERS)
            )
            trust *= self._check_alignment(frame, plate, outline, head)
            if self._gain is None or self._light is None or self._trust is None or self._light.shape != light.shape:
                self._gain, self._light, self._trust = gain, light, trust
            else:
                self._gain = self._gain + _STATS_EMA * (gain - self._gain)
                self._light = self._light + _STATS_EMA * (light - self._light)
                self._trust += (_PLATE_TRUST_FALL if trust < self._trust else _PLATE_TRUST_RISE) * (trust - self._trust)
        if self._gain is None or self._light is None or self._trust is None or self._light.shape != outline.ring.shape:
            return None  # nothing to compare with yet (the head fills the frame)
        return _PlateFit(self._gain, self._light, self._trust)

    def _check_alignment(self, frame: np.ndarray, plate: np.ndarray, outline: _Silhouette, head: _Head) -> float:
        """How well the plate's edges fall on the picture's, now and then: 1 is a fit.

        The correlation of fine detail does not care about exposure or light, and it is
        taken only on the plate's own edges: a poster that was added says nothing here
        (the ring sees it), and plain walls, where a moved camera cannot show, have none.
        """
        self._align_wait -= 1
        if self._align_wait > 0:
            return self._alignment
        self._align_wait = _ALIGN_EVERY
        cached = self._plate_detail
        if cached is None or cached[0] is not plate:
            cached = self._plate_detail = (plate, _fine_detail(plate))
        self._alignment = _edges_alignment(frame[head.roi.slices], cached[1][head.roi.slices], outline, head)
        return self._alignment

    def _update_background(self, colors: np.ndarray, outline: _Silhouette, head: _Head) -> bool:
        """The background around the head, heavily smoothed in space and time (grid).
        False when too little of it is visible to say anything (the head fills the frame)."""
        zone = ((outline.distance > _BACKGROUND_FROM) & (outline.skin < 0.5)).astype(np.float32)
        if cv2.countNonZero(zone) < _MIN_RING_SAMPLES:
            return False
        color = _normalized_blur(colors, zone, _LEVEL_SIGMA * head.cells_per_scale)
        old = self._background_color
        self._background_color = color if old is None or old.shape != color.shape else old + _STATS_EMA * (color - old)
        return True

    def _update_hair_color(self, colors: np.ndarray, outline: _Silhouette) -> None:
        """The hair's mean color, smoothed over time; stays None until there is hair."""
        zone = ((outline.hair > _HAIR_CORE[0]) & (outline.distance < _HAIR_CORE[1])).astype(np.uint8)
        if int(np.count_nonzero(zone)) < _MIN_HAIR_CELLS:
            return
        color = np.array(cv2.mean(colors, mask=zone)[:3], np.float32)
        old = self._hair_color
        self._hair_color = color if old is None else old + _STATS_EMA * (color - old)

    def _compare(
        self,
        difference: np.ndarray,
        pure_hair: np.ndarray,
        outline: _Silhouette,
        head: _Head,
        slider: float,
        use_plate: bool,
    ) -> _Scene:
        """Tell the person from the reference: ``difference`` is how much each pixel differs
        from it, ``pure_hair`` how much a pixel fully covered by hair would."""
        if use_plate != self._used_plate:  # the noise measured on the ring depends on the method
            self._used_plate, self._noise = use_plate, None
        percentile = _RING_PERCENTILE if use_plate else _RING_PERCENTILE_NO_PLATE
        self._noise = _smoothed(self._noise, _ring_percentile(difference, outline, head, percentile))
        floor = _NOISE_FLOOR + (self._noise or 0.0)
        full = cv2.max(pure_hair, floor)
        seed = (difference > cv2.max(full * np.float32(_MASS_SEED), floor)).view(np.uint8)
        mass = _thick_part(seed, head, outline, slider)
        return _Scene(difference, full, floor, mass, _sparse_strands(_clip01(difference / full), mass, head, slider))

    # ---- strands without a plate --------------------------------------------------

    def _strands_from_image(
        self,
        luma: np.ndarray,
        contrast: np.ndarray,
        band: np.ndarray,
        scene: _Scene,
        outline: _Silhouette,
        head: _Head,
        reach: float,
        slider: float,
    ) -> tuple[np.ndarray, float]:
        """Thin structures of the right polarity (working copy), and how busy the background
        is (0..1). ``contrast`` is the background's luminance minus the hair's (grid)."""
        size = 2 * max(_MIN_RADIUS, round(_lerp(_STRAND_RADIUS, slider) * head.work_scale)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        luma8 = cv2.convertScaleAbs(luma, alpha=255.0)
        response = None
        for operation, side in ((cv2.MORPH_BLACKHAT, contrast), (cv2.MORPH_TOPHAT, -contrast)):
            weight = _smoothstep(side, *_POLARITY)
            if weight.any():  # usually only one polarity exists around a head
                part = cv2.morphologyEx(luma8, operation, kernel).astype(np.float32)
                part *= head.grid_to_work(weight)
                if response is None:
                    response = part
                else:
                    response += part
        if response is None:
            return np.zeros_like(band), 0.0  # hair and background look the same: nothing to tell apart
        response *= np.float32(1.0 / 255.0)
        self._ring_level = _smoothed(self._ring_level, _ring_percentile(response, outline, head, _RING_PERCENTILE))
        level = self._ring_level or 0.0
        outside = outline.distance > 0
        typical_contrast = float(np.abs(contrast[outside]).mean()) if outside.any() else 0.0
        full = max(typical_contrast * _RELATIVE_FULL, _MIN_FULL_RANGE)
        response -= np.float32(_ABS_FLOOR + _FLOOR_PER_LEVEL * level)
        response *= np.float32(1.0 / full)
        alpha = _clip01(response)
        alpha *= head.grid_to_work(_search_zone(outline, reach))
        busy = _ramp(level / max(typical_contrast, 1e-3), *_BUSY_RATIO)
        previous = self._memory.projected(head, alpha.shape)
        alpha = cv2.bitwise_and(alpha, alpha, mask=(scene.mass < 0.5).view(np.uint8))  # alpha * (mass < 0.5)
        return _keep_attached(alpha, band, outline.distance, head, reach, previous), busy

    def _measure_room_busy(self, difference: np.ndarray, scene: _Scene, outline: _Silhouette, head: _Head) -> float:
        """How much of the background is unlike its smooth self compared to hair (0..1). A few
        lines hardly count; on a shelf, thick things look like the person and move along."""
        level = _ring_percentile(difference, outline, head, _ROOM_BUSY_PERCENTILE)
        outside = head.grid_to_work(outline.distance) > 0
        typical = float(scene.pure_hair[outside].mean()) if outside.any() else 0.0
        sample = None if level is None else _ramp(level / max(typical, 1e-3), *_ROOM_BUSY_RATIO)
        self._room_busy = _smoothed(self._room_busy, sample)
        return self._room_busy or 0.0

    # ---- warp -------------------------------------------------------------------

    def _warp_field(
        self,
        landmarks: np.ndarray,
        shape: tuple[int, ...],
        head: _Head,
        outline: _Silhouette,
        volume: float,
        width: float,
        fit: _PlateFit | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Backward displacement ``(dx, dy)`` on the grid: ``out(p) = in(p + d(p))``."""
        rays, skin_rays = self._silhouette_rays(head, outline)
        begin = np.maximum(_face_rays(landmarks, head) + _FACE_MARGIN * head.scale, skin_rays)
        limit = _MAX_ZONE_SHARE * np.maximum(rays - begin, 0.0)
        amount = np.clip(_ray_amounts(volume, width, 0.0 if fit is None else fit.level) * head.scale, -limit, limit)
        if not np.any(np.abs(amount) > _MIN_DISPLACEMENT_PX):
            return None
        xs, ys = head.cells()
        rel_x, rel_y = xs - head.center[0], ys - head.center[1]
        along_right = rel_x * head.right[0] + rel_y * head.right[1]
        along_up = rel_x * head.up[0] + rel_y * head.up[1]
        radius = np.maximum(np.hypot(along_right, along_up), 1.0)
        position = (np.arctan2(along_right, along_up) % (2.0 * math.pi)) * np.float32(_RAY_COUNT / (2.0 * math.pi))
        lower = position.astype(np.int32) % _RAY_COUNT
        upper = (lower + 1) % _RAY_COUNT
        part = position - np.floor(position)

        def interpolate(values: np.ndarray) -> np.ndarray:
            return values[lower] * (1.0 - part) + values[upper] * part

        start, edge = interpolate(begin), interpolate(rays)
        rise = _smoothstep_t((radius - start) / np.maximum(edge - start, 1.0))
        fall = 1.0 - _smoothstep_t((radius - edge) / np.float32(_WARP_FALLOFF * head.scale))
        moved = interpolate(amount) * rise * fall * _frame_edge_fade(xs, ys, shape, head) * _area_fade(head)
        # Content moves outward by ``moved`` along the ray, so sample that far back.
        unit_x = (along_right * head.right[0] + along_up * head.up[0]) / radius
        unit_y = (along_right * head.right[1] + along_up * head.up[1]) / radius
        return -moved * unit_x, -moved * unit_y

    def _silhouette_rays(self, head: _Head, outline: _Silhouette) -> tuple[np.ndarray, np.ndarray]:
        """Distances (px) from the middle of the head along each ray to the silhouette
        and to the end of the skin before it (the hair's thickness lies in between),
        smoothed over the rays and over time."""
        angles = np.linspace(0.0, 2.0 * math.pi, _RAY_COUNT, endpoint=False, dtype=np.float32)
        steps = np.linspace(0.0, _RAY_MAX * head.scale, _RAY_SAMPLES, dtype=np.float32)
        directions = np.outer(np.sin(angles), head.right) + np.outer(np.cos(angles), head.up)
        points = head.center + steps[None, :, None] * directions[:, None, :].astype(np.float32)
        grid = head.to_grid_coords(points)
        map_x, map_y = grid[..., 0], grid[..., 1]
        polar = cv2.remap(outline.distance, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        outside = polar > 0.0
        first = np.maximum(np.where(outside.any(axis=1), outside.argmax(axis=1), _RAY_SAMPLES - 1), 1)
        index = np.arange(_RAY_COUNT)
        before, after = polar[index, first - 1], polar[index, first]
        crossing = np.clip(-before / np.maximum(after - before, 1e-4), 0.0, 1.0)  # where the sign changes
        step = float(steps[1] - steps[0])
        rays = (first - 1 + crossing) * step
        skin = cv2.remap(outline.skin, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inside = (skin > _SKIN_RAY_LEVEL) & (np.arange(_RAY_SAMPLES)[None, :] <= first[:, None])
        last_skin = _RAY_SAMPLES - 1 - inside[:, ::-1].argmax(axis=1)
        skin_rays = np.where(inside.any(axis=1), last_skin, 0) * step
        # Relative to the face scale: the smoothing survives a change of distance.
        smoothed = np.stack((_circular_blur(rays, _RAY_SMOOTH), _circular_blur(skin_rays, _RAY_SMOOTH)))
        relative = smoothed / np.float32(head.scale)
        self._rays = relative if self._rays is None else self._rays + _RAY_EMA * (relative - self._rays)
        return self._rays[0] * np.float32(head.scale), self._rays[1] * np.float32(head.scale)


class _StrandMemory:
    """Strand mask with temporal hysteresis (working copy, shifted along with the head)."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None
        self._head: _Head | None = None
        self._projected: tuple[_Head, tuple[int, ...], np.ndarray] | None = None  # for this frame's head

    def update(self, alpha: np.ndarray, head: _Head) -> np.ndarray:
        previous = self.projected(head, alpha.shape)
        moved_toward = alpha - previous
        moved_toward *= np.float32(_MEMORY_RISE)
        moved_toward += previous
        mask = cv2.max(moved_toward, previous * np.float32(_MEMORY_DECAY), dst=moved_toward)
        self._mask, self._head, self._projected = mask, head, None
        return mask

    def projected(self, head: _Head, shape: tuple[int, ...]) -> np.ndarray:
        """The last mask on the current working copy, moved along with the head (treat as read-only)."""
        cached = self._projected
        if cached is not None and cached[0] is head and cached[1] == shape:
            return cached[2]
        result = self._project(head, shape)
        self._projected = (head, shape, result)
        return result

    def _project(self, head: _Head, shape: tuple[int, ...]) -> np.ndarray:
        old, last = self._mask, self._head
        if old is None or last is None or last.factor != head.factor:
            return np.zeros(shape, np.float32)
        move = head.center - last.center
        if float(np.linalg.norm(move)) > _MAX_MEMORY_SHIFT * head.scale:
            return np.zeros(shape, np.float32)
        shift_x = (float(move[0]) + last.roi.x0 - head.roi.x0) / head.factor
        shift_y = (float(move[1]) + last.roi.y0 - head.roi.y0) / head.factor
        if abs(shift_x) < _MIN_MEMORY_SHIFT and abs(shift_y) < _MIN_MEMORY_SHIFT and old.shape == shape:
            return old  # a mask is anything but exact to a fraction of a pixel
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(old, matrix, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Head and zones
# --------------------------------------------------------------------------


def _locate_head(landmarks: np.ndarray, shape: tuple[int, ...], previous_factor: int | None) -> _Head | None:
    scale = max(regions.face_scale(landmarks), 1.0)
    factor = _work_factor(scale, previous_factor)
    stride = max(_MIN_STRIDE, round(scale / _GRID_PER_SCALE))
    align = math.lcm(stride, factor)
    oval = landmarks[regions.FACE_OVAL]
    lo, hi = oval.min(axis=0), oval.max(axis=0)
    x0 = max(0, math.floor(lo[0] - _ROI_SIDES * scale))
    y0 = max(0, math.floor(lo[1] - _ROI_TOP * scale))
    x1 = min(shape[1], math.ceil(hi[0] + _ROI_SIDES * scale))
    y1 = min(shape[0], math.ceil(hi[1] + _ROI_BOTTOM * scale))
    x0 -= x0 % factor
    y0 -= y0 % factor
    x1 -= (x1 - x0) % align
    y1 -= (y1 - y0) % align
    if x1 - x0 < 8 * align or y1 - y0 < 8 * align:
        return None
    up = landmarks[_FOREHEAD] - landmarks[_CHIN]
    up = (up / max(float(np.linalg.norm(up)), 1e-6)).astype(np.float32)
    right = np.array([-up[1], up[0]], np.float32)
    center = (regions.eye_center(landmarks, "left") + regions.eye_center(landmarks, "right")) / 2
    return _Head(regions.Roi(x0, y0, x1, y1), stride, factor, scale, center, up, right)


def _work_factor(scale: float, previous: int | None) -> int:
    """Integer downscale factor; kept while the ideal one stays near it, so a head
    hovering at a threshold does not switch resolutions back and forth."""
    ideal = scale / _WORK_SCALE
    if previous is not None and abs(ideal - previous) < 0.5 + _FACTOR_HYSTERESIS:
        return previous
    return max(1, round(ideal))


def _usable_plate(ctx: FrameContext, settings: HairShapeSettings, shape: tuple[int, ...]) -> np.ndarray | None:
    plate = ctx.background
    if not settings.use_background_plate or plate is None or plate.shape != shape:
        return None
    return plate if plate.flags.c_contiguous else np.ascontiguousarray(plate)


def _silhouette(head: _Head, ctx: FrameContext) -> _Silhouette:
    cells = head.cells_per_scale
    landmarks = ctx.landmarks
    hair = head.sample(ctx.hair_mask)
    oval = _fill_polygon(head, regions.expand_polygon(landmarks[regions.FACE_OVAL], _FACE_GROW * head.scale))
    solid = (hair > _HAIR_THRESHOLD).astype(np.uint8) | oval
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, _ellipse(_SILHOUETTE_CLOSE * cells))
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, _ellipse(_SILHOUETTE_OPEN * cells))
    solid = _without_holes(solid | oval)  # the skin of a forehead between hairline and oval is no background
    distance = _distance_to_zero(1 - solid) - _distance_to_zero(solid)
    distance *= np.float32(1.0 / cells)
    hair_seed = (hair > _HAIR_NEAR_THRESHOLD).astype(np.uint8)
    to_hair = _distance_to_zero(1 - hair_seed) * np.float32(1.0 / cells)
    near_hair = 1.0 - _smoothstep(to_hair, _HAIR_NEAR * 0.7, _HAIR_NEAR)
    skin = _skin_guard(head, ctx)
    height = _height_window(head, landmarks) * _area_fade(head)
    ring = _smoothstep(distance, _RING[0], _RING[0] + 0.05) * (1.0 - _smoothstep(distance, _RING[1], _RING[1] + 0.05))
    ring = ring * (height > 0.5) * (1.0 - skin)
    return _Silhouette(distance, hair, near_hair, skin, height, ring)


def _without_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def _skin_guard(head: _Head, ctx: FrameContext) -> np.ndarray:
    """1 where strands are never touched: the face oval, near the face's skin, on body skin."""
    oval = regions.expand_polygon(ctx.landmarks[regions.FACE_OVAL], _OVAL_GUARD_GROW * head.scale)
    guard = regions.feather(_fill_polygon(head, oval).astype(np.float32), _OVAL_GUARD_FEATHER * head.cells_per_scale)
    if ctx.face_skin_mask is not None and ctx.face_skin_mask.shape == ctx.hair_mask.shape:
        confidence = cv2.dilate(head.sample(ctx.face_skin_mask), _ellipse(_FACE_SKIN_GROW * head.cells_per_scale))
        guard = cv2.max(guard, _smoothstep(confidence, *_SKIN_GUARD))
    if ctx.body_skin_mask is not None and ctx.body_skin_mask.shape == ctx.hair_mask.shape:
        guard = cv2.max(guard, _smoothstep(head.sample(ctx.body_skin_mask), *_SKIN_GUARD))
    return guard


def _height_window(head: _Head, landmarks: np.ndarray) -> np.ndarray:
    """1 above the nose base, fading out toward the chin, along the head's up axis."""
    xs, ys = head.cells()
    base = landmarks[regions.NOSE_BOTTOM]
    down = float(np.dot(landmarks[_CHIN] - base, -head.up))
    along = ((xs - base[0]) * -head.up[0] + (ys - base[1]) * -head.up[1]) / max(down, 1.0)
    return 1.0 - _smoothstep(along, _WINDOW_FULL, _WINDOW_END)


def _area_fade(head: _Head) -> np.ndarray:
    """1 inside the work area, fading out toward its border (grid resolution)."""
    w, h = head.grid_size
    reach = max(_AREA_FADE * head.cells_per_scale, 1.0)
    fade_x = np.clip(np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / reach, 0.0, 1.0).astype(np.float32)
    fade_y = np.clip(np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / reach, 0.0, 1.0).astype(np.float32)
    return fade_y[:, None] * fade_x[None, :]


def _band(outline: _Silhouette, reach: float) -> np.ndarray:
    """Where strands may be removed (grid resolution)."""
    d = outline.distance
    outer = 1.0 - _smoothstep(d, reach, reach + _REACH_FADE)
    inner = _smoothstep(d, *_BAND_INNER)
    not_confident_hair = 1.0 - _smoothstep(outline.hair, *_CONFIDENT_HAIR)
    return inner * outer * outline.height * outline.near_hair * (1.0 - outline.skin) * not_confident_hair


def _search_zone(outline: _Silhouette, reach: float) -> np.ndarray:
    """Where structures are followed to see where they end: farther out than the band."""
    d = outline.distance
    limit = reach + _LONG_STRUCTURE
    outer = 1.0 - _smoothstep(d, limit + 0.1, limit + 0.2)
    return _smoothstep(d, *_BAND_INNER) * outer * (1.0 - outline.skin)


def _fill_polygon(head: _Head, points: np.ndarray) -> np.ndarray:
    w, h = head.grid_size
    canvas = np.zeros((h, w), np.uint8)
    fixed = np.round(head.to_grid_coords(points) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [fixed], 1, cv2.LINE_8, 4)
    return canvas


def _ring_percentile(values: np.ndarray, outline: _Silhouette, head: _Head, percentile: float) -> float | None:
    """A percentile of ``values`` (working copy) on the ring; None if the ring is too small."""
    rows, cols = np.nonzero(outline.ring > 0.5)
    if len(rows) < _MIN_RING_SAMPLES:
        return None
    samples = values[head.grid_to_work_coords(rows, cols)]
    rank = int(len(samples) * percentile / 100.0)
    return float(np.partition(samples, rank)[rank])


# --------------------------------------------------------------------------
# Strands
# --------------------------------------------------------------------------


def _light_change(colors: np.ndarray, plate_colors: np.ndarray, outline: _Silhouette, head: _Head) -> np.ndarray:
    """Smooth map of how much brighter the room is now than on the plate (grid): 1 means
    unchanged. It is measured on the background around the head and spreads over the head,
    which also gives the fill the soft shadow the person casts on the wall."""
    around = (outline.distance > _BACKGROUND_FROM) & (outline.skin < 0.5)
    sigma = _LEVEL_SIGMA * head.cells_per_scale
    live_luma = cv2.cvtColor(colors, cv2.COLOR_BGR2GRAY)
    plate_luma = cv2.cvtColor(plate_colors, cv2.COLOR_BGR2GRAY)
    here = live_luma / np.maximum(plate_luma, 1e-3)
    # A person's shoulder or a ponytail outside the silhouette is not the light: what no
    # change of the light explains, and then what differs from the first estimate, is left out.
    plausible = (here >= _LIGHT_RANGE[0]) & (here <= _LIGHT_RANGE[1])
    estimate = _light_ratio(live_luma, plate_luma, (around & plausible).astype(np.float32), sigma)[0]
    consistent = np.abs(here - estimate) <= np.float32(_LIGHT_OUTLIER)
    ratio, known = _light_ratio(live_luma, plate_luma, (around & plausible & consistent).astype(np.float32), sigma)
    return 1.0 + (ratio - 1.0) * _smoothstep(known, *_LIGHT_KNOWN)


def _light_ratio(
    live_luma: np.ndarray, plate_luma: np.ndarray, zone: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth ratio of the two pictures over ``zone``, and how much of it is known around each cell."""
    live = _soft_blur(live_luma * zone, sigma)
    plate = _soft_blur(plate_luma * zone, sigma)
    return np.clip(live / np.maximum(plate, 1e-3), *_LIGHT_RANGE), _soft_blur(zone, sigma)


def _fine_detail(image: np.ndarray) -> np.ndarray:
    """A BGR image's luminance minus its surroundings: edges and thin lines."""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return luma - cv2.blur(luma, (_ALIGN_BOX, _ALIGN_BOX), borderType=cv2.BORDER_REPLICATE)


def _edges_alignment(picture: np.ndarray, plate_detail: np.ndarray, outline: _Silhouette, head: _Head) -> float:
    """1 when the plate's edges (in ``plate_detail``, the work area's fine detail) are in the
    work area's ``picture`` at the same place; 0 when they are gone or a small shift fits better."""
    zone = ((outline.distance > _ALIGN_FROM) & (outline.skin < 0.5)).astype(np.uint8)
    zone = cv2.resize(zone, (plate_detail.shape[1], plate_detail.shape[0]), interpolation=cv2.INTER_NEAREST)
    edges = ((np.abs(plate_detail) > _ALIGN_EDGE) & (zone > 0)).astype(np.uint8)
    reach = _ALIGN_REACH
    picked = np.flatnonzero(edges[reach:-reach, reach:-reach].ravel())
    if len(picked) < _ALIGN_MIN_SAMPLES:
        return 1.0
    picked = picked[:: max(1, len(picked) // _ALIGN_MAX_SAMPLES)]
    rows, cols = np.divmod(picked, plate_detail.shape[1] - 2 * reach)
    rows, cols = rows + reach, cols + reach
    wanted = plate_detail[rows, cols]
    detail = _fine_detail(picture)

    def correlation(dx: int, dy: int) -> float:
        seen = detail[rows + dy, cols + dx]
        return float(np.dot(seen, wanted)) / math.sqrt(max(float(np.dot(seen, seen) * np.dot(wanted, wanted)), 1e-12))

    here = correlation(0, 0)
    if here >= _ALIGN_SURE:
        return 1.0
    best = max(correlation(dx, dy) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1))
    return _ramp(here, *_ALIGN_MATCH) * (1.0 - _ramp(best - here, *_ALIGN_BETTER))


def _difference(image: np.ndarray, reference: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """How different a BGR image is from another (or a color), per pixel: the channels'
    differences, weighted like luminance."""
    return cv2.cvtColor(cv2.absdiff(image, reference), cv2.COLOR_BGR2GRAY)


def _opacity(flyaway: float) -> float:
    """How completely strands are replaced: quickly most of the way, fully only at 1."""
    return 1.0 - (1.0 - min(flyaway, 1.0)) ** 3


def _thick_part(seed: np.ndarray, head: _Head, outline: _Silhouette, slider: float) -> np.ndarray:
    """The mass (soft mask): what stays of ``seed`` (hair-like pixels) after removing
    everything thin. Well inside the silhouette it always counts as the mass.

    Several openings are averaged, so a lock that narrows fades out gradually
    instead of ending in a blunt stump that jitters from frame to frame.
    """
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _ellipse(_MASS_CLOSE * head.work_scale))
    radius = _lerp(_MASS_OPEN, slider) * head.work_scale
    grades = np.zeros(closed.shape, np.float32)
    for share in _MASS_GRADES:
        grades += cv2.morphologyEx(closed, cv2.MORPH_OPEN, _ellipse(radius * share))
    grades *= np.float32(1.0 / len(_MASS_GRADES))
    soft = cv2.blur(cv2.dilate(grades, _ellipse(_MASS_GROW * head.work_scale)), (5, 5))
    core = 1.0 - _smoothstep(outline.distance, -_CORE_INSET[1], -_CORE_INSET[0])
    return cv2.max(soft, head.grid_to_work(core))


def _sparse_strands(coverage: np.ndarray, mass: np.ndarray, head: _Head, slider: float) -> np.ndarray:
    """How sparse the hair outside the mass is (0..1, working copy).

    ``coverage`` is how much each pixel is covered by hair. A few loose strands
    cover little of their surroundings; a soft, wispy edge of the hair covers
    most of it, and removing that would leave a cut-out look.
    """
    free = 1.0 - mass
    covered = _small_blur(head.to_grid(coverage * free), head)
    total = _small_blur(head.to_grid(free), head)
    share = covered / np.maximum(total, 1e-3)
    return head.grid_to_work(1.0 - _smoothstep(share, _lerp(_DENSE_START, slider), _lerp(_DENSE_FULL, slider)))


def _keep_attached(
    alpha: np.ndarray, band: np.ndarray, distance: np.ndarray, head: _Head, reach: float, previous: np.ndarray
) -> np.ndarray:
    """``alpha`` within the band, without what does not start at the silhouette or what
    runs far out of the band or out of the work area (both are part of the background).
    ``alpha`` must not include the mass: its texture would connect all the strands of a head
    with each other and with the border of the work area, and they would stand or fall together.

    A door frame crosses the outer end of the search zone for its whole width; the tip
    of a strand or a speck next to it only touches it. The verdict sticks: a strand being
    removed (``previous``, the last strand mask) must go much farther to count as
    background, or it would come back for a frame whenever its tip reaches that far.
    """
    candidates = (alpha > _CANDIDATE).astype(np.uint8)
    if not candidates.any():
        return np.zeros_like(alpha)
    bridged = cv2.dilate(candidates, _RECT_3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    d = head.grid_to_work(distance)
    # Label 0 is the rest of the picture, which is never kept: only the candidates' pixels count.
    where = np.flatnonzero(labels)
    found = labels.ravel()[where]
    near = d.ravel()[where]
    attached = np.zeros(count, bool)
    attached[found[near <= _ATTACH]] = True
    limit = _LEAVES_LENGTH * head.work_scale * _LEAVES_WIDTH
    beyond = np.bincount(found[near > reach + _LONG_STRUCTURE], minlength=count).astype(np.float32)
    for border in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        beyond[border] = np.float32(limit * _LEAVES_HOLD)  # nothing in the work area's border is a loose strand
    area = stats[:, cv2.CC_STAT_AREA]
    held = np.bincount(found[previous.ravel()[where] > 0.5], minlength=count) > _HELD_SHARE * area
    leaves = beyond >= np.where(held, limit * _LEAVES_HOLD, limit)
    keep = attached & ~leaves & (area >= _MIN_AREA * head.work_scale * head.work_scale)
    keep[0] = False
    kept = np.zeros_like(alpha)
    chosen = where[keep[found]]
    kept.ravel()[chosen] = alpha.ravel()[chosen]
    finished = _finish_alpha(kept)
    finished *= band
    return finished


def _strands_from_scene(scene: _Scene, band: np.ndarray, head: _Head) -> np.ndarray:
    """Strand alpha with a plate: what differs from it, is not the mass and is not a speck of noise."""
    alpha = scene.person() * (1.0 - scene.mass)
    candidates = (alpha > _CANDIDATE).astype(np.uint8)
    if not candidates.any():
        return np.zeros_like(alpha)
    bridged = cv2.dilate(candidates, _RECT_3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= _MIN_AREA * head.work_scale * head.work_scale
    keep[0] = False
    finished = _finish_alpha(alpha * keep[labels].astype(np.float32))
    finished *= band
    return finished


def _finish_alpha(alpha: np.ndarray) -> np.ndarray:
    """Cover the strands' soft sides as well: replacing a bit of background is harmless."""
    boosted = alpha * np.float32(1.0 / _ALPHA_BOOST)
    cv2.min(boosted, 1.0, dst=boosted)
    return cv2.blur(cv2.dilate(boosted, _RECT_3), (3, 3), dst=boosted)


def _replace_by_plate(
    frame: np.ndarray, mask: np.ndarray, amount: float, head: _Head, plate: np.ndarray, fit: _PlateFit
) -> None:
    """Replace the masked pixels of ``frame`` by the matched plate."""
    placed = _masked_pixels(mask, amount, head, frame.shape[1])
    if placed is None:
        return
    index, rows, cols, weight = placed
    pixels = frame.reshape(-1, 3)
    live = np.take(pixels, index, axis=0)
    target = np.take(plate.reshape(-1, 3), index, axis=0) * fit.at(rows, cols, head.factor / head.stride)
    pixels[index] = live + (target - live) * weight[:, None]


def _replace_by_surroundings(
    frame: np.ndarray,
    luma: np.ndarray,
    mask: np.ndarray,
    amount: float,
    head: _Head,
    scene: _Scene,
    coarse: np.ndarray,
    coarse_luma: np.ndarray,
) -> None:
    """Replace the masked pixels of ``frame`` by the background estimated around them.

    The color is the coarse map of the background; its brightness follows the
    pixels around that look like background, where there are enough of them (a
    few, between dense strands, would tint the estimate).
    """
    like = scene.background_like()
    known = _small_blur(head.to_grid(like), head)
    usable = _smoothstep(head.grid_to_work(known), *_FILL_KNOWN)
    usable *= mask
    placed = _masked_pixels(usable, amount, head, frame.shape[1])
    if placed is None:
        return
    index, rows, cols, weight = placed
    average = _small_blur(head.to_grid(luma * like), head) / np.maximum(known, 1e-3)
    level = coarse_luma + (average - coarse_luma) * _smoothstep(known, *_FILL_LOCAL_SHARE)
    brightness = head.grid_to_work(level / np.maximum(coarse_luma, _MIN_FILL_LUMA))
    cell = head.factor / head.stride
    color = coarse[(rows * cell).astype(np.intp), (cols * cell).astype(np.intp)]
    target = np.minimum(color * brightness[rows, cols][:, None], 1.0)
    pixels = frame.reshape(-1, 3)
    live = np.take(pixels, index, axis=0)
    pixels[index] = live + (target - live) * weight[:, None]


def _masked_pixels(
    mask: np.ndarray, amount: float, head: _Head, frame_width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Where ``mask`` (working copy) acts at full resolution: single-number indices into
    the frame's pixels, the same pixels' rows and columns on the working copy, and weights."""
    if amount <= 0.0:
        return None
    col, row, cols, rows = cv2.boundingRect((mask > _MIN_MASK).astype(np.uint8))
    if cols == 0 or rows == 0:
        return None
    factor = head.factor
    crop = mask[row : row + rows, col : col + cols]
    size = (cols * factor, rows * factor)
    weight = (crop if factor == 1 else cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)).ravel()
    picked = np.flatnonzero(weight > _MIN_MASK)
    if len(picked) == 0:
        return None
    ys, xs = np.divmod(picked, size[0])
    ys += row * factor
    xs += col * factor
    index = (ys + head.roi.y0) * frame_width + xs + head.roi.x0
    return index, ys // factor, xs // factor, weight[picked] * np.float32(amount)


def _small_blur(grid: np.ndarray, head: _Head) -> np.ndarray:
    size = 2 * max(1, round(_FILL_SIGMA * head.cells_per_scale)) + 1
    return cv2.blur(grid, (size, size), borderType=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Warp
# --------------------------------------------------------------------------


def _ray_amounts(volume: float, width: float, plate_level: float) -> np.ndarray:
    """How far the silhouette moves along each ray (face scales; + is outward)."""
    angles = np.linspace(0.0, 360.0, _RAY_COUNT, endpoint=False, dtype=np.float32)
    degrees = np.abs((angles + 180.0) % 360.0 - 180.0)  # from up, either way around
    top = 1.0 - _smoothstep(degrees, *_TOP_WINDOW)
    side = _smoothstep(degrees, *_SIDE_WINDOW[0]) * (1.0 - _smoothstep(degrees, *_SIDE_WINDOW[1]))
    shrink = _SHRINK_WITHOUT_PLATE + (_SHRINK_WITH_PLATE - _SHRINK_WITHOUT_PLATE) * plate_level
    return volume * (_GROW_TOP if volume > 0 else shrink) * top + width * (_GROW_SIDES if width > 0 else shrink) * side


def _face_rays(landmarks: np.ndarray, head: _Head) -> np.ndarray:
    """Distance (px) from the middle of the head to the face oval, per ray."""
    rel = landmarks[regions.FACE_OVAL].astype(np.float32) - head.center
    along_right = rel[:, 0] * head.right[0] + rel[:, 1] * head.right[1]
    along_up = rel[:, 0] * head.up[0] + rel[:, 1] * head.up[1]
    theta = np.arctan2(along_right, along_up) % (2.0 * math.pi)
    radius = np.hypot(along_right, along_up)
    order = np.argsort(theta)
    angles = np.linspace(0.0, 2.0 * math.pi, _RAY_COUNT, endpoint=False)
    return np.interp(angles, theta[order], radius[order], period=2.0 * math.pi).astype(np.float32)


def _circular_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a periodic 1-D array."""
    reach = max(1, math.ceil(3.0 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    wrapped = np.concatenate((values[-reach:], values, values[:reach]))
    return np.convolve(wrapped, kernel, mode="valid").astype(np.float32)


def _frame_edge_fade(xs: np.ndarray, ys: np.ndarray, shape: tuple[int, ...], head: _Head) -> np.ndarray:
    reach = max(_FRAME_EDGE_FADE * head.scale, 1.0)
    fade_x = np.clip(np.minimum(xs, shape[1] - xs) / reach, 0.0, 1.0)
    fade_y = np.clip(np.minimum(ys, shape[0] - ys) / reach, 0.0, 1.0)
    return fade_x * fade_y


def _person_cover(scene: _Scene, outline: _Silhouette, head: _Head, *, include_thin: bool) -> np.ndarray:
    """1 where the picture shows the person (working copy): the mass, and always the skin and
    what the mask is sure is hair, whatever the image says (bright skin may look like the wall).
    Without a plate thin structures cannot be told from lines in the room, and stay out."""
    sure = head.grid_to_work(cv2.max(outline.skin, _smoothstep(outline.hair, *_CONFIDENT_HAIR)))
    if include_thin:
        return cv2.max(cv2.max(scene.mass, sure), scene.person())
    # The mass may include a door frame: what counts as a strand depends on the slider.
    solid = (scene.mass > 0.5).astype(np.uint8)
    thick = cv2.morphologyEx(solid, cv2.MORPH_OPEN, _ellipse(_COVER_OPEN * head.work_scale))
    return cv2.max(scene.mass * cv2.dilate(thick, _RECT_5).astype(np.float32), sure)


def _warp(
    frame: np.ndarray, ctx: FrameContext, head: _Head, field: tuple[np.ndarray, np.ndarray] | None, layer: _RoomLayer | None
) -> None:
    """Apply a backward displacement field in place, within its bounding box."""
    if field is None:
        return
    dx, dy = field
    active = (np.abs(dx) + np.abs(dy)) > _MIN_DISPLACEMENT_PX
    col, row, cols, rows = cv2.boundingRect(active.astype(np.uint8))
    if cols == 0 or rows == 0:
        return
    col, row, cols, rows = _grown_box(col, row, cols, rows, 1, dx.shape)
    stride = head.stride
    size = (cols * stride, rows * stride)
    x0, y0 = head.roi.x0 + col * stride, head.roi.y0 + row * stride
    map_x = cv2.resize(dx[row : row + rows, col : col + cols], size, interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(dy[row : row + rows, col : col + cols], size, interpolation=cv2.INTER_LINEAR)
    map_x += np.arange(x0, x0 + size[0], dtype=np.float32)[None, :]
    map_y += np.arange(y0, y0 + size[1], dtype=np.float32)[:, None]
    area = (slice(y0, y0 + size[1]), slice(x0, x0 + size[0]))
    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if layer is not None and layer.strength > 0.0:
        warped = _over_room(frame, warped, ctx, layer, head, area, map_x, map_y)
    frame[area] = warped
    _warp_hair_mask(ctx, area, map_x, map_y)


def _over_room(
    frame: np.ndarray,
    warped: np.ndarray,
    ctx: FrameContext,
    layer: _RoomLayer,
    head: _Head,
    area: tuple[slice, slice],
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """The warped person over the unbent room.

    A pixel where the person's coverage after the warp is ``a`` shows ``a`` of the person and
    ``1 - a`` of the room. Where the room was visible, it stays as it was. Where the person
    used to be, the warped picture itself shows the room of the place it came from, ``q``,
    so only ``room(p) - room(q)`` is missing, which the plate tells. Without a plate that is
    unknown, but then the hair shrinks so little that it hardly matters. The plate's colors are
    never painted: where nothing moved, or the plate is plain, the warped picture stays as it
    is, however wrong the person map or the plate may be. Most pixels are all person or all
    room, which are simple copies; only the rest gets the full computation.
    """
    person = layer.cover
    if head.factor > 1:
        person = cv2.resize(person, (head.roi.w, head.roi.h), interpolation=cv2.INTER_LINEAR)
    original = frame[area]
    y0, x0 = area[0].start - head.roi.y0, area[1].start - head.roi.x0
    # Grown a little: the soft edge of the person must not stay behind as a ghost.
    was_person = cv2.dilate(person[y0 : y0 + original.shape[0], x0 : x0 + original.shape[1]], _RECT_5)
    moved = cv2.remap(
        person,
        map_x - np.float32(head.roi.x0),
        map_y - np.float32(head.roi.y0),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    result = warped.copy()
    room_only = ((moved <= _CLEAR_COVER) & (was_person <= _CLEAR_COVER)).astype(np.uint8)
    cv2.copyTo(original, room_only, result)
    picked = np.flatnonzero((moved < 1.0 - _CLEAR_COVER).ravel() & (room_only.ravel() == 0))
    if len(picked):
        width = original.shape[1]
        rows, cols = np.divmod(picked, width)
        in_frame = (rows + area[0].start) * frame.shape[1] + cols + area[1].start
        live = np.take(frame.reshape(-1, 3), in_frame, axis=0)
        bent = np.take(warped.reshape(-1, 3), picked, axis=0)
        here = np.minimum(was_person.ravel()[picked] * 2.0, 1.0)[:, None]
        uncovered = bent
        if layer.fit is not None:
            plate = ctx.background
            at_pixel = np.take(plate.reshape(-1, 3), in_frame, axis=0)
            at_source = _bilinear(plate, map_x.ravel()[picked], map_y.ravel()[picked])
            uncovered = _clip01(bent + (at_pixel - at_source) * layer.fit.at(rows + y0, cols + x0, 1.0 / head.stride))
        room = live + (uncovered - live) * here
        flat = result.reshape(-1, 3)  # result is a fresh, contiguous array
        cover = moved.ravel()[picked][:, None]
        flat[picked] = room + (bent - room) * cover
    if layer.strength >= 1.0:
        return result
    return warped + (result - warped) * np.float32(layer.strength)


def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """A BGR image at the given positions, as an N x 3 array."""
    count = len(xs)
    cols = min(count, _ROOM_SAMPLE_COLS)
    rows = -(-count // cols)
    map_x, map_y = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), np.float32)
    map_x.flat[:count], map_y.flat[:count] = xs, ys
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)[:count]


def _warp_hair_mask(ctx: FrameContext, area: tuple[slice, slice], map_x: np.ndarray, map_y: np.ndarray) -> None:
    """Keep the mask aligned with the warped hair for the effects that follow."""
    warped = ctx.hair_mask.copy()
    warped[area] = cv2.remap(ctx.hair_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped.setflags(write=False)
    ctx.hair_mask = warped


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _ellipse(radius: float) -> np.ndarray:
    size = 2 * max(1, round(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _normalized_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """``values`` averaged over the pixels where ``weight`` is set, spread by about ``sigma``."""
    total = np.maximum(_soft_blur(weight, sigma), 1e-3)
    if values.ndim == 3:
        return _soft_blur(values * weight[..., None], sigma) / total[..., None]
    return _soft_blur(values * weight, sigma) / total


def _soft_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Two box filters, about a Gaussian of ``sigma`` and much cheaper at this size."""
    size = 2 * max(1, round(sigma)) + 1
    once = cv2.blur(image, (size, size), borderType=cv2.BORDER_REPLICATE)
    return cv2.blur(once, (size, size), borderType=cv2.BORDER_REPLICATE)


def _smoothed(old: float | None, sample: float | None) -> float | None:
    if sample is None or old is None:
        return old if sample is None else sample
    return old + _STATS_EMA * (sample - old)


def _grown_box(x: int, y: int, w: int, h: int, pad: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(shape[1], x + w + pad), min(shape[0], y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def _lerp(ends: tuple[float, float], t: float) -> float:
    return ends[0] + (ends[1] - ends[0]) * t


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


def _ramp(value: float, start: float, full: float) -> float:
    """0 at ``start``, 1 at ``full`` (which may be the smaller one), linear in between."""
    return min(max((float(value) - start) / (full - start), 0.0), 1.0)


def _clip01(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:  # cv2 would treat a 1-D array as an N x 1 image
        return np.clip(image, 0.0, 1.0)
    clipped = cv2.max(image, 0.0)  # several times faster than np.clip on images
    return cv2.min(clipped, 1.0, dst=clipped)


def _distance_to_zero(image: np.ndarray) -> np.ndarray:
    """Distance of each cell to the nearest zero. With no zero at all (no hair in the mask: it happens
    in a deep bow), OpenCV's answer is a number so large that the arithmetic after it overflows."""
    return cv2.min(cv2.distanceTransform(image, cv2.DIST_L2, 3), _NO_ZERO_DISTANCE)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = x - np.float32(lo)
    t *= np.float32(1.0 / max(hi - lo, 1e-6))  # lengths can vanish in odd poses
    return _smoothstep_t(t)


def _smoothstep_t(t: np.ndarray) -> np.ndarray:
    """``t * t * (3 - 2 * t)`` of ``t`` clipped to [0, 1], without temporary images: one
    just big enough costs about 0.15 ms, as the OS page-faults fresh memory of a megabyte or more."""
    t = _clip01(t)
    rest = np.multiply(t, 2.0)
    np.subtract(3.0, rest, out=rest)
    t *= t
    t *= rest
    return t


def _above(image: np.ndarray, level: float) -> np.ndarray:
    """``(image > level).astype(float32)`` for a float32 image and level, at a third of the cost."""
    return cv2.threshold(image, float(level), 1.0, cv2.THRESH_BINARY)[1]

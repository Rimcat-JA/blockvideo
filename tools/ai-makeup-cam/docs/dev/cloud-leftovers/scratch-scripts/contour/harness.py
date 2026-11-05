"""Run tracker + segmentation on frames and apply skin -> contour -> tearbag with variants."""
import importlib
from dataclasses import replace

import cv2
import numpy as np

from clips import *  # noqa
from makeup_cam import regions
from makeup_cam.config import MakeupSettings, ContourSettings, TearbagSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects.skin import SkinEffect
from makeup_cam.segmentation import Segmenter
from makeup_cam.tracking import FaceTracker
import makeup_cam.effects.contour as C


def contexts(frames, smoothing=0.5, interval=2):
    """Yield (frame_uint8, ctx) per frame."""
    tracker = FaceTracker(S / "models" / "face_landmarker.task", smoothing=smoothing)
    seg = Segmenter(S / "models" / "selfie_multiclass_256x256.tflite", interval=interval, background=False)
    try:
        for i, f in enumerate(frames):
            ts = 33 * (i + 1)
            lm, z = tracker.detect(f, ts)
            m = seg.update(f, ts, i + 1)
            h, w = f.shape[:2]
            yield f, FrameContext(h, w, lm, z, m.hair, m.face_skin, m.body_skin)
    finally:
        tracker.close(); seg.close()


def made_up(f, ctx, contour=None, tearbag=None, skin=True, effects=None):
    img = f.astype(np.float32) / 255.0
    if ctx.landmarks is None:
        return f.copy()
    if skin:
        img = SkinEffect().apply(img, ctx, MakeupSettings().skin)
    if contour is not None:
        img = C.ContourEffect().apply(img, ctx, contour)
    if tearbag is not None:
        eff = effects["tearbag"] if effects else C.TearbagEffect()
        img = eff.apply(img, ctx, tearbag)
    return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


def weight_map(f, ctx, settings):
    """Gray face with shade (blue) and light (red) weights, normalized for visibility."""
    lm = ctx.landmarks
    scale = regions.face_scale(lm)
    roi = regions.roi_around(lm[regions.FACE_OVAL], f.shape, pad=0.0, margin_px=C._ROI_MARGIN * scale)
    grid = C._Grid.over(roi, scale)
    amounts = {"shade": settings.shading, "light": settings.highlight}
    strokes = [s for s in C._STROKES if amounts[s.tone] * getattr(settings, s.area) > 0]
    shade, light = C._contour_weights(grid, lm, scale, [s for s in C._STROKE_SIDES if amounts[s.stroke.tone] * getattr(settings, s.stroke.area) > 0], settings, amounts, ctx)
    shade = grid.upsample(shade) if shade is not None else np.zeros((roi.h, roi.w), np.float32)
    light = grid.upsample(light) if light is not None else np.zeros((roi.h, roi.w), np.float32)
    pass
    g = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255
    v = g[roi.slices]
    s = shade / max(shade.max(), 1e-6); l = light / max(light.max(), 1e-6)
    v[..., 0] = np.clip(v[..., 0] + s * 0.8, 0, 1); v[..., 1] *= 1 - 0.5 * (s + l).clip(0, 1); v[..., 2] = np.clip(v[..., 2] + l * 0.8, 0, 1)
    return (g * 255).astype(np.uint8), roi


def crop(img, lm, pad=0.12, height=420):
    box = regions.roi_around(lm[regions.FACE_OVAL], img.shape, pad=pad)
    c = img[box.slices]
    f = height / c.shape[0]
    return cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def still(name):
    if name == "frame_near":
        src = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
        big = cv2.resize(src, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(big[0:720, 560:1840])
    return cv2.imread(str(S / "testimgs" / f"{name}.png"))


def hstack(imgs):
    h = max(i.shape[0] for i in imgs)
    return np.hstack([cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs])


def vstack(imgs):
    w = max(i.shape[1] for i in imgs)
    return np.vstack([cv2.copyMakeBorder(i, 0, 2, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs])

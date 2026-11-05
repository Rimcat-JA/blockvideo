"""Visual tuning: run the real MakeupPipeline with only SkinEffect and save before/after crops."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
from makeup_cam.config import MakeupSettings  # noqa: E402
from makeup_cam.effects.skin import SkinEffect  # noqa: E402
from makeup_cam.pipeline import MakeupPipeline  # noqa: E402
from makeup_cam.segmentation import Segmenter  # noqa: E402
from makeup_cam.tracking import FaceTracker  # noqa: E402
from makeup_cam import regions  # noqa: E402

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = S / "renders" / "skin"
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "default": {},
    "strong": dict(smoothing=1.0, texture=0.2, blemish=1.0, redness=1.0, under_eye=1.0, tone_strength=0.5, brightness=0.2),
    "smooth_max": dict(smoothing=1.0, texture=0.0, blemish=0.0, redness=0.0, under_eye=0.0, tone_strength=0.0, brightness=0.0),
    "color_max": dict(smoothing=0.0, blemish=0.0, redness=1.0, under_eye=1.0, tone_strength=1.0, brightness=0.3),
    "smooth_only": dict(blemish=0.0, redness=0.0, under_eye=0.0, tone_strength=0.0, brightness=0.0),
    "blemish_only": dict(smoothing=0.0, blemish=1.0, redness=0.0, under_eye=0.0, tone_strength=0.0, brightness=0.0),
    "under_eye_only": dict(smoothing=0.0, blemish=0.0, redness=0.0, under_eye=1.0, tone_strength=0.0, brightness=0.0),
    "redness_only": dict(smoothing=0.0, blemish=0.0, redness=1.0, under_eye=0.0, tone_strength=0.0, brightness=0.0),
    "bright_max": dict(smoothing=0.0, blemish=0.0, redness=0.0, under_eye=0.0, tone_strength=0.0, brightness=0.3),
    "dark_tone": dict(tone_color="#8D5A3C", tone_strength=0.6, brightness=-0.2),
}


def close_frame() -> np.ndarray:
    """A webcam-like close framing (face scale ~120 px) cut from the astronaut portrait."""
    from skimage import data
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    crop = img[10:240, 60:470]
    return cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_CUBIC)


def run(frame: np.ndarray, overrides: dict) -> tuple[np.ndarray, np.ndarray]:
    settings = MakeupSettings()
    for k, v in overrides.items():
        setattr(settings.skin, k, v)
    settings.general.mirror_preview = False
    tracker = FaceTracker(S / "models" / "face_landmarker.task", 0.0)
    seg = Segmenter(S / "models" / "selfie_multiclass_256x256.tflite", interval=1, background=False)
    pipe = MakeupPipeline(settings, tracker=tracker, segmenter=seg, effects=[SkinEffect()])
    result = None
    for i in range(10):
        result = pipe.process(frame, 33 * (i + 1))
    pipe.close()
    return result.output, result.landmarks


def crop_box(lm: np.ndarray, shape) -> regions.Roi:
    return regions.roi_around(lm[regions.FACE_OVAL], shape, pad=0.12)


def panel(before, after, roi, height=640):
    b = before[roi.slices]
    a = after[roi.slices]
    diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    f = height / b.shape[0]
    tiles = [cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) for t in (b, a, diff)]
    return np.hstack(tiles)


def detail_box(lm: np.ndarray, shape) -> regions.Roi:
    """Right cheek, under-eye and nose: where texture and spots show."""
    s = regions.face_scale(lm)
    pts = np.array([lm[regions.RIGHT_EYE_OUTER_CORNER] + [-0.15 * s, 0], lm[regions.NOSE_TIP] + [0.25 * s, 0.25 * s]])
    return regions.roi_around(pts, shape, pad=0.0)


def main(names: list[str], presets: list[str]) -> None:
    frames = {n: cv2.imread(str(S / "testimgs" / f"{n}.png")) for n in names if n != "frame_close"}
    if "frame_close" in names:
        frames["frame_close"] = close_frame()
    for name, frame in frames.items():
        for preset in presets:
            out, lm = run(frame, PRESETS[preset])
            roi = crop_box(lm, frame.shape)
            path = OUT / f"{name}_{preset}.png"
            top = panel(frame, out, roi)
            bottom = panel(frame, out, detail_box(lm, frame.shape), height=480)
            pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), np.uint8) if bottom.shape[1] < top.shape[1] else None
            bottom = np.hstack([bottom, pad]) if pad is not None else bottom[:, : top.shape[1]]
            cv2.imwrite(str(path), np.vstack([top, bottom]))
            print("wrote", path)


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["frame_720p", "frame_dark", "frame_tilt", "frame_far", "frame_close"]
    presets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PRESETS)
    main(names, presets)

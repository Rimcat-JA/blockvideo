"""Visual tuning: run the real MakeupPipeline with only HairEffect and save before/after crops.

usage: render.py [preset ...]   (default: all presets)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
from makeup_cam import regions  # noqa: E402
from makeup_cam.config import MakeupSettings  # noqa: E402
from makeup_cam.effects.hair import HairEffect  # noqa: E402
from makeup_cam.pipeline import MakeupPipeline  # noqa: E402
from makeup_cam.segmentation import Segmenter  # noqa: E402
from makeup_cam.tracking import FaceTracker  # noqa: E402

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = S / "renders" / "hair"
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "default": {},
    "gloss_max": dict(gloss=1.0, smoothness=0.0),
    "smooth_max": dict(gloss=0.0, smoothness=1.0),
    "interview_dark": dict(color_strength=0.8, gloss=0.4, smoothness=0.4),
    "red_dye": dict(color="#8E2F2A", color_strength=1.0, gloss=0.0, smoothness=0.0),
    "strong": dict(gloss=1.0, smoothness=1.0, color="#6B2E2A", color_strength=1.0, brightness=0.3, contrast=0.4, edge_softness=1.0),
    "tone": dict(gloss=0.0, smoothness=0.0, brightness=-0.5, contrast=0.5),
    "bright": dict(gloss=0.0, smoothness=0.0, brightness=0.5, contrast=-0.5),
}
FRAMES = ["frame_720p", "frame_dark", "frame_tilt", "frame_far", "close", "actor_1072", "actor_1002", "actor_1029", "actor_1076"]


def close_frame() -> np.ndarray:
    """A webcam-like close framing (face scale ~120 px) cut from the astronaut portrait."""
    from skimage import data

    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    crop = img[0:230, 60:470]
    return cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_CUBIC)


def dark_hair_frame() -> np.ndarray:
    """frame_720p with the hair darkened to near-black brown (soft mask), to judge dark hair."""
    frame = cv2.imread(str(S / "testimgs" / "frame_720p.png")).astype(np.float32) / 255.0
    hair = np.load(S / "tmp_hair" / "cache_frame_720p.npz")["hair"]
    m = cv2.GaussianBlur(np.clip((hair - 0.25) / 0.45, 0, 1), (0, 0), 1.5)[..., None]
    dark = frame * np.array([0.30, 0.24, 0.22], np.float32)
    return np.clip((frame * (1 - m) + dark * m) * 255 + 0.5, 0, 255).astype(np.uint8)


def load(name: str) -> np.ndarray:
    if name == "close":
        return close_frame()
    if name == "dark_hair":
        return dark_hair_frame()
    if name.startswith("actor_"):
        return cv2.imread(str(S / "tmp_hair" / f"{name}.png"))
    return cv2.imread(str(S / "testimgs" / f"{name}.png"))


def run(frame: np.ndarray, overrides: dict) -> tuple[np.ndarray, np.ndarray | None, dict]:
    settings = MakeupSettings()
    for k, v in overrides.items():
        setattr(settings.hair, k, v)
    settings.general.mirror_preview = False
    tracker = FaceTracker(S / "models" / "face_landmarker.task", 0.0)
    seg = Segmenter(S / "models" / "selfie_multiclass_256x256.tflite", interval=1, background=False)
    pipe = MakeupPipeline(settings, tracker=tracker, segmenter=seg, effects=[HairEffect()])
    result = None
    for i in range(8):
        result = pipe.process(frame, 33 * (i + 1))
    pipe.close()
    return result.output, result.landmarks, result.timings_ms


def hair_box(frame: np.ndarray, lm: np.ndarray) -> regions.Roi:
    s = regions.face_scale(lm)
    oval = lm[regions.FACE_OVAL]
    lo = oval.min(axis=0) - [1.3 * s, 1.6 * s]
    hi = oval.max(axis=0) + [1.3 * s, -0.8 * s]
    return regions.roi_around(np.array([lo, hi]), frame.shape, pad=0.0)


def detail_box(frame: np.ndarray, lm: np.ndarray) -> regions.Roi:
    """Top of the head and the hairline on one side, for strand-level detail."""
    s = regions.face_scale(lm)
    top = lm[10]
    pts = np.array([top + [-1.6 * s, -1.6 * s], top + [0.2 * s, 0.4 * s]])
    return regions.roi_around(pts, frame.shape, pad=0.0)


def top_box(frame: np.ndarray, lm: np.ndarray) -> regions.Roi:
    """The top of the head (angel-ring area)."""
    s = regions.face_scale(lm)
    top = lm[10]
    pts = np.array([top + [-1.5 * s, -1.4 * s], top + [1.5 * s, 0.3 * s]])
    return regions.roi_around(pts, frame.shape, pad=0.0)


def panel(before: np.ndarray, after: np.ndarray, roi: regions.Roi, height: int = 560) -> np.ndarray:
    b, a = before[roi.slices], after[roi.slices]
    diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 6, 0, 255).astype(np.uint8)
    f = height / b.shape[0]
    tiles = [cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) for t in (b, a, diff)]
    return np.hstack(tiles)


def main(presets: list[str]) -> None:
    for preset in presets:
        rows, top_rows = [], []
        for name in FRAMES:
            frame = load(name)
            out, lm, timings = run(frame, PRESETS[preset])
            if lm is None:
                print(name, "no face")
                continue
            rows.append(panel(frame, out, hair_box(frame, lm), 420))
            print(f"{preset:15s} {name:11s} hair {timings.get('hair', float('nan')):.2f} ms")
            if name in ("frame_720p", "close", "actor_1072", "actor_1002"):
                cv2.imwrite(str(OUT / f"{preset}_{name}.png"), panel(frame, out, hair_box(frame, lm)))
            if name == "close":
                cv2.imwrite(str(OUT / f"{preset}_detail.png"), panel(frame, out, detail_box(frame, lm), 640))
            if name.startswith("actor_"):
                top_rows.append(panel(frame, out, top_box(frame, lm), 300))
        width = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
        cv2.imwrite(str(OUT / f"{preset}_all.png"), np.vstack(rows))
        width = max(r.shape[1] for r in top_rows)
        top_rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in top_rows]
        cv2.imwrite(str(OUT / f"{preset}_tops.png"), np.vstack(top_rows))


if __name__ == "__main__":
    main(sys.argv[1:] or list(PRESETS))

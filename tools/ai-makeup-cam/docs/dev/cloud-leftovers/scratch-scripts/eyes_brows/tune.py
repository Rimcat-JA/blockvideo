"""Visual tuning: real MakeupPipeline with only eyeshadow / eyeliner / brows,
zoomed before/after/diff crops of the eye area into renders/eyes_brows/.

usage: python tune.py eyeshadow|eyeliner|brows|all [tag] [images...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import MakeupSettings
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.effects.eyes import EyeshadowEffect, EyelinerEffect
from makeup_cam.effects.brows import BrowEffect

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = S / "renders" / "eyes_brows"
IMAGES = ["frame_720p", "frame_dark", "frame_tilt", "frame_far", "frame_near"]
MINE = ("eyeshadow", "eyeliner", "brows")

VARIANTS = {
    "eyeshadow": [
        ("default", {}),
        ("strong_shimmer", {"opacity": 0.7, "height": 0.8, "softness": 0.8, "shimmer": 0.8}),
        ("max_hard_plum", {"color": "#6E3B5C", "opacity": 1.0, "height": 1.0, "softness": 0.0, "shimmer": 1.0}),
    ],
    "eyeliner": [
        ("default", {}),
        ("cat_eye", {"opacity": 0.9, "thickness": 0.8, "wing": 0.8, "lower": 0.4, "softness": 0.2}),
        ("max_smudge", {"opacity": 1.0, "thickness": 1.0, "wing": 1.0, "lower": 1.0, "softness": 1.0, "color": "#101010"}),
    ],
    "brows": [
        ("default", {}),
        ("strong_thick", {"opacity": 0.8, "thickness": 0.3, "softness": 0.3}),
        ("max_thin_hard", {"opacity": 1.0, "thickness": -0.5, "softness": 0.0, "color": "#2A1D16"}),
    ],
    "all": [
        ("default", {}),
        ("strong", {"eyeshadow": {"opacity": 0.6, "height": 0.7, "shimmer": 0.5},
                    "eyeliner": {"opacity": 0.85, "thickness": 0.7, "wing": 0.6, "lower": 0.3},
                    "brows": {"opacity": 0.7, "thickness": 0.2}}),
    ],
}


def load(name: str) -> np.ndarray:
    if name == "frame_near":  # synthetic close-up (webcam-typical face size)
        src = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
        big = cv2.resize(src, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(big[0:720, 560:1840])
    return cv2.imread(str(S / "testimgs" / f"{name}.png"))


def zoom(img, box, width):
    crop = img[box.slices]
    f = width / crop.shape[1]
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def settings_for(effect: str, values: dict) -> MakeupSettings:
    s = MakeupSettings()
    for section in ("skin", "lips", "blush", "hair", *MINE):
        getattr(s, section).enabled = section in MINE and (effect == "all" or section == effect)
    for key, value in values.items():
        if isinstance(value, dict):
            for k, v in value.items():
                setattr(getattr(s, key), k, v)
        else:
            setattr(getattr(s, effect), key, value)
    return s


def main() -> None:
    effect = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    images = sys.argv[3:] or IMAGES
    width = 900
    for name in images:
        img = load(name)
        pipe = MakeupPipeline(settings_for(effect, {}), model_dir=S / "models",
                              effects=[EyeshadowEffect(), EyelinerEffect(), BrowEffect()])
        ts = 0
        for _ in range(8):
            ts += 33
            res = pipe.process(img, ts)
        lm = res.landmarks
        pts = np.vstack([lm[regions.LEFT_BROW], lm[regions.RIGHT_BROW], lm[regions.LEFT_EYE], lm[regions.RIGHT_EYE]])
        box = regions.roi_around(pts, img.shape, pad=0.12)
        rows = []
        for vname, values in VARIANTS[effect]:
            pipe.set_settings(settings_for(effect, values))
            ts += 33
            res = pipe.process(img, ts)
            ms = sum(res.timings_ms.get(k, 0.0) for k in MINE)
            before = label(zoom(img, box, width), f"{name} before")
            after = label(zoom(res.output, box, width), f"{vname} ({ms:.1f} ms)")
            diff = cv2.absdiff(res.output, img).astype(np.float32)
            diffz = label(zoom(np.clip(diff * 4, 0, 255).astype(np.uint8), box, width), "diff x4")
            rows.append(np.vstack([before, after, diffz]) if False else np.hstack([after, diffz]))
            if vname == "default":
                rows.insert(0, np.hstack([before, np.zeros_like(before)]))
        pipe.close()
        sheet = np.vstack(rows)
        path = OUT / f"{effect}_{name}{('_' + tag) if tag else ''}.png"
        cv2.imwrite(str(path), sheet)
        print("wrote", path, sheet.shape)


if __name__ == "__main__":
    main()

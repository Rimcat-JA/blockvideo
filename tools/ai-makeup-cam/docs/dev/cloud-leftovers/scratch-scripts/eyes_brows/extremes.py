"""Extreme settings close-ups (real pipeline, near 2x frame and far frame)."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_eyes_brows")
from tune import load, settings_for, label, S, OUT
from makeup_cam import regions
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.effects.eyes import EyeshadowEffect, EyelinerEffect
from makeup_cam.effects.brows import BrowEffect

VARIANTS = [
    ("shadow max soft0", "eyeshadow", {"opacity": 1.0, "height": 1.0, "softness": 0.0, "shimmer": 1.0}),
    ("shadow max soft1", "eyeshadow", {"opacity": 1.0, "height": 1.0, "softness": 1.0, "shimmer": 0.0, "color": "#3A2A24"}),
    ("shadow low h .15", "eyeshadow", {"opacity": 1.0, "height": 0.15, "softness": 0.0}),
    ("liner max soft0", "eyeliner", {"opacity": 1.0, "thickness": 1.0, "wing": 1.0, "lower": 1.0, "softness": 0.0, "color": "#000000"}),
    ("brows max thick", "brows", {"opacity": 1.0, "thickness": 0.5, "softness": 0.0, "color": "#1E140F"}),
    ("brows max soft", "brows", {"opacity": 1.0, "thickness": 0.5, "softness": 1.0}),
    ("all strong", "all", {"eyeshadow": {"opacity": 1.0, "height": 0.8, "shimmer": 0.7, "color": "#8A4B3C"},
                           "eyeliner": {"opacity": 1.0, "thickness": 0.8, "wing": 0.7, "lower": 0.6},
                           "brows": {"opacity": 0.9, "thickness": 0.2}}),
]
for name in ("frame_near", "frame_far"):
    img = load(name)
    pipe = MakeupPipeline(settings_for("all", {}), model_dir=S / "models", effects=[EyeshadowEffect(), EyelinerEffect(), BrowEffect()])
    ts = 0
    for _ in range(8):
        ts += 33
        res = pipe.process(img, ts)
    lm = res.landmarks
    s = regions.side_indices("right")
    box = regions.roi_around(np.vstack([lm[s.eye], lm[s.brow]]), img.shape, pad=0.15)
    width = 600
    def zoom(im):
        crop = im[box.slices]
        return cv2.resize(crop, (width, round(crop.shape[0] * width / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
    tiles = [label(zoom(img), "before")]
    for vname, effect, values in VARIANTS:
        pipe.set_settings(settings_for(effect, values))
        ts += 33
        tiles.append(label(zoom(pipe.process(img, ts).output), vname))
    pipe.close()
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)])
    cv2.imwrite(str(OUT / f"extremes_{name}.png"), grid)
    print("wrote", OUT / f"extremes_{name}.png")

"""Close single-eye before/after crops through the real pipeline (near 2x frame + 720p).

usage: python closeup.py tag  (variants defined below)
"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_eyes_brows")
from tune import load, settings_for, label, S, OUT, MINE
from makeup_cam import regions
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.effects.eyes import EyeshadowEffect, EyelinerEffect
from makeup_cam.effects.brows import BrowEffect

VARIANTS = [
    ("default all", "all", {}),
    ("shadow default", "eyeshadow", {}),
    ("shadow shimmer .6", "eyeshadow", {"shimmer": 0.6}),
    ("liner default", "eyeliner", {}),
    ("brows default", "brows", {}),
]
tag = sys.argv[1]
for name in ("frame_near", "frame_720p"):
    img = load(name)
    pipe = MakeupPipeline(settings_for("all", {}), model_dir=S / "models", effects=[EyeshadowEffect(), EyelinerEffect(), BrowEffect()])
    ts = 0
    for _ in range(8):
        ts += 33
        res = pipe.process(img, ts)
    lm = res.landmarks
    s = regions.side_indices("left")
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
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 6, 3)])
    cv2.imwrite(str(OUT / f"closeup_{name}_{tag}.png"), grid)
    print("wrote", OUT / f"closeup_{name}_{tag}.png")

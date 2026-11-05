"""Final gallery: before | defaults (all three) | strong look, for every test frame (real pipeline)."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_eyes_brows")
from tune import load, settings_for, label, S, OUT, IMAGES, VARIANTS
from makeup_cam import regions
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.effects.eyes import EyeshadowEffect, EyelinerEffect
from makeup_cam.effects.brows import BrowEffect

rows, timings = [], {}
for name in IMAGES:
    img = load(name)
    pipe = MakeupPipeline(settings_for("all", {}), model_dir=S / "models", effects=[EyeshadowEffect(), EyelinerEffect(), BrowEffect()])
    ts = 0
    for _ in range(8):
        ts += 33
        res = pipe.process(img, ts)
    lm = res.landmarks
    pts = np.vstack([lm[regions.LEFT_BROW], lm[regions.RIGHT_BROW], lm[regions.LEFT_EYE], lm[regions.RIGHT_EYE]])
    box = regions.roi_around(pts, img.shape, pad=0.12)
    def zoom(im):
        crop = im[box.slices]
        return cv2.resize(crop, (600, round(crop.shape[0] * 600 / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
    tiles = [label(zoom(img), f"{name} before")]
    for vname, values in VARIANTS["all"]:
        pipe.set_settings(settings_for("all", values))
        runs = []
        for _ in range(30 if vname == "default" else 1):
            ts += 33
            res = pipe.process(img, ts)
            runs.append({k: res.timings_ms[k] for k in ("eyeshadow", "eyeliner", "brows")})
        if vname == "default":
            timings[name] = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
        tiles.append(label(zoom(res.output), vname))
    pipe.close()
    h = min(t.shape[0] for t in tiles)
    rows.append(np.hstack([t[:h] for t in tiles]))
w = max(r.shape[1] for r in rows)
cv2.imwrite(str(OUT / "final_gallery.png"), np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows]))
for name, t in timings.items():
    print(name, {k: round(v, 2) for k, v in t.items()}, "sum", round(sum(t.values()), 2))

"""Landmarks for the test images (cached) + a zoomed eye-area crop with key points drawn."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import MakeupSettings
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.effects.base import Effect

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
TMP = S / "tmp_eyes_brows"
OUT = S / "renders" / "eyes_brows"

class Nop(Effect):
    name = "eyeshadow"
    order = 1
    def apply(self, frame, ctx, settings):
        return frame

for name in ["frame_720p", "frame_dark", "frame_tilt", "frame_far"]:
    img = cv2.imread(str(S / "testimgs" / f"{name}.png"))
    pipe = MakeupPipeline(MakeupSettings(), model_dir=S / "models", effects=[Nop()])
    ts = 0
    for _ in range(8):
        ts += 33
        res = pipe.process(img, ts)
    pipe.close()
    lm = res.landmarks.copy()
    np.save(TMP / f"lm_{name}.npy", lm)
    print(name, "scale", regions.face_scale(lm))
    if name == "frame_720p":
        pts = np.vstack([lm[regions.LEFT_BROW], lm[regions.RIGHT_BROW], lm[regions.LEFT_EYE], lm[regions.RIGHT_EYE]])
        box = regions.roi_around(pts, img.shape, pad=0.08)
        f = 5
        crop = cv2.resize(img[box.slices], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        def P(p):
            return (int((p[0]-box.x0)*f), int((p[1]-box.y0)*f))
        for idxs, col in [(regions.LEFT_EYE_UPPER,(0,255,0)), (regions.RIGHT_EYE_UPPER,(0,255,0)),
                          (regions.LEFT_EYE_LOWER,(255,255,0)), (regions.RIGHT_EYE_LOWER,(255,255,0)),
                          (regions.LEFT_BROW_UPPER,(0,0,255)), (regions.RIGHT_BROW_UPPER,(0,0,255)),
                          (regions.LEFT_BROW_LOWER,(255,0,255)), (regions.RIGHT_BROW_LOWER,(255,0,255)),
                          (regions.LEFT_EYE_CREASE,(0,200,255)), (regions.RIGHT_EYE_CREASE,(0,200,255))]:
            for i, k in enumerate(idxs):
                cv2.circle(crop, P(lm[k]), 4 if i == 0 else 2, col, -1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / "probe_points.png"), crop)
        cv2.imwrite(str(OUT / "probe_plain.png"), cv2.resize(img[box.slices], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))

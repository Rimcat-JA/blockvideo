"""Max / 99th-percentile per-pixel change (8-bit levels) of each effect at given settings."""
import sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import EyeshadowSettings, EyelinerSettings, BrowSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import eyes, brows

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
TMP = S / "tmp_eyes_brows"
img = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
lm = np.load(TMP / "lm_frame_720p.npy")
img = np.ascontiguousarray(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)[0:720, 560:1840])
lm = lm * 2 - np.array([560, 0], np.float32)
f = img.astype(np.float32) / 255
ctx = FrameContext(720, 1280, landmarks=lm)
cases = [(eyes.EyeshadowEffect(), EyeshadowSettings()), (eyes.EyeshadowEffect(), EyeshadowSettings(shimmer=1.0)),
         (eyes.EyelinerEffect(), EyelinerSettings()), (brows.BrowEffect(), BrowSettings())]
for eff, st in cases:
    out = eff.apply(f.copy(), ctx, st)
    d = (out - f) * 255
    dl = d.mean(axis=2)
    print(f"{eff.name:10s} darkest {dl.min():6.1f}  p1 {np.percentile(dl[dl!=0], 1):6.1f}  brightest {dl.max():5.1f} changed px {np.count_nonzero(np.abs(dl) > 1)}")

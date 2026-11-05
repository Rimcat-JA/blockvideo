"""Effect timings (in place, ~30 runs) at the 720p test frame (scale 58) and a 2x close-up (scale 117)."""
import sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import EyeshadowSettings, EyelinerSettings, BrowSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import eyes, brows
from tests.conftest import build_face_frame_720p

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
TMP = S / "tmp_eyes_brows"

def timeit(fn, n=30):
    fn()
    best = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best.append(time.perf_counter() - t)
    return np.median(best) * 1000, np.min(best) * 1000

frames = []
img = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
lm = np.load(TMP / "lm_frame_720p.npy")
frames.append(("frame_720p", img, lm))
frames.append(("near_2x", np.ascontiguousarray(cv2.resize(img, None, fx=2, fy=2)[0:720, 560:1840]), lm * 2 - np.array([560, 0], np.float32)))
frames.append(("face_frame_720p(fixture)", build_face_frame_720p(), np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)))
for name, img, lm in frames:
    f = img.astype(np.float32) / 255
    ctx = FrameContext(720, 1280, landmarks=lm)
    print(f"{name} scale {regions.face_scale(lm):.0f}")
    for label, eff, st in [("eyeshadow", eyes.EyeshadowEffect(), EyeshadowSettings()),
                           ("eyeshadow+shimmer", eyes.EyeshadowEffect(), EyeshadowSettings(shimmer=0.8)),
                           ("eyeliner", eyes.EyelinerEffect(), EyelinerSettings()),
                           ("eyeliner+lower", eyes.EyelinerEffect(), EyelinerSettings(lower=0.5, wing=1, thickness=1, softness=1)),
                           ("brows", brows.BrowEffect(), BrowSettings())]:
        med, mn = timeit(lambda: eff.apply(f, ctx, st))
        print(f"  {label:18s} median {med:.2f} ms  min {mn:.2f} ms")

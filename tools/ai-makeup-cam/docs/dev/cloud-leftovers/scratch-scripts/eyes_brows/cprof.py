import cProfile, pstats, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.config import EyeshadowSettings, EyelinerSettings, BrowSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import eyes, brows
S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
img = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
lm = np.load(S / "tmp_eyes_brows" / "lm_frame_720p.npy")
f = img.astype(np.float32) / 255
ctx = FrameContext(720, 1280, landmarks=lm)
which = sys.argv[1]
eff, st = {"shadow": (eyes.EyeshadowEffect(), EyeshadowSettings()), "liner": (eyes.EyelinerEffect(), EyelinerSettings()),
           "brows": (brows.BrowEffect(), BrowSettings())}[which]
eff.apply(f, ctx, st)
pr = cProfile.Profile()
pr.enable()
for _ in range(100):
    eff.apply(f, ctx, st)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)

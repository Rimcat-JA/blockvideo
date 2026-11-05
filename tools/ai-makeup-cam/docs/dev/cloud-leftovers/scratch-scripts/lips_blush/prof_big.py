import sys, cProfile, pstats, time
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests")
from conftest import FACE_LANDMARKS_720P, build_face_frame_720p
from makeup_cam import regions
from makeup_cam.config import LipSettings, BlushSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects.lips import LipEffect
from makeup_cam.effects.blush import BlushEffect
f0 = build_face_frame_720p().astype(np.float32) / 255
lm0 = np.load(FACE_LANDMARKS_720P).astype(np.float32)
zoom = float(sys.argv[2])
c = lm0[regions.NOSE_TIP]
m = np.float32([[zoom, 0, 640 - zoom * c[0]], [0, zoom, 360 - zoom * c[1]]])
frame = cv2.warpAffine(f0, m, (1280, 720))
lm = (lm0 @ m[:, :2].T + m[:, 2]).astype(np.float32)
ctx = FrameContext(720, 1280, landmarks=lm)
e, st = (LipEffect(), LipSettings(finish="gloss", gloss=1.0)) if sys.argv[1] == "lips" else (BlushEffect(), BlushSettings())
frames = [frame.copy() for _ in range(20)]
pr = cProfile.Profile(); pr.enable()
for f in frames: e.apply(f, ctx, st)
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(10)

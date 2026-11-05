import sys
import cv2
import numpy as np
if len(sys.argv) > 1:
    cv2.setNumThreads(int(sys.argv[1]))
from tests.conftest import build_face_frame_720p
from makeup_cam.config import EyeshadowSettings, MakeupSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import eyes, blush
lm = np.load("tests/data/face_landmarks_720p.npy").astype(np.float32)
f = build_face_frame_720p().astype(np.float32) / 255
ctx = FrameContext(720, 1280, landmarks=lm)
for eff, st in [(eyes.EyeshadowEffect(), EyeshadowSettings()), (blush.BlushEffect(), MakeupSettings().blush)]:
    outs = [eff.apply(f.copy(), ctx, st) for _ in range(30)]
    print(eff.name, "threads", cv2.getNumThreads(), "nondeterministic runs:", sum(not np.array_equal(outs[0], o) for o in outs))

import sys, numpy as np, cProfile, pstats
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests")
from makeup_cam.config import SkinSettings
from makeup_cam.effects.skin import SkinEffect
from makeup_cam.effects.base import FrameContext
from conftest import build_face_frame_720p
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)
frame = build_face_frame_720p().astype(np.float32) / 255
seg = np.ones((720, 1280), np.float32)
eff = SkinEffect(); st = SkinSettings()
for i in range(5): eff.apply(frame, FrameContext(720, 1280, landmarks=lm, face_skin_mask=seg), st)
def run():
    for i in range(200):
        eff.apply(frame, FrameContext(720, 1280, landmarks=lm, face_skin_mask=seg), st)
cProfile.run("run()", "prof.out")
p = pstats.Stats("prof.out"); p.sort_stats("tottime").print_stats(25)

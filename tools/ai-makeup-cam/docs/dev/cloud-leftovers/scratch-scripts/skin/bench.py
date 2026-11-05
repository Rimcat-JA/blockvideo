import sys, time, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests")
from makeup_cam.config import SkinSettings
from makeup_cam.effects.skin import SkinEffect
from makeup_cam.effects.base import FrameContext
from conftest import build_face_frame_720p
S = "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/"
import render
def bench(frame_u8, lm, label, seg=None, n=40):
    frame = frame_u8.astype(np.float32) / 255
    eff = SkinEffect(); st = SkinSettings()
    times = []
    for i in range(n):
        f = frame.copy()
        ctx = FrameContext(720, 1280, landmarks=lm, face_skin_mask=seg)
        t = time.perf_counter(); eff.apply(f, ctx, st); times.append((time.perf_counter() - t) * 1000)
    times = np.array(times[5:])
    print(f"{label}: median {np.median(times):.2f} ms  p90 {np.percentile(times, 90):.2f} ms")
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)
fr = build_face_frame_720p()
seg = np.zeros((720, 1280), np.float32); seg[90:270, 520:680] = 1.0
bench(fr, lm, "face_frame_720p (no seg)")
bench(fr, lm, "face_frame_720p (seg)", seg)
close = render.close_frame()
from makeup_cam.tracking import FaceTracker
tr = FaceTracker(S + "models/face_landmarker.task", 0.0)
lmc = None
for i in range(3): lmc, _ = tr.detect(close, 33 * (i + 1))
tr.close()
from makeup_cam import regions
print("close scale", regions.face_scale(lmc))
segc = np.ones((720, 1280), np.float32)
bench(close, lmc, "close (seg)", segc)

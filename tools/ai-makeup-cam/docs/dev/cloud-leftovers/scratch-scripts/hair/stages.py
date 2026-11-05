"""Per-function timings of HairEffect.apply (wraps module functions)."""
import sys, time, collections
import numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:]
import bench
from makeup_cam.effects import hair as H
from makeup_cam import regions, blend
from makeup_cam.effects.base import FrameContext

acc = collections.defaultdict(float)
def wrap(mod, name):
    f = getattr(mod, name)
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); acc[name] += time.perf_counter() - t; return r
    setattr(mod, name, g)
for n in ["_hair_area", "_head_blobs", "_hair_weights", "_inner_face", "_measure", "_color_cast", "_comb", "_gloss_zone", "_strand_detail", "_look", "_tone", "_hair_luminance", "_angel_ring", "_height_above_eyes"]:
    wrap(H, n)
name = sys.argv[1] if len(sys.argv) > 1 else "frame_720p"
frame, lm, hair, face = bench.load(name)
base = frame.astype(np.float32) / 255.0
eff = H.HairEffect()
ctx = FrameContext(720, 1280, landmarks=lm, hair_mask=hair, face_skin_mask=face)
s = bench.SETTINGS["all"]
N = 50
imgs = [base.copy() for _ in range(N)]
t0 = time.perf_counter()
for i in range(N):
    eff.apply(imgs[i], ctx, s)
tot = (time.perf_counter() - t0) / N * 1000
print(f"total {tot:.2f} ms")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:18s} {v / N * 1000:.3f} ms")
a = H._hair_area(H._hair_blobs(hair), lm, frame.shape, None)
print("roi", a.roi, "work", a.work_size, "scale", a.scale)

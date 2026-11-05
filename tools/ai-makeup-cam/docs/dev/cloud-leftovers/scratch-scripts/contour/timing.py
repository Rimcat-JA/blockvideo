import time
from harness import *
from makeup_cam.config import ContourSettings, TearbagSettings
for src, idx in [("frame_near", None), ("1029_DFA_NEU_XX", 20)]:
    frames = [still(src)] * 4 if idx is None else load_clip(src)[: idx + 1]
    for f, ctx in contexts(frames):
        pass
    print(src, "scale", regions.face_scale(ctx.landmarks))
    img = f.astype(np.float32) / 255
    for name, eff, st in [("contour", C.ContourEffect(), ContourSettings()), ("tearbag", C.TearbagEffect(), TearbagSettings())]:
        ts = []
        for _ in range(60):
            x = img.copy()
            t = time.perf_counter(); eff.apply(x, ctx, st); ts.append((time.perf_counter() - t) * 1000)
        print(f"  {name}: median {np.median(ts):.2f} ms  p90 {np.percentile(ts, 90):.2f}")
    # 2x closer face (bigger scale) synthetic: scale landmarks around center
    lm = ctx.landmarks; c = lm.mean(0)
    big = (lm - c) * 1.6 + np.array([640, 360], np.float32)
    ctx2 = FrameContext(720, 1280, big, None, None, None, None)
    ts = []
    for _ in range(40):
        x = img.copy(); t = time.perf_counter(); C.ContourEffect().apply(x, ctx2, ContourSettings()); ts.append((time.perf_counter() - t) * 1000)
    print(f"  contour x1.6 face (scale {regions.face_scale(big):.0f}): median {np.median(ts):.2f} ms")

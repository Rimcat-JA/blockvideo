import cProfile, pstats
from harness import *
from makeup_cam.config import ContourSettings, TearbagSettings
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
img = f.astype(np.float32) / 255
eff = C.ContourEffect(); st = ContourSettings()
tb = C.TearbagEffect(); tst = TearbagSettings()
for _ in range(5): eff.apply(img.copy(), ctx, st)
pr = cProfile.Profile(); pr.enable()
for _ in range(50): eff.apply(img.copy(), ctx, st); tb.apply(img.copy(), ctx, tst)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(18)

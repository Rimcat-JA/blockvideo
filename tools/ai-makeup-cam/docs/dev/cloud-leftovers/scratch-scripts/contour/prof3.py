import cProfile, pstats, sys
from harness import *
from makeup_cam.config import ContourSettings, TearbagSettings
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
img = f.astype(np.float32) / 255
which = sys.argv[1]
eff = C.ContourEffect() if which == "contour" else C.TearbagEffect()
st = ContourSettings() if which == "contour" else TearbagSettings()
xs = [img.copy() for _ in range(200)]
pr = cProfile.Profile(); pr.enable()
for x in xs: eff.apply(x, ctx, st)
pr.disable()
pstats.Stats(pr).sort_stats(sys.argv[2] if len(sys.argv) > 2 else "tottime").print_stats(26)

import cProfile, pstats, time
from common import *
from makeup_cam.config import TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import teeth as T
frames, lms = load("1060_IEO_HAP_HI")
imgs = [f.astype(np.float32) / 255 for f in frames]
eff = T.TeethEffect()
s = TeethSettings()
def run():
    for img, lm in zip(imgs, lms):
        eff.apply(img.copy(), FrameContext(height=720, width=1280, landmarks=lm), s)
run()
ts = []
for img, lm in zip(imgs, lms):
    x = img.copy(); c = FrameContext(height=720, width=1280, landmarks=lm)
    t0 = time.perf_counter(); eff.apply(x, c, s); ts.append((time.perf_counter() - t0) * 1000)
print("scale", regions.face_scale(lms[20]), "ms med", np.median(ts), "p90", np.percentile(ts, 90), "max", max(ts))
cProfile.run("run()", "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_teeth/prof.out")
st = pstats.Stats("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_teeth/prof.out"); st.sort_stats("tottime").print_stats(18)

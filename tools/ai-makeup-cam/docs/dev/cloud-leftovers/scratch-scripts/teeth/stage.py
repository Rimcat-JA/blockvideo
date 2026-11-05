import time
from common import *
from makeup_cam import blend
from makeup_cam.config import TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import teeth as T
frames, lms = load("1060_IEO_HAP_HI")
i = 30
img = frames[i].astype(np.float32) / 255; lm = lms[i]
eff = T.TeethEffect(); s = TeethSettings()
for _ in range(5):
    eff.apply(img.copy(), FrameContext(height=720, width=1280, landmarks=lm), s)
scale = regions.face_scale(lm)
fp = scale * T._feather(s)
roi = regions.roi_around(lm[regions.LIPS_INNER], img.shape, 0.0, 3 * fp + T._gap_kernel_size(scale) / 2 + 2)
print("roi", roi.w, roi.h, "scale", scale)
view = img[roi.slices].copy()
N = 200
def t(name, f):
    t0 = time.perf_counter()
    for _ in range(N): r = f()
    print(f"{name:28s} {(time.perf_counter()-t0)/N*1000:.3f} ms")
    return r
lab = t("bgr_to_lab", lambda: blend.bgr_to_lab(view))
chroma = t("chroma blur", lambda: cv2.GaussianBlur(lab[..., 1:], (0, 0), max(T._CHROMA_SIGMA * scale, 0.5), borderType=cv2.BORDER_REPLICATE))
inner = regions.expand_polygon(lm[regions.LIPS_INNER], -T._STATS_INSET * scale)
interior = t("interior mask", lambda: regions.polygon_mask(inner, roi) > 0.5)
t("teeth_reference", lambda: T._teeth_reference(lab[..., 0][interior], chroma[..., 0][interior]))
t("skin_reference", lambda: T._skin_reference(img, lm, scale))
area = t("area mask", lambda: regions.polygon_mask(lm[regions.LIPS_INNER], roi, fp))
st = eff.stats
lv = T._Levels.from_settings(s, 1.0)
t("make_up", lambda: T._make_up(lab, chroma, area, st, lv, scale))
lv0 = T._Levels(0.3, 0.4, 0.0)
t("make_up no even", lambda: T._make_up(lab, chroma, area, st, lv0, scale))
made = T._make_up(lab, chroma, area, st, lv, scale)
t("lab_to_bgr x2", lambda: (blend.lab_to_bgr(made), blend.lab_to_bgr(lab)))
t("full apply", lambda: eff.apply(img.copy(), FrameContext(height=720, width=1280, landmarks=lm), s))
t("frame copy", lambda: img.copy())
x = img.copy(); c = FrameContext(height=720, width=1280, landmarks=lm)
t("apply in place (no copy)", lambda: eff.apply(x, c, s))
import cProfile, pstats
cProfile.run("for _ in range(300): eff.apply(x, c, s)", "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_teeth/p2.out")
pstats.Stats("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/tmp_teeth/p2.out").sort_stats("tottime").print_stats(14)

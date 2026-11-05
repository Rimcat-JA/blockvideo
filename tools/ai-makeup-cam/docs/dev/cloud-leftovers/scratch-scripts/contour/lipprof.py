import time
from harness import *
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
lm = ctx.landmarks; scale = regions.face_scale(lm)
roi = regions.roi_around(lm[regions.FACE_OVAL], f.shape, pad=0.0, margin_px=C._ROI_MARGIN * scale)
G = np.zeros((roi.h, roi.w, 3), np.float32)
def T(fn, n=200):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 3)
feather_px = C._LIP_FEATHER * scale
outline = regions.expand_polygon(lm[regions.LIPS_OUTER], C._LIP_GUARD * scale)
area = regions.roi_around(outline, (roi.y1, roi.x1), pad=0.0, margin_px=3.0 * feather_px + 2.0)
print("area", area.w, area.h)
print("expand", T(lambda: regions.expand_polygon(lm[regions.LIPS_OUTER], C._LIP_GUARD * scale)))
print("solid", T(lambda: C._solid_shapes((area.w, area.h), [outline - np.asarray(area.offset, np.float32)], feather_px)))
keep = (1 - C._solid_shapes((area.w, area.h), [outline - np.asarray(area.offset, np.float32)], feather_px))[..., None]
local = (slice(area.y0 - roi.y0, area.y1 - roi.y0), slice(area.x0 - roi.x0, area.x1 - roi.x0))
def mul():
    G[local] *= keep
print("mul", T(mul))
print("whole", T(lambda: C._protect_lips(lm, roi, scale, G, G)))
c = np.zeros((area.h, area.w), np.uint8)
print("blur", T(lambda: C._blur(C._to_unit(c), feather_px)))
print("fill", T(lambda: cv2.fillPoly(c, [C._fixed(outline)], 255, cv2.LINE_AA, 4)))

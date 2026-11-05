import time
from harness import *
from makeup_cam.config import ContourSettings, TearbagSettings, hex_to_bgr
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
img = f.astype(np.float32) / 255
st = ContourSettings()
lm = ctx.landmarks
def T(fn, n=60):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 3), r
scale = regions.face_scale(lm)
roi = regions.roi_around(lm[regions.FACE_OVAL], img.shape, pad=0.0, margin_px=C._ROI_MARGIN * scale)
grid = C._Grid.over(roi, scale)
print("roi", roi.w, roi.h, "grid", grid.size)
amounts = {"shade": st.shading, "light": st.highlight}
active = list(C._STROKE_SIDES)
t, w = T(lambda: C._contour_weights(grid, lm, scale, active, st, amounts, ctx)); print("weights", t)
unit = scale * float(grid.ratio.mean()); pts = grid.points(lm)
t, oval = T(lambda: regions.face_oval_mask(pts, grid.area, expand_px=-C._OVAL_INSET * unit, feather_px=C._OVAL_FEATHER * unit)); print("  oval", t)
t, sk = T(lambda: C._segmentation_weight(grid, ctx, oval)); print("  seg", t)
t, _ = T(lambda: C._feature_guard(grid, pts, unit)); print("  guard", t)
def draws():
    canvas = np.zeros((grid.height, grid.width), np.uint8)
    for item in active:
        p = item.points(pts)
        C._draw_path(canvas, C._Path(p, item.radii * np.float32(unit), item.levels))
t, _ = T(draws); print("  draws", t)
t, _ = T(lambda: [item.points(pts) for item in active]); print("  points", t)
view = img[roi.slices].copy()
t, small = T(lambda: grid.downsample(view)); print("downsample", t)
filt = C._shading_filter(hex_to_bgr(st.shading_color)); li = C._Light.of(hex_to_bgr(st.highlight_color))
t, (g, a) = T(lambda: C._tone_factors(small, *w, filt, li)); print("factors", t)
t, (G, A) = T(lambda: (grid.upsample(g), grid.upsample(a))); print("upsample", t)
t, _ = T(lambda: C._protect_lips(lm, roi, scale, G.copy(), A.copy())); print("lips", t)
t, _ = T(lambda: C._apply_factors(view.copy(), G, A)); print("apply(+copy)", t)
t, _ = T(lambda: view.copy()); print("  copy", t)

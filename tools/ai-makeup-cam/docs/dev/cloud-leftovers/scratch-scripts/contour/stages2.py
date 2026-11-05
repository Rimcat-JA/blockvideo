import time
from harness import *
from makeup_cam.config import ContourSettings, hex_to_bgr
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
img = f.astype(np.float32) / 255
st = ContourSettings()
acc = {}
def tick(name, t0):
    t = time.perf_counter(); acc.setdefault(name, []).append((t - t0) * 1000); return t
for _ in range(300):
    frame = img.copy()
    t = time.perf_counter()
    amounts = {"shade": st.shading, "light": st.highlight}
    active = [s for s in C._STROKE_SIDES if amounts[s.stroke.tone] * getattr(st, s.stroke.area) > 0.0]
    lm = ctx.landmarks; scale = regions.face_scale(lm)
    roi = regions.roi_around(lm[regions.FACE_OVAL], frame.shape, pad=0.0, margin_px=C._ROI_MARGIN * scale)
    grid = C._Grid.over(roi, scale)
    t = tick("setup", t)
    shade, light = C._contour_weights(grid, lm, scale, active, st, amounts, ctx)
    t = tick("weights", t)
    view = frame[roi.slices]
    small = grid.downsample(view)
    t = tick("downsample", t)
    gain = C._tone_gain(small, shade, light, C._shading_filter(hex_to_bgr(st.shading_color)), C._Light.of(hex_to_bgr(st.highlight_color)))
    t = tick("tone_gain", t)
    gain = grid.upsample(gain)
    t = tick("upsample", t)
    C._protect_lips(lm, roi, scale, gain)
    t = tick("lips", t)
    C._apply_gain(view, gain)
    t = tick("apply", t)
tot = 0
for k, v in acc.items():
    m = np.percentile(v, 25); tot += m
    print(f"{k:12s} p25 {m:.3f}  median {np.median(v):.3f}")
print("total p25", round(tot, 3))

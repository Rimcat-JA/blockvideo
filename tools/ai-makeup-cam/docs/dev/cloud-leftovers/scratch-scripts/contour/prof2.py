import cProfile, pstats
from harness import *
from makeup_cam.config import ContourSettings
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
lm = ctx.landmarks; st = ContourSettings()
scale = regions.face_scale(lm)
roi = regions.roi_around(lm[regions.FACE_OVAL], f.shape, pad=0.0, margin_px=C._ROI_MARGIN * scale)
grid = C._Grid.over(roi, scale)
amounts = {"shade": st.shading, "light": st.highlight}
strokes = list(C._STROKES)
pr = cProfile.Profile(); pr.enable()
for _ in range(200): C._contour_weights(grid, lm, scale, list(C._STROKE_SIDES), st, amounts, ctx)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(15)

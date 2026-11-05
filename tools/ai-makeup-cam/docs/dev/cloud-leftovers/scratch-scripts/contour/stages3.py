import time
from harness import *
from makeup_cam.config import ContourSettings
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
st = ContourSettings(); lm = ctx.landmarks; scale = regions.face_scale(lm)
roi = regions.roi_around(lm[regions.FACE_OVAL], f.shape, pad=0.0, margin_px=C._ROI_MARGIN * scale)
grid = C._Grid.over(roi, scale)
def T(name, fn, n=500):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    print(f"{name:14s} {np.percentile(ts, 25) * 1e3:.3f} ms")
    return r
unit = scale * float(grid.ratio.mean()); points = grid.points(lm)
T("points", lambda: grid.points(lm))
inset = T("expand", lambda: regions.expand_polygon(points[regions.FACE_OVAL], -C._OVAL_INSET * unit))
oval = T("oval", lambda: C._blur(C._filled(grid.size, [inset]), C._OVAL_FEATHER * unit))
skin = T("seg", lambda: C._segmentation_weight(grid, ctx, oval))
T("skinblur", lambda: C._blur(skin, C._MASK_FEATHER * unit))
T("squeeze", lambda: {side: C._cheek_squeeze(lm, side, scale) for side in regions.SIDES})
T("item.points", lambda: [it.points(points) for it in C._STROKE_SIDES])
pp = [it.points(points) for it in C._STROKE_SIDES]
T("sample", lambda: C._sample(skin, pp[6]))
groups = {}
for it, p in zip(C._STROKE_SIDES, pp):
    groups.setdefault((it.stroke.tone, it.stroke.blur), []).append(C._Path(p, it.radii * np.float32(unit), it.levels))
def draw():
    for g in groups.values():
        c = np.zeros((grid.height, grid.width), np.uint8); C._draw_paths(c, g)
T("draw", draw)
T("guard", lambda: C._feature_guard(grid, points, unit))
print({k: sum(len(p.points) - 1 for p in g) for k, g in groups.items()})

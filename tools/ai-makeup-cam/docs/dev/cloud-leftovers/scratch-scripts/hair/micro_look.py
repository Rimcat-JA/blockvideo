import sys, time
import numpy as np
import bench
from makeup_cam import blend
from makeup_cam.effects import hair as H

frame, lm, hair, face = bench.load(sys.argv[1] if len(sys.argv) > 1 else "frame_720p")
img = frame.astype(np.float32) / 255
area = H._hair_area(H._hair_blobs(hair), lm, img.shape, None)
view = img[area.roi.slices]
luma = blend.luminance(view)
wl = area.to_work(luma)
fw = area.to_work(face[area.roi.slices])
weights = H._hair_weights(area.to_work(hair[area.roi.slices]), fw.copy(), area, 0.5)
lock, hl = H._hair_luminance(wl, weights.hair, area.work_scale)
stats = H._measure(area.sample(view), wl, lock, weights.interior, fw > 0.6)
rl = area.to_render(luma)
target = H._scene_color((0.13, 0.16, 0.23), stats.cast)

def t(label, f, n=300):
    f(); st = time.perf_counter()
    for _ in range(n): f()
    print(f"{label:34s} {(time.perf_counter() - st) / n * 1000:.3f} ms")

for name, kw in {"smooth": dict(smoothness=1), "gloss": dict(gloss=1), "dye": dict(color_strength=1), "tone": dict(brightness=0.3),
                 "default": dict(gloss=0.3, smoothness=0.3)}.items():
    from makeup_cam.config import HairSettings
    s = HairSettings(**{**dict(gloss=0, smoothness=0, color_strength=0, brightness=0, contrast=0), **kw})
    lv = H._Levels.from_settings(s, 1.0)
    t("look " + name, lambda: H._look(rl, hl, lock, weights, area, stats, lv, target))
t("strand_detail", lambda: H._strand_detail(rl, hl, area))
t("gloss_zone", lambda: H._gloss_zone(lock, weights, area, stats))
t("comb", lambda: H._comb(hl, area.work_scale))
d, b = H._strand_detail(rl, hl, area)
z = np.ones_like(rl) * 0.3
t("gloss()", lambda: H._gloss(d / (b + 0.03), z, 0.8))
g = np.ones_like(rl)
t("channel_maps sheen only", lambda: H._channel_maps(g, g * 0, np.ones(3), None, None, None, z, np.ones(3)))
t("channel_maps all", lambda: H._channel_maps(g, g * 0.5, np.ones(3) * 0.8, z, np.ones(3), z, z, np.ones(3)))

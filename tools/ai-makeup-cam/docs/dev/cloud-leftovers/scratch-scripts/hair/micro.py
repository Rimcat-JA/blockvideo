"""Isolated timings of the hair effect's steps on one cached frame."""
import sys, time
import numpy as np
import bench
from makeup_cam import blend, regions
from makeup_cam.config import HairSettings
from makeup_cam.effects import hair as H

name = sys.argv[1] if len(sys.argv) > 1 else "frame_720p"
frame, lm, hair, face = bench.load(name)
img = frame.astype(np.float32) / 255
s = HairSettings(gloss=1.0, smoothness=1.0, color_strength=0.8, brightness=0.2, contrast=0.2)
levels = H._Levels.from_settings(s, 1.0)

def t(label, f, n=200):
    f(); st = time.perf_counter()
    for _ in range(n): f()
    print(f"{label:28s} {(time.perf_counter() - st) / n * 1000:.3f} ms")

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
sc, off = H._look(rl, hl, lock, weights, area, stats, levels, target)
t("hair_area", lambda: H._hair_area(H._hair_blobs(hair), lm, img.shape, None))
t("luminance full", lambda: blend.luminance(view))
t("to_work luma", lambda: area.to_work(luma))
t("to_work masks x2", lambda: (area.to_work(hair[area.roi.slices]), area.to_work(face[area.roi.slices])))
t("hair_weights", lambda: H._hair_weights(area.to_work(hair[area.roi.slices]), fw, area, 0.5))
t("  inner_face", lambda: H._inner_face(area))
t("hair_luminance", lambda: H._hair_luminance(wl, weights.hair, area.work_scale))
t("measure", lambda: H._measure(area.sample(view), wl, lock, weights.interior, fw > 0.6))
t("to_render luma", lambda: area.to_render(luma))
t("look", lambda: H._look(rl, hl, lock, weights, area, stats, levels, target))
t("  comb", lambda: H._comb(hl, area.work_scale))
t("  gloss_zone", lambda: H._gloss_zone(lock, weights, area, stats))
t("  angel_ring", lambda: H._angel_ring(weights.hair, area))
t("final composite", lambda: (view * area.render_to_full(sc) + area.render_to_full(off)).clip(0, 1))
eff = H.HairEffect()
from makeup_cam.effects.base import FrameContext
ctx = FrameContext(720, 1280, landmarks=lm, hair_mask=hair, face_skin_mask=face)
t("apply (all)", lambda: eff.apply(img.copy(), ctx, s), 50)
t("img.copy", lambda: img.copy(), 50)
print(area.roi, area.work_size, area.render_size)

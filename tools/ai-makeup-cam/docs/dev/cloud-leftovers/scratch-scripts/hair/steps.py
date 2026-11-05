"""Average time of each step of HairEffect.apply (replicated with timestamps)."""
import collections, sys, time
import numpy as np
import bench
from makeup_cam import blend, regions
from makeup_cam.config import HairSettings, hex_to_bgr
from makeup_cam.effects import hair as H

frame, lm, hair, face = bench.load(sys.argv[1] if len(sys.argv) > 1 else "frame_720p")
base = frame.astype(np.float32) / 255
settings = HairSettings() if "--all" not in sys.argv else HairSettings(gloss=1, smoothness=1, color_strength=0.8, brightness=0.2, contrast=0.2)
acc = collections.defaultdict(float)
N = 200
for _ in range(N):
    img = base.copy()
    t = [time.perf_counter()]
    def mark(name):
        now = time.perf_counter(); acc[name] += now - t[0]; t[0] = now
    levels = H._Levels.from_settings(settings, 1.0)
    area = H._hair_area(H._hair_blobs(hair), lm, img.shape, None); mark("area")
    view = img[area.roi.slices]
    luma = blend.luminance(view); work_luma = area.to_work(luma); mark("luma+to_work")
    fw = area.to_work(face[area.roi.slices]); hw = area.to_work(hair[area.roi.slices]); mark("masks to_work")
    weights = H._hair_weights(hw, fw, area, settings.edge_softness); mark("weights")
    lock, hl = H._hair_luminance(work_luma, weights.hair, area.work_scale); mark("hair_luminance")
    stats = H._measure(area.sample(view), work_luma, lock, weights.interior, fw > 0.6); mark("measure")
    target = H._scene_color(hex_to_bgr(settings.color), stats.cast)
    rl = area.to_render(luma); mark("to_render")
    sm, om = H._look(rl, hl, lock, weights, area, stats, levels, target); mark("look")
    view *= area.render_to_full(sm)
    if om is not None:
        view += area.render_to_full(om)
    np.clip(view, 0, 1, out=view); mark("composite")
total = sum(acc.values())
for k, v in acc.items():
    print(f"{k:16s} {v / N * 1000:.3f} ms")
print(f"{'total':16s} {total / N * 1000:.3f} ms")

"""Full-pipeline timing on frame_720p: natural preset, background segmentation, 30 fps pacing."""
import os, statistics, sys, time
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
import cv2, numpy as np
from makeup_cam.models import ensure_model, SELFIE_MULTICLASS
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.presets import load_preset
from makeup_cam.segmentation import Segmenter

S = "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad"
frame = cv2.imread(f"{S}/testimgs/frame_720p.png")
rng = np.random.default_rng(0)
frames = [cv2.add(frame, rng.integers(0, 3, frame.shape, dtype=np.uint8)) for _ in range(8)]

def stats(name, vals):
    vals = sorted(vals)
    print(f"  {name:13s} median {statistics.median(vals):6.1f} ms  p90 {vals[int(len(vals) * 0.9)]:6.1f} ms")

for preset in sys.argv[1:]:
    pipeline = MakeupPipeline(load_preset(preset))
    timings = []
    t0 = time.monotonic()
    for i in range(200):
        start = time.perf_counter()
        r = pipeline.process(frames[i % len(frames)], int(i * 1000 / 30))
        timings.append(r.timings_ms)
        time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - start)))
    pipeline.close()
    timings = timings[30:]
    print(f"preset {preset}: {len(timings)} frames, face found in all: {all('lips' in t or 'skin' in t for t in timings)}")
    keys = ["landmarks", "segmentation", "hair", "skin", "brows", "eyeshadow", "eyeliner", "blush", "lips"]
    for k in keys:
        vals = [t.get(k) for t in timings if k in t]
        if vals: stats(k, vals)
    stats("effects sum", [sum(t.get(k, 0) for k in keys[2:]) for t in timings])
    stats("total", [t["total"] for t in timings])

seg = Segmenter(ensure_model(SELFIE_MULTICLASS), interval=1, background=False)
vals = []
for i in range(20):
    s = time.perf_counter(); seg.update(frame, i * 33, i); vals.append((time.perf_counter() - s) * 1000)
seg.close()
print("inline segmentation model run (384 long side + EMA + upsample):")
stats("segmentation", vals[3:])

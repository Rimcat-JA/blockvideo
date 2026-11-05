import sys, time, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.tracking import FaceTracker
from makeup_cam.segmentation import Segmenter
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.config import MakeupSettings
from makeup_cam.effects.base import Effect
M = sys.argv[1]; frame = cv2.imread(sys.argv[2])
rng = np.random.default_rng(0)
frames = [np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8) for _ in range(8)]
def med(ts): return f"median {np.median(ts):.1f} ms, p90 {np.percentile(ts, 90):.1f} ms"
tr = FaceTracker(M + "/face_landmarker.task", 0.5); ts = []
for i in range(80):
    t = time.perf_counter(); lm, _ = tr.detect(frames[i % 8], 33 * i); ts.append((time.perf_counter() - t) * 1000)
    assert lm is not None
tr.close(); print("tracker.detect 1280x720:", med(ts[10:]))
seg = Segmenter(M + "/selfie_multiclass_256x256.tflite", interval=1, background=False); ts = []
for i in range(30):
    t = time.perf_counter(); seg.update(frames[i % 8], 33 * i, i); ts.append((time.perf_counter() - t) * 1000)
seg.close(); print("segmenter.update sync (model run + EMA + upsample):", med(ts[5:]))
class Lips(Effect):
    name = "lips"
    def apply(self, frame, ctx, s): return frame
class Hair(Effect):
    name = "hair"; order = 10; needs_face = False; needs_hair_mask = True
    def apply(self, frame, ctx, s): return frame
for interval in (2, 1):
    s = MakeupSettings(); s.general.segmentation_interval = interval
    p = MakeupPipeline(s, model_dir=__import__("pathlib").Path(M), effects=[Hair(), Lips()])
    rows = []; masks_ready = None
    for i in range(90):
        t0 = time.perf_counter()
        r = p.process(frames[i % 8], 33 * i)
        rows.append(dict(r.timings_ms))
        if masks_ready is None and "hair" in r.timings_ms: masks_ready = i
        # emulate a 30 fps camera: wait for the next frame slot
        time.sleep(max(0.0, 0.033 - (time.perf_counter() - t0)))
    p.close()
    rows = rows[15:]
    print(f"pipeline (no-op effects, async seg interval={interval}): total {med([r['total'] for r in rows])}; landmarks {med([r['landmarks'] for r in rows])}; segmentation(caller) {med([r['segmentation'] for r in rows])}; first masks at frame {masks_ready}")

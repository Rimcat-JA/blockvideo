import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import tracking
M = sys.argv[1] + "/face_landmarker.task"
from skimage import data
face = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (720, 720), interpolation=cv2.INTER_CUBIC)
rng = np.random.default_rng(1)
def frame_at(x, dark=False):
    c = np.full((720, 1280, 3), (70, 75, 80), np.uint8)
    x = int(round(x)); xs, xe = max(x, 0), min(x + 720, 1280)
    c[:, xs:xe] = face[:, xs - x: xe - x]
    if dark: c = np.clip(((c/255.0)**1.6)*255*np.array([0.85,0.95,1.05]),0,255)
    return np.clip(c + rng.normal(0, 2.5, c.shape), 0, 255).astype(np.uint8)
# reference landmarks at x=280
ref_tr = tracking.FaceTracker(M, smoothing=0)
for i in range(3): ref, _ = ref_tr.detect(frame_at(280), i * 33)
ref_tr.close()
class FullFrame:
    """Baseline: whole frame downscaled to 640 (contract's original suggestion)."""
    def __init__(self): self.m = tracking.MediaPipeFaceLandmarker(M); self.ts = 0
    def detect(self, f, ts):
        small = cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        r = self.m.detect(small, ts)
        return (None, None) if r is None else (r[:, :2] * (1280, 720), None)
    def close(self): self.m.close()
for period, amp, dark in ((3.0, 250, False), (1.5, 250, False), (3.0, 250, True)):
    xs = 280 + amp * np.sin(2 * np.pi * np.arange(150) / 30 / period)
    frames = [frame_at(x, dark) for x in xs]
    for name, mk in (("full640", FullFrame), ("window s=0", lambda: tracking.FaceTracker(M, 0)), ("window s=.5", lambda: tracking.FaceTracker(M, .5)), ("window s=1", lambda: tracking.FaceTracker(M, 1.0))):
        tr = mk(); errs = []; found = 0; jit = []
        prev = None
        for i, (f, x) in enumerate(zip(frames, xs)):
            lm, _ = tr.detect(f, i * 33)
            if lm is None: prev = None; continue
            found += 1
            gt = ref + (x - 280, 0)
            errs.append(np.abs(lm - gt).mean())
        tr.close()
        e = np.array(errs) if errs else np.array([np.nan])
        print(f"period={period} dark={dark} {name:12s} found={found}/150 err_mean={e.mean():.2f}px err_p95={np.percentile(e,95):.2f}px")

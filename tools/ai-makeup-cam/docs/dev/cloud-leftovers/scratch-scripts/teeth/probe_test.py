import sys, time
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import numpy as np, cv2
from conftest import build_face_frame_720p, FACE_LANDMARKS_720P
from makeup_cam import regions, blend
from makeup_cam.config import TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import teeth as T
frame = build_face_frame_720p().astype(np.float32) / 255
lm = np.load(FACE_LANDMARKS_720P).astype(np.float32)
full = regions.Roi(0, 0, 1280, 720)
inner = regions.polygon_mask(regions.expand_polygon(lm[regions.LIPS_INNER], -1.0), full) > 0.99
print("scale", regions.face_scale(lm), "opening", T._opening(lm) / regions.face_scale(lm), "inner px", inner.sum())
for name, img in [("orig", frame), ("dim_yellow", np.clip(frame * np.float32([0.62, 0.8, 0.86]), 0, 1))]:
    lab = blend.bgr_to_lab(img)
    teeth = inner & (lab[..., 0] > np.percentile(lab[..., 0][inner], 40))
    for vals in [dict(whitening=1, yellow_reduction=0, evenness=0), dict(whitening=0, yellow_reduction=1, evenness=0), dict(whitening=0, yellow_reduction=0, evenness=1), dict()]:
        e = T.TeethEffect()
        out = e.apply(img.copy(), FrameContext(height=720, width=1280, landmarks=lm), TeethSettings(**vals))
        lo = blend.bgr_to_lab(out)
        st = e.stats
        print(name, vals, "stats L%.1f a%.1f b%.1f pres %.2f" % (st.teeth_l, st.teeth_a, st.teeth_b, st.presence()), "dL %.2f db %.2f" % ((lo[..., 0] - lab[..., 0])[teeth].mean(), (lo[..., 2] - lab[..., 2])[teeth].mean()), "maxL", lo[..., 0][teeth].max().round(1), "neutral", np.round(st.neutral(), 1), "minb", lo[..., 2][teeth].min().round(1))
    t0 = time.perf_counter()
    for _ in range(50):
        e.apply(img.copy(), FrameContext(height=720, width=1280, landmarks=lm), TeethSettings())
    print("ms incl copy", (time.perf_counter() - t0) / 50 * 1000)

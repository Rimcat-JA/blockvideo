"""Dump the hair weight, interior, band and gloss zone (working copy) for one frame."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(Path(__file__).parent))
from makeup_cam import blend, regions
from makeup_cam.effects import hair as H
from makeup_cam.effects.base import FrameContext
from makeup_cam.segmentation import Segmenter
from makeup_cam.tracking import FaceTracker
from render import load, S

name = sys.argv[1]
frame = load(name)
tracker = FaceTracker(S / "models" / "face_landmarker.task", 0.0)
seg = Segmenter(S / "models" / "selfie_multiclass_256x256.tflite", interval=1, background=False)
for i in range(8):
    lm, _ = tracker.detect(frame, 33 * (i + 1))
    masks = seg.update(frame, 33 * (i + 1), i + 1)
tracker.close(); seg.close()
ctx = FrameContext(*frame.shape[:2], landmarks=lm, hair_mask=masks.hair, face_skin_mask=masks.face_skin)
area = H._hair_area(H._hair_blobs(ctx.hair_mask), lm, frame.shape, None)
view = frame[area.roi.slices].astype(np.float32) / 255
luma = blend.luminance(view)
wl = area.to_work(luma)
lock, _ = H._hair_luminance(wl, H._hair_weights(ctx, area, 0.5).hair, area.work_scale)
w = H._hair_weights(ctx, area, 0.5)
st = H._measure(area.sample(view), wl, lock, w.interior, H._forehead_skin(ctx, area))
print("stats", st)
band = H._angel_ring(w.hair, area)
zone = H._gloss_zone(lock, w, area, st)
hl = zone
tiles = [w.hair, w.interior, band, hl, zone / max(zone.max(), 1e-6)]
big = [cv2.resize((t * 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) for t in tiles]
small_img = cv2.resize(frame[area.roi.slices], big[0].shape[::-1])
cv2.imwrite(str(S / "renders" / "hair" / f"zone_{name}.png"), np.hstack([cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)] + big))
roi = area.roi
print("hair max in roi", float(masks.hair[roi.slices].max()), "mean", float(masks.hair[roi.slices].mean()))
print("face max", float(masks.face_skin[roi.slices].max()))
over = frame.copy().astype(np.float32)
over[..., 2] = over[..., 2] * (1 - masks.hair) + 255 * masks.hair
over[..., 1] = over[..., 1] * (1 - masks.face_skin) + 255 * masks.face_skin
cv2.imwrite(str(S / "renders" / "hair" / f"mask_{name}.png"), over[roi.slices].astype(np.uint8))

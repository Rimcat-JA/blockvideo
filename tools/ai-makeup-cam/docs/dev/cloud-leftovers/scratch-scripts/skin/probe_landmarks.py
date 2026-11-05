"""Detect landmarks on the test images once and cache them (for fast iteration)."""
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.tracking import FaceTracker
from makeup_cam import regions
S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
out = S / "tmp_skin"
for name in ["frame_720p", "frame_dark", "frame_tilt", "frame_far"]:
    img = cv2.imread(str(S / "testimgs" / f"{name}.png"))
    tr = FaceTracker(S / "models" / "face_landmarker.task", smoothing=0.0)
    lm = None
    for i in range(3):
        lm, z = tr.detect(img, 33 * (i + 1))
    tr.close()
    np.save(out / f"lm_{name}.npy", lm)
    print(name, img.shape, regions.face_scale(lm), regions.roi_around(lm[regions.FACE_OVAL], img.shape, 0.0))

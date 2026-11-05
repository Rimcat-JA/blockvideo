"""Cache landmarks of the test images and print cheek geometry measurements."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.tracking import FaceTracker

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = S / "tmp_lips_blush"
for name in ["frame_720p", "frame_dark", "frame_tilt", "frame_far"]:
    img = cv2.imread(str(S / "testimgs" / f"{name}.png"))
    tracker = FaceTracker(S / "models" / "face_landmarker.task", smoothing=0.0)
    lm = None
    for i in range(3):
        lm, _ = tracker.detect(img, 33 * (i + 1))
    tracker.close()
    np.save(OUT / f"lm_{name}.npy", lm)
    scale = regions.face_scale(lm)
    across = regions.eye_center(lm, "left") - regions.eye_center(lm, "right")
    across /= np.linalg.norm(across)
    for side, edge_ids, sign in (("right", [234, 93], -1), ("left", [454, 323], 1)):
        c = regions.cheek_center(lm, side)
        edge = lm[edge_ids].mean(axis=0)
        width = float(np.dot(edge - c, sign * across)) / scale
        nose = lm[regions.NOSE_TIP]
        inner = float(np.dot(c - nose, sign * across)) / scale
        mc = lm[regions.MOUTH_RIGHT_CORNER if side == "right" else regions.MOUTH_LEFT_CORNER]
        print(name, side, f"scale={scale:.1f} cheek->edge={width:.3f} nose->cheek={inner:.3f} cheek={c.round(1)} mouthcorner={mc.round(1)}")
    lips_w = np.linalg.norm(lm[291] - lm[61])
    lower_t = np.linalg.norm(lm[17] - lm[14])
    upper_t = np.linalg.norm(lm[0] - lm[13])
    print(name, f"lips width={lips_w/scale:.3f} lower={lower_t/scale:.3f} upper={upper_t/scale:.3f}")

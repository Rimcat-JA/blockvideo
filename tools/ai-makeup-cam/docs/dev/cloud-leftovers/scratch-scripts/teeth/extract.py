"""Decode CREMA-D clips -> 1280x720 webcam-like frames + landmarks (cached npz)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
from makeup_cam.tracking import FaceTracker  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = SCRATCH / "tmp_teeth" / "clips"
OUT.mkdir(parents=True, exist_ok=True)


def webcam(frame):
    big = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((720, 1280, 3), np.uint8)
    canvas[:] = big[:, :1].mean(axis=(0, 1))
    canvas[:, 160:1120] = big
    return canvas


def extract(name):
    target = OUT / f"{name}.npz"
    if target.exists():
        return
    cap = cv2.VideoCapture(str(SCRATCH / "testvids" / f"{name}.flv"))
    frames, lms = [], []
    tracker = FaceTracker(SCRATCH / "models" / "face_landmarker.task", smoothing=0.5)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w = webcam(f)
        lm, _ = tracker.detect(w, int(i * 1000 / 30) + 1)
        frames.append(w)
        lms.append(np.full((478, 2), np.nan, np.float32) if lm is None else lm)
        i += 1
    tracker.close()
    np.savez_compressed(target, frames=np.stack(frames), landmarks=np.stack(lms))
    print(name, len(frames), sum(np.isfinite(l).all() for l in lms))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        extract(n)

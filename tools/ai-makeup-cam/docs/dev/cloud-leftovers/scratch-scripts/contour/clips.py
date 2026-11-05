"""Shared helpers for contour tuning: load CREMA-D clips as webcam-like 1280x720 frames."""
import sys
from pathlib import Path

import cv2
import numpy as np

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
ROOT = "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam"
sys.path.insert(0, ROOT)
VIDS = S / "testvids"
TMP = S / "tmp_contour"
OUT = S / "renders" / "contour"
ACTORS = ["1002", "1012", "1013", "1029", "1046", "1060", "1072", "1073", "1074", "1076", "1082"]


def load_clip(name: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(VIDS / f"{name}.flv"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        big = cv2.resize(f, (960, 720), interpolation=cv2.INTER_CUBIC)
        frames.append(cv2.copyMakeBorder(big, 0, 0, 160, 160, cv2.BORDER_REPLICATE))
    cap.release()
    return frames


def cache_landmarks(name: str) -> np.ndarray:
    path = TMP / f"lm_{name}.npy"
    if path.exists():
        return np.load(path)
    from makeup_cam.tracking import FaceTracker
    frames = load_clip(name)
    tracker = FaceTracker(S / "models" / "face_landmarker.task", smoothing=0.5)
    out = np.full((len(frames), 478, 2), np.nan, np.float32)
    for i, f in enumerate(frames):
        lm, _ = tracker.detect(f, 33 * (i + 1))
        if lm is not None:
            out[i] = lm
    tracker.close()
    np.save(path, out)
    return out


if __name__ == "__main__":
    for a in ACTORS:
        for c in ("IEO_HAP_HI", "DFA_NEU_XX", "ITS_HAP_XX"):
            lm = cache_landmarks(f"{a}_{c}")
            print(a, c, lm.shape, int(np.isnan(lm[:, 0, 0]).sum()), "missing", flush=True)

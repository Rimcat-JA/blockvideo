import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
from makeup_cam import regions  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
CLIPS = SCRATCH / "tmp_teeth" / "clips"
RENDERS = SCRATCH / "renders" / "teeth"
ACTORS = ["1002", "1012", "1013", "1029", "1046", "1060", "1072", "1073", "1074", "1076", "1082"]


def load(name):
    d = np.load(CLIPS / f"{name}.npz")
    return d["frames"], d["landmarks"]


def opening(lm):
    return float(np.linalg.norm(lm[13] - lm[14]))


def mouth_box(lm, size=(240, 140)):
    pts = lm[regions.LIPS_OUTER]
    c = pts.mean(axis=0)
    w = (pts[:, 0].max() - pts[:, 0].min()) * 1.25
    h = w * size[1] / size[0]
    x0, y0 = int(c[0] - w / 2), int(c[1] - h / 2)
    return x0, y0, int(w), int(h)


def crop(img, box, size=(240, 140)):
    x0, y0, w, h = box
    sub = img[max(y0, 0):y0 + h, max(x0, 0):x0 + w]
    return cv2.resize(sub, size, interpolation=cv2.INTER_CUBIC)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return out

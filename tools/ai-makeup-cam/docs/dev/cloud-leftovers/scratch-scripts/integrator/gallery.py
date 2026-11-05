"""Render before/after face-crop galleries for preset JSON files.

usage: gallery.py OUT.png PRESET.json [PRESET.json ...]
Rows = test frames; columns = before, then one 'after' per preset.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.__main__ import render_still
from makeup_cam.config import MakeupSettings

T = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/testimgs")
import os
FRAMES = os.environ.get("FRAMES", "frame_720p,frame_dark,frame_tilt,frame_far").split(",")
CELL = int(sys.argv[2]) if sys.argv[2].isdigit() else 420
presets = [Path(p) for p in sys.argv[2:] if not p.isdigit()]

CROP = os.environ.get("CROP", "face")
EYES = [33, 133, 263, 362, 70, 300, 105, 334, 145, 374]

def crop_box(lm, shape):
    if CROP == "eyes":
        pts = lm[EYES]
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        pad = (x1 - x0) * 0.08
        return int(x0 - pad), int(y0 - pad * 2), int(x1 + pad), int(y1 + pad * 2)
    x0, y0 = lm.min(0); x1, y1 = lm.max(0)
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * 1.55
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.12 * h
    a = int(max(0, cx - side / 2)); b = int(max(0, cy - side / 2))
    return a, b, int(min(shape[1], a + side)), int(min(shape[0], b + side))

def label(img, text):
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 12, 26), (30, 30, 30), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def load_frame(name):
    if name != "frame_near":
        return cv2.imread(str(T / f"{name}.png"))
    # Synthetic webcam close-up: the 720p frame zoomed 2x around the face.
    base = cv2.imread(str(T / "frame_720p.png"))
    lm = render_still(base, MakeupSettings()).landmarks
    cx, cy = lm.mean(0)
    x0 = int(min(max(cx - 320, 0), 640)); y0 = int(min(max(cy - 150, 0), 360))
    return cv2.resize(base[y0:y0 + 360, x0:x0 + 640], (1280, 720), interpolation=cv2.INTER_CUBIC)

rows = []
for name in FRAMES:
    frame = load_frame(name)
    cells = []
    box = None
    for preset in presets:
        settings = MakeupSettings.from_dict(json.loads(preset.read_text(encoding="utf-8")))
        r = render_still(frame, settings)
        if box is None:
            box = crop_box(r.landmarks, frame.shape)
            a, b, c, d = box
            cells.append(label(cv2.resize(frame[b:d, a:c], (CELL, CELL * (d - b) // (c - a)), interpolation=cv2.INTER_CUBIC), f"{name} before"))
        a, b, c, d = box
        cells.append(label(cv2.resize(r.output[b:d, a:c], (CELL, CELL * (d - b) // (c - a)), interpolation=cv2.INTER_CUBIC), f"after: {preset.stem}"))
        print(name, preset.stem, r.face_found, {k: round(v, 1) for k, v in r.timings_ms.items()}, file=sys.stderr)
    rows.append(np.hstack([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=255) for c in cells]))
cv2.imwrite(sys.argv[1], np.vstack(rows))

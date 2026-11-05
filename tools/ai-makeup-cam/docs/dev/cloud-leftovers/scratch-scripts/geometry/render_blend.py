"""Lip color with each blend mode (opacity 0.6) for a visual sanity check."""
import sys
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R, blend as B
from makeup_cam.config import hex_to_bgr

S = sys.argv[1]
img = cv2.imread(f"{S}/testimgs/frame_720p.png").astype(np.float32) / 255
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy")
fs = R.face_scale(lm)
roi = R.roi_around(lm[R.LIPS_OUTER], img.shape, pad=0.35)
mask = R.lips_mask(lm, roi, feather_px=0.025 * fs)
tiles = [img[roi.slices]]
names = ["original"]
for color in ("#C25B62", "#8E2F3C"):
    for mode in ("color", "soft_light", "multiply", "normal"):
        tiles.append(B.apply_color(img[roi.slices], hex_to_bgr(color), mask, 0.6, mode))
        names.append(f"{mode} {color}")
z = 3
rows = []
for i in range(0, len(tiles), 3):
    row = [np.clip(cv2.resize(t, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC) * 255, 0, 255).astype(np.uint8) for t in tiles[i:i + 3]]
    while len(row) < 3:
        row.append(np.zeros_like(row[0]))
    for t, n in zip(row, names[i:i + 3]):
        cv2.putText(t, n, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(np.hstack(row))
cv2.imwrite(f"{S}/renders/geometry/blend_modes_lips.png", np.vstack(rows))

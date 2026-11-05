"""Brow maps on the near (2x) crop: area, hairness, presence, final weight."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions, blend
from makeup_cam.effects import brows

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
img = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
lm = np.load(S / "tmp_eyes_brows" / "lm_frame_720p.npy")
img = np.ascontiguousarray(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)[0:720, 560:1840])
lm = lm * 2 - np.array([560, 0], np.float32)
shift = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # simulate landmark drift (px, up)
lm = lm.copy()
for side in ("left", "right"):
    lm[regions.side_indices(side).brow] -= np.array([0, shift], np.float32)
scale = regions.face_scale(lm)
f = img.astype(np.float32) / 255
rows = []
for side in ("right", "left"):
    feather = scale * (brows._SOFT_BASE + brows._SOFT_RANGE * 0.5)
    roi = regions.roi_around(lm[regions.side_indices(side).brow], img.shape, brows._ROI_PAD, 3.5 * feather + 2)
    area = regions.brow_mask(lm, side, roi, 0.0, feather)
    luma = blend.luminance(f[roi.slices])
    core, skin = area > 0.5, brows._skin_around(blend.luminance(f[roi.slices]), area < 0.02)
    hair = brows._hairness(luma, core, skin, scale)
    w = brows._fill_weight(luma, area, scale)
    heat = lambda m: cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view = img[roi.slices]
    panels = [view, heat(area), heat(hair), heat(area * w)]
    k = 3
    rows.append(np.hstack([cv2.resize(p, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) for p in panels]))
w = max(r.shape[1] for r in rows)
out = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows])
cv2.imwrite(str(S / f"renders/eyes_brows/mask_brows_shift{shift:g}.png"), out)
print(out.shape)

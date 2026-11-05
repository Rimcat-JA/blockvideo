"""Overlay of the shadow density / sheen weight on the near (2x) eye crop, for a few settings."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.effects import eyes

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
TMP = S / "tmp_eyes_brows"
img = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
lm = np.load(TMP / "lm_frame_720p.npy")
img = np.ascontiguousarray(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)[0:720, 560:1840])
lm = lm * 2 - np.array([560, 0], np.float32)
scale = regions.face_scale(lm)
rows = []
for side in ("right", "left"):
    for height, soft in [(0.5, 0.6), (0.25, 0.2), (1.0, 1.0), (1.0, 0.0)]:
        axes = eyes._eye_axes(lm, side)
        outline = eyes._lid_outline(lm, side, axes, scale)
        roi = regions.roi_around(outline, img.shape, 0.15, 4)
        region = regions.polygon_mask(outline, roi, eyes._REGION_FEATHER * scale)
        u = axes.along_coordinate(roi)
        t = eyes._lid_coordinate(lm, side, roi, height)
        mf = eyes._MIN_FADE_PX / (height * eyes._lid_depth(lm, side))
        dens = region * eyes._outer_bound(roi, axes, scale) * eyes._fade_profile(t, soft, mf) * eyes._along_profile(u)
        sheen = region * eyes._sheen_profile(u, t)
        view = img[roi.slices]
        heat = lambda m: cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panels = [cv2.addWeighted(view, 0.45, heat(dens), 0.55, 0), cv2.addWeighted(view, 0.45, heat(sheen), 0.55, 0)]
        f = 3
        row = np.hstack([cv2.resize(p, (p.shape[1] * f, p.shape[0] * f), interpolation=cv2.INTER_NEAREST) for p in panels])
        cv2.putText(row, f"{side} h={height} s={soft}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(row)
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
out = np.vstack(rows)
cv2.imwrite(str(S / "renders/eyes_brows/mask_shadow.png"), out)
print(out.shape)

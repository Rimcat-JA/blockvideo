"""Synthetic blink: upper lash points collapsed toward the lower lash line; render liner + shadow."""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import EyeshadowSettings, EyelinerSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import eyes
S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
img = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
lm0 = np.load(S / "tmp_eyes_brows" / "lm_frame_720p.npy")
img = np.ascontiguousarray(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)[0:720, 560:1840])
lm0 = lm0 * 2 - np.array([560, 0], np.float32)
gray = np.full_like(img, 200)
tiles = []
for openness in (1.0, 0.5, 0.1, 0.0):
    lm = lm0.copy()
    for side in regions.SIDES:
        s = regions.side_indices(side)
        up, lo = lm0[s.eye_upper], lm0[s.eye_lower]
        lm[s.eye_upper] = lo + (up - lo) * openness
    ctx = FrameContext(720, 1280, landmarks=lm)
    for base in (img, gray):
        f = base.astype(np.float32) / 255
        f = eyes.EyeshadowEffect().apply(f, ctx, EyeshadowSettings(opacity=0.8))
        f = eyes.EyelinerEffect().apply(f, ctx, EyelinerSettings(opacity=1.0, thickness=0.7, wing=0.6, lower=0.5))
        assert np.isfinite(f).all()
        out = (f * 255).round().astype(np.uint8)
        s = regions.side_indices("left")
        box = regions.roi_around(np.vstack([lm0[s.eye], lm0[s.brow]]), img.shape, pad=0.1)
        crop = cv2.resize(out[box.slices], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"open {openness}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        tiles.append(crop)
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite(str(S / "renders/eyes_brows/blink.png"), np.vstack(rows))

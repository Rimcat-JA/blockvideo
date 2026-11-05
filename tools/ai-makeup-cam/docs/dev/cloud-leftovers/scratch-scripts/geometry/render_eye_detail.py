"""Zoomed eyelid (several heights), under-eye and eyeliner masks on one eye."""
import sys
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R
S = sys.argv[1]
for name in ("frame_720p", "frame_tilt"):
    img = cv2.imread(f"{S}/testimgs/{name}.png").astype(np.float32) / 255
    lm = np.load(f"{S}/tmp_geometry/{name}.npy")
    fs = R.face_scale(lm)
    tiles = []
    for side in ("right", "left"):
        s = R.side_indices(side)
        roi = R.roi_around(lm[s.eye + s.brow + s.under_eye_bottom], img.shape, pad=0.15)
        for h in (0.3, 0.6, 1.0):
            out = img[roi.slices].copy()
            m = R.eyelid_mask(lm, side, roi, h, feather_px=0.04 * fs)[..., None] * 0.7
            out = out * (1 - m) + np.float32([0.6, 0.2, 0.5]) * m
            u = R.under_eye_mask(lm, side, roi, depth=0.6, feather_px=0.04 * fs)[..., None] * 0.6
            out = out * (1 - u) + np.float32([0.9, 0.7, 0.2]) * u
            liner = R.polyline_mask(lm[s.eye_upper], 0.045 * fs, roi, feather_px=0.004 * fs, thickness_profile=np.linspace(0.25, 1.0, 9))[..., None] * 0.9
            out = out * (1 - liner)
            z = 7
            t = np.clip(cv2.resize(out, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC) * 255, 0, 255).astype(np.uint8)
            cv2.putText(t, f"{side} h={h}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
    hmin = min(t.shape[0] for t in tiles); wmin = min(t.shape[1] for t in tiles)
    tiles = [t[:hmin, :wmin] for t in tiles]
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(f"{S}/renders/geometry/eye_detail_{name}.png", grid)

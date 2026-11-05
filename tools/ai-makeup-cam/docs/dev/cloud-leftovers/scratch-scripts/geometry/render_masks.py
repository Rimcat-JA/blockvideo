"""Overlay every region mask helper on the test frames (zoomed)."""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R

S = Path(sys.argv[1])
OUT = S / "renders/geometry"

def tint(frame, mask, roi, bgr, strength=0.65):
    view = frame[roi.slices]
    m = (mask * strength)[..., None]
    view[:] = view * (1 - m) + np.array(bgr, np.float32) * m

for name in sys.argv[2:]:
    img = cv2.imread(str(S / "testimgs" / f"{name}.png")).astype(np.float32) / 255
    lm = np.load(S / "tmp_geometry" / f"{name}.npy")
    fs = R.face_scale(lm)
    roi = R.roi_around(lm, img.shape, pad=0.1, margin_px=0.3 * fs)
    out = img.copy()
    tint(out, R.face_oval_mask(lm, roi, feather_px=0.08 * fs) * 0.35, roi, (0, 1, 0))
    tint(out, R.lips_mask(lm, roi, outline=0.0, feather_px=0.02 * fs), roi, (0, 0, 1))
    for side in R.SIDES:
        s = R.side_indices(side)
        tint(out, R.eyelid_mask(lm, side, roi, height=0.6, feather_px=0.04 * fs), roi, (1, 0, 0.6))
        tint(out, R.under_eye_mask(lm, side, roi, depth=0.7, feather_px=0.04 * fs), roi, (1, 0.6, 0))
        tint(out, R.brow_mask(lm, side, roi, thickness=-0.2, feather_px=0.015 * fs), roi, (0, 0.5, 1))
        tint(out, R.eye_opening_mask(lm, side, roi), roi, (1, 1, 0), 0.4)
        c = R.cheek_center(lm, side)
        tint(out, R.ellipse_mask(c, (0.32 * fs, 0.22 * fs), -15 if side == "right" else 15, roi, feather_px=0.12 * fs), roi, (0.6, 0.3, 1))
        tint(out, R.polygon_mask(lm[s.nose_wing], roi, feather_px=0.03 * fs), roi, (0, 1, 1))
        liner = lm[s.eye_upper]
        prof = np.linspace(0.3, 1.0, len(liner))
        tint(out, R.polyline_mask(liner, 0.06 * fs, roi, feather_px=0.006 * fs, thickness_profile=prof), roi, (0.05, 0.05, 0.05), 0.9)
    crop = out[roi.slices]
    z = 1000 / max(roi.w, roi.h)
    crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / f"masks_{name}.png"), np.clip(crop * 255, 0, 255).astype(np.uint8))
    # Open-mouth check: lips mask alone in grayscale
    lroi = R.roi_around(lm[R.LIPS_OUTER], img.shape, pad=0.2)
    lips = R.lips_mask(lm, lroi, feather_px=0.02 * fs)
    zl = 600 / lroi.w
    both = np.hstack([img[lroi.slices], np.repeat(lips[..., None], 3, axis=2)])
    cv2.imwrite(str(OUT / f"lipsmask_{name}.png"), cv2.resize(np.clip(both * 255, 0, 255).astype(np.uint8), None, fx=zl, fy=zl, interpolation=cv2.INTER_NEAREST))
    print(name, "face_scale", round(fs, 1), "roi", roi)

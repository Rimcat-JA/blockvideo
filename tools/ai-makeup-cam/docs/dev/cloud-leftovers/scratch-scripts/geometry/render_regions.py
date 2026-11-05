"""Draw every named region on the test frames (zoomed) for visual checks."""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R

S = Path(sys.argv[1])
OUT = S / "renders/geometry"

def zoom_crop(img, lm, pad=0.15, target=1100):
    roi = R.roi_around(lm, img.shape, pad)
    z = target / max(roi.w, roi.h)
    crop = cv2.resize(img[roi.slices], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    return crop, roi, z

def P(pts, roi, z):
    return ((np.asarray(pts, np.float64).reshape(-1, 2) - roi.offset) * z).astype(np.int32)

def label(img, p, text, color, scale=0.38):
    org = (int(p[0]) + 3, int(p[1]) - 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def line(img, lm, idx, roi, z, color, closed=False, labels=False, mark_first=False):
    pts = P(lm[idx], roi, z)
    cv2.polylines(img, [pts], closed, color, 1, cv2.LINE_AA)
    if mark_first:
        cv2.circle(img, tuple(pts[0]), 6, color, 2, cv2.LINE_AA)
    if labels:
        for k, (i, p) in enumerate(zip(idx, pts)):
            cv2.circle(img, tuple(p), 2, color, -1, cv2.LINE_AA)
            label(img, p, f"{i}" if k else f"{i}*", color)

def draw_regions(name):
    img = cv2.imread(str(S / "testimgs" / f"{name}.png"))
    lm = np.load(S / "tmp_geometry" / f"{name}.npy")
    crop, roi, z = zoom_crop(img, lm)
    line(crop, lm, R.FACE_OVAL, roi, z, (0, 200, 0), closed=True)
    line(crop, lm, R.LIPS_OUTER, roi, z, (60, 60, 255), closed=True)
    line(crop, lm, R.LIPS_INNER, roi, z, (255, 0, 255), closed=True)
    line(crop, lm, R.NOSE_BRIDGE, roi, z, (200, 200, 200))
    p = P(lm[[R.NOSE_TIP]], roi, z)[0]; cv2.circle(crop, tuple(p), 5, (255, 255, 255), 2); label(crop, p, "NOSE_TIP", (255, 255, 255))
    for side in R.SIDES:
        s = R.side_indices(side)
        tag = "L" if side == "left" else "R"
        line(crop, lm, s.eye_upper, roi, z, (0, 255, 255), labels=False, mark_first=True)
        line(crop, lm, s.eye_lower, roi, z, (0, 165, 255), labels=False)
        line(crop, lm, s.iris, roi, z, (255, 255, 0), closed=True)
        line(crop, lm, s.eye_crease, roi, z, (180, 120, 255))
        line(crop, lm, s.under_eye_top, roi, z, (255, 180, 120))
        line(crop, lm, s.under_eye_bottom, roi, z, (255, 120, 60))
        line(crop, lm, s.brow, roi, z, (0, 120, 255), closed=True)
        line(crop, lm, s.nose_wing, roi, z, (120, 255, 120), closed=True)
        c = P(R.cheek_center(lm, side), roi, z)[0]
        cv2.circle(crop, tuple(c), 8, (147, 20, 255), 2, cv2.LINE_AA); label(crop, c, f"{tag}_CHEEK", (147, 20, 255), 0.5)
        e = P(lm[s.eye_upper[0]], roi, z)[0]; label(crop, e + (0, 30), f"{tag} inner", (0, 255, 255), 0.45)
    cv2.imwrite(str(OUT / f"regions_{name}.png"), crop)
    return img, lm

def draw_eye_labels(name, img, lm):
    for side in R.SIDES:
        s = R.side_indices(side)
        idx = s.eye_upper + s.brow_upper + s.brow_lower
        crop, roi, z = zoom_crop(img, lm[idx], pad=0.35, target=900)
        line(crop, lm, s.eye_upper, roi, z, (0, 255, 255), labels=True, mark_first=True)
        line(crop, lm, s.eye_lower, roi, z, (0, 165, 255), labels=True, mark_first=True)
        line(crop, lm, s.brow_upper, roi, z, (0, 120, 255), labels=True, mark_first=True)
        line(crop, lm, s.brow_lower, roi, z, (60, 200, 255), labels=True, mark_first=True)
        line(crop, lm, s.iris, roi, z, (255, 255, 0), closed=True, labels=True)
        line(crop, lm, s.eye_crease, roi, z, (180, 120, 255), labels=True, mark_first=True)
        cv2.imwrite(str(OUT / f"labels_{name}_{side}_eye.png"), crop)
    crop, roi, z = zoom_crop(img, lm[R.LIPS_OUTER], pad=0.2, target=900)
    line(crop, lm, R.LIPS_UPPER_OUTER, roi, z, (60, 60, 255), labels=True, mark_first=True)
    line(crop, lm, R.LIPS_LOWER_OUTER, roi, z, (60, 160, 255), labels=True)
    line(crop, lm, R.LIPS_UPPER_INNER, roi, z, (255, 0, 255), labels=True, mark_first=True)
    line(crop, lm, R.LIPS_LOWER_INNER, roi, z, (255, 120, 255), labels=True)
    cv2.imwrite(str(OUT / f"labels_{name}_lips.png"), crop)
    idx = R.RIGHT_CHEEK_CENTER + R.LEFT_CHEEK_CENTER + R.RIGHT_NOSE_WING + R.LEFT_NOSE_WING
    crop, roi, z = zoom_crop(img, lm[idx], pad=0.25, target=1000)
    for side in R.SIDES:
        s = R.side_indices(side)
        line(crop, lm, s.cheek_center, roi, z, (147, 20, 255), closed=True, labels=True)
        line(crop, lm, s.nose_wing, roi, z, (120, 255, 120), closed=True, labels=True)
        line(crop, lm, s.under_eye_top, roi, z, (255, 180, 120), labels=True, mark_first=True)
        line(crop, lm, s.under_eye_bottom, roi, z, (255, 120, 60), labels=True, mark_first=True)
    cv2.imwrite(str(OUT / f"labels_{name}_cheeks.png"), crop)

for name in sys.argv[2:]:
    img, lm = draw_regions(name)
    draw_eye_labels(name, img, lm)
print("ok")

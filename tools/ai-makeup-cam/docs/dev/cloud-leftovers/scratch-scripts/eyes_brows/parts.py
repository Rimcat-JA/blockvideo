import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions, blend
from makeup_cam.effects import eyes, brows
S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
lm = np.load(S / "tmp_eyes_brows" / "lm_frame_720p.npy")
img = cv2.imread(str(S / "testimgs" / "frame_720p.png")).astype(np.float32) / 255
def t(label, fn, n=200):
    fn(); s = time.perf_counter()
    for _ in range(n): fn()
    print(f"{label:28s} {(time.perf_counter() - s) / n * 1e3:.3f} ms")
scale = regions.face_scale(lm); side = "left"
axes = eyes._eye_axes(lm, side)
width = scale * 0.025; feather = scale * 0.0084
upper, prof = eyes._upper_liner(lm, side, axes, width, scale * 0.05)
roi = regions.roi_around(np.vstack([upper, lm[regions.side_indices(side).eye]]), img.shape, 0, width + 4)
print("liner roi", roi.w, roi.h)
t("eye_axes", lambda: eyes._eye_axes(lm, side))
t("upper_liner", lambda: eyes._upper_liner(lm, side, axes, width, scale * 0.05))
t("polyline_mask", lambda: regions.polyline_mask(upper, width, roi, feather, prof))
t("eye_opening_mask", lambda: regions.eye_opening_mask(lm, side, roi))
m = regions.polyline_mask(upper, width, roi, feather, prof)
t("ink", lambda: eyes._ink(img[roi.slices], (0.1, 0.13, 0.18), m))
t("face_scale", lambda: regions.face_scale(lm))
# brows
feather = scale * 0.035
roi = regions.roi_around(lm[regions.side_indices(side).brow], img.shape, 0.2, 3.5 * feather + 2)
print("brow roi", roi.w, roi.h)
t("brow_mask", lambda: regions.brow_mask(lm, side, roi, 0.0, feather))
area = regions.brow_mask(lm, side, roi, 0.0, feather)
luma = blend.luminance(img[roi.slices])
t("luminance", lambda: blend.luminance(img[roi.slices]))
t("fill_weight", lambda: brows._fill_weight(luma, area, scale))
skin = brows._skin_around(luma, area < 0.02)
t("skin_around", lambda: brows._skin_around(luma, area < 0.02))
t("hairness", lambda: brows._hairness(luma, area > 0.5, skin, scale))
w = area * brows._fill_weight(luma, area, scale)
v = img[roi.slices]
t("mix multiply", lambda: blend.mix(v, blend.blend_multiply(v, (0.2, 0.26, 0.36)), w))

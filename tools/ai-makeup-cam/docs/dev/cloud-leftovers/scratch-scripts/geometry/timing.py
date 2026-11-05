"""Time the mask helpers at 1280x720 for the test face and a 2x closer face."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R, blend as B

lm0 = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy")

def bench(fn, n=60, repeats=5):
    fn()
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t) / n * 1e3)
    return best

for zoom in (1.0, 2.0):
    c = lm0.mean(0)
    lm = (lm0 - c) * zoom + np.array([640, 380])
    fs = R.face_scale(lm)
    face = R.roi_around(lm, (720, 1280), pad=0.05, margin_px=0.25 * fs)
    lips_roi = R.roi_around(lm[R.LIPS_OUTER], (720, 1280), pad=0.1, margin_px=0.1 * fs)
    s = R.side_indices("left")
    eye_roi = R.roi_around(lm[s.eye + s.brow], (720, 1280), pad=0.1, margin_px=0.15 * fs)
    cheek = R.cheek_center(lm, "left")
    cheek_roi = R.roi_around(np.array([cheek]), (720, 1280), pad=0, margin_px=0.8 * fs)
    print(f"--- zoom {zoom}: face_scale {fs:.0f}px, face ROI {face.w}x{face.h}, lips ROI {lips_roi.w}x{lips_roi.h}, eye ROI {eye_roi.w}x{eye_roi.h}")
    rows = {
        "face_oval_mask (feather 0.08*fs)": lambda: R.face_oval_mask(lm, face, feather_px=0.08 * fs),
        "face_oval_mask (no feather)": lambda: R.face_oval_mask(lm, face),
        "lips_mask (feather 0.02*fs)": lambda: R.lips_mask(lm, lips_roi, feather_px=0.02 * fs),
        "eyelid_mask (h=0.6, feather 0.05*fs)": lambda: R.eyelid_mask(lm, "left", eye_roi, 0.6, feather_px=0.05 * fs),
        "under_eye_mask (feather 0.05*fs)": lambda: R.under_eye_mask(lm, "left", eye_roi, feather_px=0.05 * fs),
        "brow_mask (feather 0.015*fs)": lambda: R.brow_mask(lm, "left", eye_roi, feather_px=0.015 * fs),
        "eye_opening_mask": lambda: R.eye_opening_mask(lm, "left", eye_roi),
        "polyline_mask eyeliner (taper)": lambda: R.polyline_mask(lm[s.eye_upper], 0.05 * fs, eye_roi, feather_px=0.005 * fs, thickness_profile=np.linspace(0.3, 1, 9)),
        "ellipse_mask cheek (feather 0.12*fs)": lambda: R.ellipse_mask(cheek, (0.35 * fs, 0.25 * fs), 15, cheek_roi, feather_px=0.12 * fs),
        "polygon_mask nose wing (feather 0.03*fs)": lambda: R.polygon_mask(lm[s.nose_wing], face, feather_px=0.03 * fs),
    }
    lips_view = np.random.default_rng(0).random((lips_roi.h, lips_roi.w, 3), dtype=np.float32)
    lips_m = R.lips_mask(lm, lips_roi, feather_px=0.02 * fs)
    face_view = np.random.default_rng(0).random((face.h, face.w, 3), dtype=np.float32)
    face_m = R.face_oval_mask(lm, face, feather_px=0.08 * fs)
    rows["apply_color color (lips ROI)"] = lambda: B.apply_color(lips_view, (0.38, 0.36, 0.76), lips_m, 0.5, "color")
    rows["apply_color soft_light (lips ROI)"] = lambda: B.apply_color(lips_view, (0.38, 0.36, 0.76), lips_m, 0.5, "soft_light")
    rows["apply_color color (face ROI)"] = lambda: B.apply_color(face_view, (0.38, 0.36, 0.76), face_m, 0.5, "color")
    rows["apply_color soft_light (face ROI)"] = lambda: B.apply_color(face_view, (0.38, 0.36, 0.76), face_m, 0.5, "soft_light")
    rows["adjust_brightness_contrast (face ROI)"] = lambda: B.adjust_brightness_contrast(face_view, 0.03, 0.1, face_m)
    for name, fn in rows.items():
        print(f"  {name:42s} {bench(fn):6.3f} ms")

import sys, time, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R
from makeup_cam.effects import skin
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)
ws = skin._workspace(lm, (720, 1280, 3))
print("small size", ws.small_size, "scale", ws.scale)
def t(label, fn, n=200):
    fn(); s = time.perf_counter()
    for _ in range(n): fn()
    print(f"{label}: {(time.perf_counter()-s)/n*1000:.3f} ms")
L, roi, s = ws.landmarks, ws.small_roi, ws.scale
t("oval", lambda: R.face_oval_mask(L, roi, -0.07*s, 0.1*s))
t("eye", lambda: R.eye_opening_mask(L, "left", roi, 0.06*s, 0.05*s))
t("brow", lambda: R.brow_mask(L, "left", roi, 0.8, 0.05*s))
t("mouth", lambda: R.polygon_mask(R.expand_polygon(L[R.LIPS_OUTER], 0.04*s), roi, 0.05*s))
t("nostril", lambda: skin._nostril_mask(L, "left", roi, 0.03*s, 0.05*s))
t("under_eye", lambda: R.under_eye_mask(L, "left", roi, 0.9, 0.07*s))
small = np.random.rand(ws.small_size[1], ws.small_size[0], 3).astype(np.float32)
t("lab", lambda: cv2.cvtColor(small, cv2.COLOR_BGR2Lab))
t("ycrcb", lambda: cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb))
t("transform", lambda: cv2.transform(small, np.eye(3, dtype=np.float32)))
u8 = (small*255).astype(np.uint8)
t("median7", lambda: cv2.medianBlur(u8, 7))
t("gauss3ch", lambda: cv2.GaussianBlur(small, (0, 0), 3.0))
big = np.random.rand(160, 140, 3).astype(np.float32)
t("full gray", lambda: cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests")
from conftest import build_face_frame_720p
frame = build_face_frame_720p().astype(np.float32) / 255
patch = frame[ws.roi.slices]
sm = skin._resize(patch, ws.small_size, cv2.INTER_AREA)
print(sm.flags['C_CONTIGUOUS'], sm.dtype, sm.shape)
t("lab real", lambda: cv2.cvtColor(sm, cv2.COLOR_BGR2Lab))
lab = cv2.cvtColor(sm, cv2.COLOR_BGR2Lab)
t("lab2bgr real", lambda: cv2.cvtColor(lab, cv2.COLOR_Lab2BGR))
t("lab rand again", lambda: cv2.cvtColor(small, cv2.COLOR_BGR2Lab))

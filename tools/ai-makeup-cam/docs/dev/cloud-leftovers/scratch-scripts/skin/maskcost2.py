import sys, time, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions as R
from makeup_cam.effects import skin
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)
def t(label, fn, n=300):
    fn(); s = time.perf_counter()
    for _ in range(n): fn()
    print(f"{label}: {(time.perf_counter()-s)/n*1000:.3f} ms")
for ws_scale in (48.0, 24.0):
    skin._WORK_SCALE = ws_scale
    ws = skin._workspace(lm, (720, 1280, 3))
    L, roi, s = ws.landmarks, ws.small_roi, ws.scale
    print("grid", ws.small_size)
    t(" oval feather", lambda: R.face_oval_mask(L, roi, -0.07*s, 0.1*s))
    t(" oval nofeather", lambda: R.face_oval_mask(L, roi, -0.07*s))
    t(" oval nofeather+feather()", lambda: R.feather(R.face_oval_mask(L, roi, -0.07*s), 0.1*s))
    t(" eye nofeather", lambda: R.eye_opening_mask(L, "left", roi, 0.06*s))
    t(" eye feather", lambda: R.eye_opening_mask(L, "left", roi, 0.06*s, 0.05*s))
    t(" gauss", lambda: cv2.GaussianBlur(np.zeros(roi.h*roi.w, np.float32).reshape(roi.h, roi.w), (0,0), 0.05*s))

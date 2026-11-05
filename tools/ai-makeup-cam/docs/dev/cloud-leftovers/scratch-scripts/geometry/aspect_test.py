import sys
from pathlib import Path
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
S = Path(sys.argv[1])
opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(S / "models/face_landmarker.task")), running_mode=vision.RunningMode.IMAGE, num_faces=1)
fl = vision.FaceLandmarker.create_from_options(opts)
def detect(bgr):
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32) if r.face_landmarks else None
full = cv2.imread(str(S / "testimgs/frame_720p.png"))
a = detect(full)
sq = np.ascontiguousarray(full[:, 280:1000])
b = detect(sq); b[:, 0] += 280
print("full vs square crop: mean abs diff", np.abs(a - b).mean(0), "max", np.abs(a - b).max(0))
for idx in (468, 473, 1, 152, 10, 13, 14):
    print(idx, a[idx], b[idx])
np.save(S / "tmp_geometry/frame_720p_sq.npy", b)
# Also via VIDEO mode
opts2 = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(S / "models/face_landmarker.task")), running_mode=vision.RunningMode.VIDEO, num_faces=1)
fv = vision.FaceLandmarker.create_from_options(opts2)
rgb = np.ascontiguousarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
for t in range(3):
    r = fv.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 33 * (t + 1))
    v = np.array([[p.x * 1280, p.y * 720] for p in r.face_landmarks[0]], np.float32)
    print("video", t, np.abs(v - b).mean(0))
fv.close(); fl.close()

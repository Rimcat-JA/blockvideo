import sys
from pathlib import Path
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
S = Path(sys.argv[1])
M = str(S / "models/face_landmarker.task")
fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=M), running_mode=vision.RunningMode.IMAGE, num_faces=1))
def detect(bgr):
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32) if r.face_landmarks else None
sqref = np.load(S / "tmp_geometry/frame_720p_sq.npy")
for name in ["frame_720p", "frame_tilt", "frame_far", "frame_dark"]:
    full = cv2.imread(str(S / f"testimgs/{name}.png"))
    pad = cv2.copyMakeBorder(full, 0, 560, 0, 0, cv2.BORDER_CONSTANT, value=(70, 75, 80))
    p = detect(pad)
    print(name, "letterbox detected:", p is not None)
    if p is not None:
        np.save(S / f"tmp_geometry/{name}_lb.npy", p)
        if name == "frame_720p":
            print("  letterbox vs sqcrop", np.abs(p - sqref).mean(0), np.abs(p - sqref).max(0))
    fv = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=M), running_mode=vision.RunningMode.VIDEO, num_faces=1))
    rgb = np.ascontiguousarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    prev = None
    for t in range(12):
        r = fv.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 33 * (t + 1))
        if not r.face_landmarks:
            print("  video", t, "none"); continue
        v = np.array([[q.x * 1280, q.y * 720] for q in r.face_landmarks[0]], np.float32)
        if prev is not None and t in (1, 2, 3, 5, 11):
            print("  video step", t, "delta", np.abs(v - prev).max(), "vs letterbox", None if p is None else np.abs(v - p).mean())
        prev = v
    np.save(S / f"tmp_geometry/{name}_vid.npy", prev)
    fv.close()
fl.close()

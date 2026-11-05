"""Converged (VIDEO mode, repeated frame) landmarks for the scratch test images.

The first detection on a non-square frame is vertically off by up to ~28 px;
tracking on the same frame converges within ~10 frames. Images where the
full-frame detector misses the face are processed on a square center crop.
"""
import sys
from pathlib import Path
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

S = Path(sys.argv[1])
M = str(S / "models/face_landmarker.task")

def converge(bgr, frames=15):
    fv = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=M), running_mode=vision.RunningMode.VIDEO, num_faces=1))
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    lm = None
    for t in range(frames):
        r = fv.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 33 * (t + 1))
        if r.face_landmarks:
            lm = np.array([[q.x * w, q.y * h] for q in r.face_landmarks[0]], np.float32)
    fv.close()
    return lm

for name in sys.argv[2:]:
    bgr = cv2.imread(str(S / "testimgs" / f"{name}.png"))
    lm = converge(bgr)
    for x0, y0, size in ((280, 0, 720), (440, 200, 400)):
        if lm is not None:
            break
        lm = converge(np.ascontiguousarray(bgr[y0:y0 + size, x0:x0 + size]))
        if lm is not None:
            lm += (x0, y0)
            print(name, "via crop", x0, y0, size)
    np.save(S / "tmp_geometry" / f"{name}.npy", lm)
    print(name, lm.min(0), lm.max(0))

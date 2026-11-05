import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.segmentation import Segmenter
from makeup_cam.tracking import FaceTracker
from makeup_cam.pipeline import draw_landmarks
M = sys.argv[1]; T = sys.argv[2]; OUT = sys.argv[3]
rows = []
for name in ("astronaut_512.png", "frame_720p.png", "frame_dark.png", "frame_tilt.png", "frame_far.png"):
    frame = cv2.imread(f"{T}/{name}")
    seg = Segmenter(M + "/selfie_multiclass_256x256.tflite", interval=1, background=False)
    tr = FaceTracker(M + "/face_landmarker.task", 0.5)
    for i in range(4):
        masks = seg.update(frame, 33 * i, i); lm, z = tr.detect(frame, 33 * i)
    seg.close(); tr.close()
    over = frame.astype(np.float32)
    for mask, color in ((masks.hair, (255, 0, 255)), (masks.face_skin, (0, 255, 0)), (masks.body_skin, (255, 255, 0))):
        over = over * (1 - 0.5 * mask[..., None]) + np.array(color, np.float32) * 0.5 * mask[..., None]
    over = over.astype(np.uint8)
    if lm is not None: over = draw_landmarks(over, lm)
    h = 360; row = np.hstack([cv2.resize(frame, (round(frame.shape[1] * h / frame.shape[0]), h)), cv2.resize(over, (round(frame.shape[1] * h / frame.shape[0]), h))])
    rows.append(cv2.copyMakeBorder(row, 0, 0, 0, 1280 * 2 - row.shape[1], cv2.BORDER_CONSTANT))
    print(name, "face", lm is not None)
cv2.imwrite(OUT, np.vstack(rows))

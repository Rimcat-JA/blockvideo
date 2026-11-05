import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import tracking
from skimage import data
M = sys.argv[1] + "/face_landmarker.task"
imgs = {n: (lambda a: cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR))(getattr(data, n)()) for n in ("coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry", "coins")}
far = cv2.imread(sys.argv[2] + "/frame_far.png")
for conf in (0.4, 0.3, 0.25):
    res = []
    for n, img in list(imgs.items()) + [("far", far)]:
        tr = tracking.FaceTracker(None, 0, model=tracking.MediaPipeFaceLandmarker(M, conf))
        hits = sum(tr.detect(img, 33 * i)[0] is not None for i in range(6))
        tr.close(); res.append(f"{n}={hits}")
    print(conf, " ".join(res))

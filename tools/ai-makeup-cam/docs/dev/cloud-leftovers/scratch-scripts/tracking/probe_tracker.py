import sys, time, glob, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import tracking
M = sys.argv[1]
for conf in (0.5, 0.4, 0.3):
    for path in sorted(glob.glob(sys.argv[2] + '/*.png')):
        bgr = cv2.imread(path)
        model = tracking.MediaPipeFaceLandmarker(M + "/face_landmarker.task", min_detection_confidence=conf)
        tr = tracking.FaceTracker(None, smoothing=0.0, model=model)
        found = []
        for i in range(6):
            lm, z = tr.detect(bgr, 33 * i)
            found.append(lm is not None)
        iod = tracking.face_scale_px(lm) if lm is not None else 0
        print(conf, path.split('/')[-1], found, "iod %.1f" % iod, tr._window)
        tr.close()

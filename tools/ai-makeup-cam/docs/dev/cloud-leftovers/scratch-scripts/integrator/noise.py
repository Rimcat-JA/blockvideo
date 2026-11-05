import os, sys
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
import numpy as np
from makeup_cam.tracking import FaceTracker
from makeup_cam.segmentation import Segmenter
t = FaceTracker(os.environ["M"] + "/face_landmarker.task")
s = Segmenter(os.environ["M"] + "/selfie_multiclass_256x256.tflite", background=False)
img = np.zeros((720, 1280, 3), np.uint8)
t.detect(img, 0); s.update(img, 0, 0)
t.close(); s.close()
print("done")

import logging, sys, time
import cv2, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
logging.basicConfig(level=logging.WARNING)
from makeup_cam.config import MakeupSettings
from makeup_cam.models import ensure_model, FACE_LANDMARKER, SELFIE_MULTICLASS
from makeup_cam.tracking import FaceTracker
from makeup_cam.segmentation import Segmenter
from makeup_cam.pipeline import MakeupPipeline

img = cv2.imread(sys.argv[1])
s = MakeupSettings()
p = MakeupPipeline(s, tracker=FaceTracker(ensure_model(FACE_LANDMARKER)), segmenter=Segmenter(ensure_model(SELFIE_MULTICLASS), background=False))
for i in range(20):
    t = time.perf_counter()
    r = p.process(img, i * 33)
    print(i, r.face_found, {k: round(v, 1) for k, v in r.timings_ms.items()})
p.close()

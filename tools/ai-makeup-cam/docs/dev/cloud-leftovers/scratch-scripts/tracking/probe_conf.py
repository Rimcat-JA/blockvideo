import sys, glob, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = sys.argv[1]
buf = open(M + '/face_landmarker.task', 'rb').read()
def mk(conf):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=conf, min_face_presence_confidence=conf, min_tracking_confidence=conf))
def crop_sq(img):
    h, w = img.shape[:2]; x0 = (w - h)//2
    return img[:, x0:x0+h]
for path in sorted(glob.glob(sys.argv[2] + '/frame_*.png')):
    bgr = cv2.imread(path)
    res = []
    for conf in (0.5, 0.3, 0.2):
        for name, img in (("full640", cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)), ("center", cv2.resize(crop_sq(bgr), (640, 640), interpolation=cv2.INTER_AREA))):
            fl = mk(conf)
            found = [len(fl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), 33*(i+1)).face_landmarks) for i in range(3)]
            fl.close()
            res.append(f"{conf}/{name}:{found}")
    print(path.split('/')[-1], ' '.join(res))

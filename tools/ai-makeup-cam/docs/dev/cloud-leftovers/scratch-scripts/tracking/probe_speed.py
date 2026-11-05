import sys, time, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = sys.argv[1]; IMG = sys.argv[2]
bgr = cv2.imread(IMG)
print("frame", bgr.shape)
def mk_fl():
    buf = open(M + '/face_landmarker.task', 'rb').read()
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO, num_faces=1))
def mk_seg():
    buf = open(M + '/selfie_multiclass_256x256.tflite', 'rb').read()
    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))
def prep(img, long):
    h, w = img.shape[:2]; s = long / max(h, w)
    if s < 1: img = cv2.resize(img, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for long in (1280, 640, 480):
    fl = mk_fl(); ts = 0
    for i in range(40):
        ts += 33; t = time.perf_counter()
        rgb = prep(bgr, long)
        r = fl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if i == 5: times = []
        if i >= 5: times.append((time.perf_counter() - t) * 1000)
    print("landmarks long", long, "faces", len(r.face_landmarks), "ms median %.1f" % np.median(times))
    fl.close()
for long in (384, 256):
    sg = mk_seg(); ts = 0
    for i in range(25):
        ts += 33; t = time.perf_counter()
        rgb = prep(bgr, long)
        r = sg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        ms = [m.numpy_view() for m in r.confidence_masks]
        if i == 5: times = []
        if i >= 5: times.append((time.perf_counter() - t) * 1000)
    print("seg long", long, len(ms), ms[0].shape, ms[0].dtype, "ms median %.1f" % np.median(times))
    sg.close()

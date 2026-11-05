import sys, time, threading, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = sys.argv[1]
bgr = cv2.imread(sys.argv[2])
rgb = cv2.cvtColor(cv2.resize(bgr, (384, 216), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
sg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_buffer=open(M + '/selfie_multiclass_256x256.tflite','rb').read()), running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))
def py_work(n):
    t = time.perf_counter(); s = 0
    for i in range(n): s += i * i
    return (time.perf_counter() - t) * 1000
base = py_work(2_000_000); print("pure python alone ms", round(base, 1))
ts = [0]
def seg_loop(stop):
    while not stop.is_set():
        ts[0] += 33; sg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts[0])
stop = threading.Event(); th = threading.Thread(target=seg_loop, args=(stop,)); th.start(); time.sleep(0.3)
print("pure python with seg thread ms", round(py_work(2_000_000), 1))
stop.set(); th.join()
t = time.perf_counter(); n = 10
for i in range(n): ts[0] += 33; sg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts[0])
print("seg alone ms", round((time.perf_counter() - t) * 1000 / n, 1))
sg.close()

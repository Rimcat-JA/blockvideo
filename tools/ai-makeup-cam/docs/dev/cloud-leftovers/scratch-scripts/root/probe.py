import sys, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from skimage import data
M = sys.argv[1]
rgb = np.ascontiguousarray(data.astronaut())
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=M+'/face_landmarker.task'), running_mode=vision.RunningMode.VIDEO, num_faces=1))
r = fl.detect_for_video(img, 0)
print('faces', len(r.face_landmarks), len(r.face_landmarks[0]))
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=M+'/selfie_multiclass_256x256.tflite'), running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))
s = seg.segment_for_video(img, 0)
print('masks', len(s.confidence_masks), s.confidence_masks[0].numpy_view().shape, [float(m.numpy_view().mean()) for m in s.confidence_masks])
print(seg.labels if hasattr(seg,'labels') else '')
import time; t=time.time()
for i in range(10): fl.detect_for_video(img, 33*(i+1)); seg.segment_for_video(img, 33*(i+1))
print('ms/frame', (time.time()-t)*100)
cv2.imwrite('probe/astronaut.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

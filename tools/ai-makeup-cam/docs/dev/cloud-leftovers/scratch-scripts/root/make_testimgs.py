import numpy as np, cv2
from skimage import data
rgb = data.astronaut(); bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
# 1280x720 webcam-like frame: face fairly large, centered
canvas = np.full((720,1280,3), (70,75,80), np.uint8)
face = cv2.resize(bgr, (720,720), interpolation=cv2.INTER_CUBIC)
canvas[:, 280:1000] = face
cv2.imwrite('testimgs/frame_720p.png', canvas)
# darker / warm indoor light
dark = np.clip(((canvas/255.0)**1.6)*255*np.array([0.85,0.95,1.05]),0,255).astype(np.uint8)
cv2.imwrite('testimgs/frame_dark.png', dark)
# rotated head tilt
M = cv2.getRotationMatrix2D((640,300), 14, 1.0)
cv2.imwrite('testimgs/frame_tilt.png', cv2.warpAffine(canvas, M, (1280,720), borderMode=cv2.BORDER_REFLECT))
# smaller/farther face
small = np.full((720,1280,3), (70,75,80), np.uint8)
f2 = cv2.resize(bgr,(400,400)); small[200:600, 440:840] = f2
cv2.imwrite('testimgs/frame_far.png', small)
cv2.imwrite('testimgs/astronaut_512.png', bgr)
print('ok')

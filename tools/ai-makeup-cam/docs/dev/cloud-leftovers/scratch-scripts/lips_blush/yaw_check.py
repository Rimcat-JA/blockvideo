"""Pseudo head-yaw: perspective-warp the near frame so the image-right half of the
face recedes, run the real pipeline (tracker + blush + lips) and save crops."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import MakeupSettings
from makeup_cam.effects.blush import BlushEffect
from makeup_cam.effects.lips import LipEffect
from makeup_cam.pipeline import MakeupPipeline
S = Path('/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad')
img = cv2.imread(str(S / 'tmp_lips_blush/frame_near.png'))
h, w = img.shape[:2]
cx = 640
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
rows = []
for squeeze in (0.0, 0.25, 0.4):
    dst = np.float32([[0, 0], [w - squeeze * 900, squeeze * 120], [w - squeeze * 900, h - squeeze * 120], [0, h]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    s = MakeupSettings()
    s.blush.opacity = 0.5
    s.blush.shape = "oval"
    pipe = MakeupPipeline(s, model_dir=S / 'models', effects=[BlushEffect(), LipEffect()])
    for i in range(8):
        res = pipe.process(warped, 33 * (i + 1))
    pipe.close()
    lm = res.landmarks
    box = regions.roi_around(lm[regions.FACE_OVAL], warped.shape, pad=0.05)
    a = cv2.resize(warped[box.slices], (360, int(360 * box.h / box.w)))
    b = cv2.resize(res.output[box.slices], (360, int(360 * box.h / box.w)))
    d = cv2.resize(np.clip(cv2.absdiff(res.output, warped).astype(np.float32) * 6, 0, 255).astype(np.uint8)[box.slices], (360, int(360 * box.h / box.w)))
    rows.append(np.hstack([a, b, d]))
hh = min(r.shape[0] for r in rows)
cv2.imwrite(str(S / 'renders/lips_blush/blush_yaw.png'), np.vstack([r[:hh] for r in rows]))

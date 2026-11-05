"""Both effects at default settings through the real pipeline (with segmentation): face crops before/after."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import MakeupSettings
from makeup_cam.effects.base import Effect
from makeup_cam.effects.blush import BlushEffect
from makeup_cam.effects.lips import LipEffect
from makeup_cam.pipeline import MakeupPipeline
S = Path('/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad')

class MaskProbe(Effect):  # named "skin" so the pipeline runs segmentation (face-skin mask for the blush)
    name = "skin"
    order = 1
    def apply(self, frame, ctx, settings):
        return frame

tiles = []
for name in ["frame_720p", "frame_dark", "frame_tilt", "frame_far"]:
    img = cv2.imread(str(S / 'testimgs' / f'{name}.png'))
    pipe = MakeupPipeline(MakeupSettings(), model_dir=S / 'models', effects=[MaskProbe(), BlushEffect(), LipEffect()])
    for i in range(8):
        res = pipe.process(img, 33 * (i + 1))
    print(name, {k: round(v, 2) for k, v in res.timings_ms.items()})
    pipe.close()
    box = regions.roi_around(res.landmarks[regions.FACE_OVAL], img.shape, pad=0.05)
    pair = [cv2.resize(x[box.slices], (420, int(420 * box.h / box.w)), interpolation=cv2.INTER_CUBIC) for x in (img, res.output)]
    tiles.append(np.hstack(pair))
h = min(t.shape[0] for t in tiles)
grid = np.vstack([np.hstack([tiles[0][:h], tiles[1][:h]]), np.hstack([tiles[2][:h], tiles[3][:h]])])
cv2.imwrite(str(S / 'renders/lips_blush/combined_default.png'), grid)

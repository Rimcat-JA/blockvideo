"""Compare blush blend candidates on the cheeks (default geometry, stronger opacity)."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import blend, regions
from makeup_cam.config import BlushSettings
from makeup_cam.effects import blush as B
from makeup_cam.effects.base import FrameContext
S = '/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/'

def tint(base, color, w):
    c = np.asarray(color, np.float32)
    f = c / float(blend.luminance(c.reshape(1, 1, 3))[0, 0])
    return blend.mix(base, np.clip(base * f, 0, 1), w)

modes = {
    "soft_light": lambda b, c, w: blend.mix(b, blend.blend_soft_light(b, c), w),
    "multiply": lambda b, c, w: blend.mix(b, blend.blend_multiply(b, c), w),
    "mix35": B._flush,
    "color": lambda b, c, w: blend.mix(b, blend.blend_color(b, c), w),
    "tint": tint,
}
rows = []
for name in ["frame_720p", "frame_dark"]:
    img = cv2.imread(S + 'testimgs/' + name + '.png').astype(np.float32) / 255
    lm = np.load(S + 'tmp_lips_blush/lm_' + name + '.npy')
    ctx = FrameContext(720, 1280, landmarks=lm)
    box = regions.roi_around(lm[regions.FACE_OVAL], img.shape, pad=0.02)
    tiles = [img]
    for mname, fn in modes.items():
        B._flush = fn
        tiles.append(B.BlushEffect().apply(img.copy(), ctx, BlushSettings(opacity=0.4)))
    zs = []
    for t, label in zip(tiles, ["before"] + list(modes)):
        c = (np.clip(t[box.slices], 0, 1) * 255).astype(np.uint8)
        c = cv2.resize(c, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        cv2.putText(c, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        zs.append(c)
    rows.append(np.hstack(zs[:3])); rows.append(np.hstack(zs[3:]))
cv2.imwrite(S + 'renders/lips_blush/blush_modes.png', np.vstack(rows))

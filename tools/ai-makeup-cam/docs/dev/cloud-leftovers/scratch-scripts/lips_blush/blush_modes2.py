"""Compare blush blend candidates at the default opacity, closer crops."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import blend, regions
from makeup_cam.config import BlushSettings
from makeup_cam.effects import blush as B
from makeup_cam.effects.base import FrameContext
S = '/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/'

def tint_f(c):
    c = np.asarray(c, np.float32)
    return c / float(blend.luminance(c.reshape(1, 1, 3))[0, 0])

def tint(base, color, w):
    return blend.mix(base, np.clip(base * tint_f(color), 0, 1), w)

def tint_mul(base, color, w):
    t = np.clip(base * tint_f(color), 0, 1)
    return blend.mix(base, blend.mix(t, blend.blend_multiply(base, color), 0.5), w)

modes = {
    "multiply": lambda b, c, w: blend.mix(b, blend.blend_multiply(b, c), w),
    "tint": tint,
    "tint+mul": tint_mul,
}
op = float(sys.argv[1]) if len(sys.argv) > 1 else 0.22
rows = []
for name in ["frame_720p", "frame_dark", "frame_near"]:
    path = S + ('tmp_lips_blush/' if name == 'frame_near' else 'testimgs/') + name + '.png'
    img = cv2.imread(path).astype(np.float32) / 255
    lm = np.load(S + 'tmp_lips_blush/lm_' + name + '.npy')
    ctx = FrameContext(720, 1280, landmarks=lm)
    sc = regions.face_scale(lm)
    pts = np.vstack([lm[regions.RIGHT_EYE], lm[[234, 61, 1]]])
    box = regions.roi_around(pts, img.shape, pad=0.02)
    tiles = [img]
    for mname, fn in modes.items():
        B._flush = fn
        tiles.append(B.BlushEffect().apply(img.copy(), ctx, BlushSettings(opacity=op)))
    zs = []
    for t, label in zip(tiles, ["before"] + list(modes)):
        c = (np.clip(t[box.slices], 0, 1) * 255).astype(np.uint8)
        c = cv2.resize(c, None, fx=420 / c.shape[1], fy=420 / c.shape[1], interpolation=cv2.INTER_CUBIC)
        cv2.putText(c, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        zs.append(c)
    h = min(z.shape[0] for z in zs)
    rows.append(np.hstack([z[:h] for z in zs]))
w = min(r.shape[1] for r in rows)
cv2.imwrite(S + f'renders/lips_blush/blush_modes_{op}.png', np.vstack([r[:, :w] for r in rows]))

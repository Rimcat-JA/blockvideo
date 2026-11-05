"""Gloss on matte lips: make the lips matte first, then apply satin/gloss, zoomed."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import LipSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects.lips import LipEffect
S = '/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/'
rows = []
for name in ["frame_720p", "frame_dark"]:
    img = cv2.imread(S + 'testimgs/' + name + '.png').astype(np.float32) / 255
    lm = np.load(S + 'tmp_lips_blush/lm_' + name + '.npy')
    ctx = FrameContext(720, 1280, landmarks=lm)
    eff = LipEffect()
    matte = eff.apply(img.copy(), ctx, LipSettings(color="#C9858A", opacity=1.0, finish="matte", feather=0.3))
    matte = eff.apply(matte, ctx, LipSettings(color="#C9858A", opacity=1.0, finish="matte", feather=0.3))
    outs = [matte]
    for finish, gloss, op in (("satin", 0.3, 0.45), ("satin", 1.0, 0.45), ("gloss", 0.3, 0.45), ("gloss", 1.0, 0.45), ("gloss", 1.0, 0.1)):
        outs.append(eff.apply(matte.copy(), ctx, LipSettings(finish=finish, gloss=gloss, opacity=op)))
    box = regions.roi_around(lm[regions.LIPS_OUTER], img.shape, pad=0.25)
    tiles = []
    for o in outs:
        c = (np.clip(o[box.slices], 0, 1) * 255).astype(np.uint8)
        tiles.append(cv2.resize(c, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC))
    rows.append(np.hstack(tiles[:3]))
    rows.append(np.hstack(tiles[3:]))
w = min(r.shape[1] for r in rows)
cv2.imwrite(S + 'renders/lips_blush/lips_gloss_on_matte.png', np.vstack([r[:, :w] for r in rows]))

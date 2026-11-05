import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.effects import skin as K
from makeup_cam.effects.base import FrameContext
S = "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/"
rows = []
for name in sys.argv[1].split(","):
    img = cv2.imread(S + f"testimgs/{name}.png").astype(np.float32) / 255
    lm = np.load(S + f"tmp_skin/lm_{name}.npy")
    ws = K._workspace(lm, img.shape)
    small = K._resize(img[ws.roi.slices], ws.small_size, cv2.INTER_AREA)
    masks = K._skin_masks(ws, FrameContext(720, 1280, landmarks=lm))
    level = K._skin_level(small, masks.skin)
    spots, clean = K._blemishes(small, ws.scale, level)
    w = spots * masks.skin
    f = 480 / small.shape[0]
    up = lambda x: cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    vis = up(small) * (1 - 0.7 * up(w)[..., None]) + np.array([0, 0, 1], np.float32) * 0.7 * up(w)[..., None]
    fixed = small + (clean - small) * w[..., None]
    rows.append(np.hstack([up(small), vis, up(fixed)]))
    print(name, "scale", ws.scale, "max w", w.max())
W = min(r.shape[1] for r in rows); cv2.imwrite(S + "tmp_skin/spot_debug.png", (np.vstack([r[:, :W] for r in rows]) * 255).clip(0, 255).astype(np.uint8))

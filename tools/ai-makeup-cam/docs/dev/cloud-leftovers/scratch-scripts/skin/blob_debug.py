import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.effects import skin
from makeup_cam.effects.base import FrameContext
S = "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/"
name = sys.argv[1] if len(sys.argv) > 1 else "frame_720p"
img = cv2.imread(S + f"testimgs/{name}.png").astype(np.float32) / 255
lm = np.load(S + f"tmp_skin/lm_{name}.npy")
ws = skin._workspace(lm, img.shape)
patch = img[ws.roi.slices]
small = skin._resize(patch, ws.small_size, cv2.INTER_AREA)
ctx = FrameContext(720, 1280, landmarks=lm)
masks = skin._skin_masks(ws, ctx)
level = skin._skin_level(small, masks.skin)
sig = float(sys.argv[2]) if len(sys.argv) > 2 else 0.035
g = cv2.GaussianBlur(small[..., 1], (0, 0), sig * ws.scale)
lxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3); lyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3); lxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
lmin = (lxx + lyy) / 2 - np.sqrt(((lxx - lyy) / 2) ** 2 + lxy ** 2)
r = lmin * (sig * ws.scale) ** 2 / 4 / level
print("scale", ws.scale, "level", level, "r pct", np.percentile(r[masks.skin > 0.5], [50, 90, 95, 99, 99.9]))
vis = np.clip(r / 0.03, 0, 1) * masks.skin
f = 8
a = cv2.resize((small * 255).astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
b = cv2.cvtColor(cv2.resize((vis * 255).astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
cv2.imwrite(S + "tmp_skin/blob_debug.png", np.hstack([a, b]))

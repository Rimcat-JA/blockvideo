import sys, time
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import BlushSettings, hex_to_bgr
from makeup_cam.effects import blush as B
S='/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/'
name = sys.argv[1]
path = S + ('tmp_lips_blush/' if name == 'frame_near' else 'testimgs/') + name + '.png'
img = cv2.imread(path).astype(np.float32) / 255
lm = np.load(S+'tmp_lips_blush/lm_'+name+'.npy')
st = BlushSettings()
def timeit(label, fn, n=50):
    fn(); t = time.perf_counter()
    for _ in range(n): r = fn()
    print(f"{label:12s} {(time.perf_counter()-t)/n*1000:.3f} ms")
    return r
scale = timeit("scale", lambda: regions.face_scale(lm))
shape = B._SHAPES[st.shape]
fp = scale * (B._SOFT_BASE + B._SOFT_RANGE * st.softness)
ells = timeit("ellipses", lambda: [B._cheek_ellipse(lm, s, shape, st, scale) for s in regions.SIDES])
roi = timeit("roi", lambda: B._blush_roi(ells, img.shape, fp))
print(roi, roi.w * roi.h)
grid = B._Grid(roi, max(1, int(scale // B._GRID_SCALE)))
print("factor", grid.factor, grid.area)
mask = timeit("mask", lambda: B._blush_mask(grid, ells, fp))
allowed = timeit("allowed", lambda: B._allowed_area(grid, lm, scale, fp, None, mask))
up = timeit("upsample", lambda: grid.upsample(mask * allowed) * np.float32(0.22))
base = img[roi.slices]
timeit("flush", lambda: B._flush(base, hex_to_bgr(st.color), up))
def wb():
    f = img.copy()
    f[roi.slices] = base
timeit("writeback+copy", wb)
timeit("copy", lambda: img.copy())

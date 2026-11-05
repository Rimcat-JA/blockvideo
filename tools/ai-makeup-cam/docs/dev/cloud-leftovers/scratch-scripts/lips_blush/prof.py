"""Time one effect on a cached frame: median and mean over 30 runs (+ optional profile)."""
import sys, time, cProfile, pstats
import numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.config import MakeupSettings
from makeup_cam.effects.base import FrameContext
S='/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/'
which, name = sys.argv[1], sys.argv[2]
path = S + ('tmp_lips_blush/' if name == 'frame_near' else 'testimgs/') + name + '.png'
img = cv2.imread(path).astype(np.float32) / 255
lm = np.load(S+'tmp_lips_blush/lm_'+name+'.npy')
if which == 'lips':
    from makeup_cam.effects.lips import LipEffect as C
else:
    from makeup_cam.effects.blush import BlushEffect as C
eff = C()
settings = getattr(MakeupSettings(), which)
profile = False
for arg in sys.argv[3:]:
    if arg == 'profile':
        profile = True
        continue
    k, v = arg.split('=')
    cur = getattr(settings, k)
    setattr(settings, k, type(cur)(v))
ctx = FrameContext(720, 1280, landmarks=lm)
for _ in range(5):
    eff.apply(img.copy(), ctx, settings)
times = []
for _ in range(30):
    f = img.copy()
    t = time.perf_counter()
    eff.apply(f, ctx, settings)
    times.append((time.perf_counter() - t) * 1000)
print(f"{which} {name} {' '.join(sys.argv[3:])}: median {np.median(times):.2f} ms, mean {np.mean(times):.2f} ms, min {np.min(times):.2f} ms")
if profile:
    pr = cProfile.Profile(); pr.enable()
    for _ in range(10):
        eff.apply(img.copy(), ctx, settings)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(15)

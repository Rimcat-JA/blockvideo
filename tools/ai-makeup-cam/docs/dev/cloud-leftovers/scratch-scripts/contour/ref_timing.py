import time
from harness import *
from makeup_cam.config import MakeupSettings, ContourSettings, TearbagSettings
from makeup_cam.effects.blush import BlushEffect
from makeup_cam.effects.lips import LipEffect
from makeup_cam.effects.eyes import EyeshadowEffect, EyelinerEffect
from makeup_cam.effects.brows import BrowEffect
frames = [still("frame_near")] * 4
for f, ctx in contexts(frames):
    pass
img = f.astype(np.float32) / 255
ms = MakeupSettings()
for name, eff, st in [("skin", SkinEffect(), ms.skin), ("blush", BlushEffect(), ms.blush), ("lips", LipEffect(), ms.lips), ("eyeshadow", EyeshadowEffect(), ms.eyeshadow), ("eyeliner", EyelinerEffect(), ms.eyeliner), ("brows", BrowEffect(), ms.brows), ("contour", C.ContourEffect(), ContourSettings()), ("tearbag", C.TearbagEffect(), TearbagSettings())]:
    ts = []
    for _ in range(80):
        x = img.copy(); t = time.perf_counter(); eff.apply(x, ctx, st); ts.append((time.perf_counter() - t) * 1000)
    print(f"{name:10s} median {np.median(ts):.2f}  min {np.min(ts):.2f}")

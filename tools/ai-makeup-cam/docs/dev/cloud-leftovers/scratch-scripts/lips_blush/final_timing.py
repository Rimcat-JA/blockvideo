"""Final timings (30 runs, perf_counter) on the face_frame fixture and scaled-landmark variants."""
import sys
import time
import numpy as np
import cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests")
from conftest import FACE_LANDMARKS_720P, build_face_frame_720p
from makeup_cam import regions
from makeup_cam.config import BlushSettings, LipSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects.blush import BlushEffect
from makeup_cam.effects.lips import LipEffect

base_frame = build_face_frame_720p().astype(np.float32) / 255
base_lm = np.load(FACE_LANDMARKS_720P).astype(np.float32)

def variant(zoom: float):
    """Frame + landmarks zoomed around the face center (zoom 1 = fixture)."""
    if zoom == 1.0:
        return base_frame, base_lm
    c = base_lm[regions.NOSE_TIP]
    m = np.float32([[zoom, 0, 640 - zoom * c[0]], [0, zoom, 360 - zoom * c[1]]])
    frame = cv2.warpAffine(base_frame, m, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lm = base_lm @ m[:, :2].T + m[:, 2]
    return frame, lm.astype(np.float32)

cases = [
    ("lips default", LipEffect, LipSettings()),
    ("lips gloss max", LipEffect, LipSettings(opacity=1.0, finish="gloss", gloss=1.0, feather=1.0, outline=0.2)),
    ("blush default", BlushEffect, BlushSettings()),
    ("blush max", BlushEffect, BlushSettings(opacity=1.0, size=1.6, softness=1.0, shape="lifted")),
]
for zoom in (0.17, 0.5, 1.0, 2.0, 3.5):
    frame, lm = variant(zoom)
    ctx = FrameContext(720, 1280, landmarks=lm)
    for label, cls, settings in cases:
        eff = cls()
        frames = [frame.copy() for _ in range(35)]
        times = []
        for f in frames:
            t = time.perf_counter()
            out = eff.apply(f, ctx, settings)
            times.append((time.perf_counter() - t) * 1000)
        assert out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
        times = times[5:]
        print(f"scale {regions.face_scale(lm):6.1f}  {label:15s} median {np.median(times):5.2f} ms  mean {np.mean(times):5.2f} ms")

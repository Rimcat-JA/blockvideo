import sys; sys.path.insert(0, '/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests'); sys.path.insert(0, '/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam')
import numpy as np
from dataclasses import replace
from conftest import build_face_frame_720p
from makeup_cam.config import SkinSettings
from makeup_cam.effects.skin import SkinEffect
from makeup_cam.effects.base import FrameContext
lm = np.load('/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy').astype(np.float32)
f = build_face_frame_720p().astype(np.float32) / 255
zero = dict(smoothing=0, blemish=0, redness=0, under_eye=0, tone_strength=0, brightness=0)
cases = {'brightness .25': dict(brightness=0.25), 'tone .8': dict(tone_strength=0.8), 'under_eye 1': dict(under_eye=1.0),
         'redness 1': dict(redness=1.0), 'default': None}
for name, kw in cases.items():
    st = SkinSettings() if kw is None else replace(SkinSettings(), **{**zero, **kw})
    worst = 0
    for dx, dy in [(0.5, 0), (0, 0.5), (0.5, 0.5), (-0.5, 0.3)]:
        a = SkinEffect().apply(f.copy(), FrameContext(720, 1280, landmarks=lm), st)
        b = SkinEffect().apply(f.copy(), FrameContext(720, 1280, landmarks=lm + np.float32([dx, dy])), st)
        d = np.abs(b - a).max(axis=2); y, x = np.unravel_index(d.argmax(), d.shape)
        worst = max(worst, d.max())
    print(f'{name:15s} worst {worst:.4f} (last at {x},{y})')

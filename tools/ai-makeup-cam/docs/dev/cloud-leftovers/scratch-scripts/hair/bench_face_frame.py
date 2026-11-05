"""HairEffect timing on the conftest face_frame_720p (+ fixture landmarks, real segmentation mask), 30 runs."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from conftest import build_face_frame_720p  # noqa: E402
from makeup_cam.config import HairSettings  # noqa: E402
from makeup_cam.effects.base import FrameContext  # noqa: E402
from makeup_cam.effects.hair import HairEffect  # noqa: E402
from makeup_cam.segmentation import Segmenter  # noqa: E402

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
frame8 = build_face_frame_720p()
landmarks = np.load(ROOT / "tests" / "data" / "face_landmarks_720p.npy").astype(np.float32)
seg = Segmenter(S / "models" / "selfie_multiclass_256x256.tflite", interval=1, background=False)
for i in range(4):
    masks = seg.update(frame8, 33 * (i + 1), i + 1)
seg.close()
frame = frame8.astype(np.float32) / 255.0
ctx = FrameContext(720, 1280, landmarks=landmarks, hair_mask=masks.hair, face_skin_mask=masks.face_skin)
for label, settings in {
    "default": HairSettings(),
    "all-on": HairSettings(gloss=1.0, smoothness=1.0, color_strength=0.8, brightness=0.2, contrast=0.2),
}.items():
    effect = HairEffect()
    times = []
    for _ in range(30):
        img = frame.copy()
        t = time.perf_counter()
        effect.apply(img, ctx, settings)
        times.append((time.perf_counter() - t) * 1000)
    print(f"{label}: median {np.median(times):.2f} ms, min {min(times):.2f} ms, max {max(times):.2f} ms")

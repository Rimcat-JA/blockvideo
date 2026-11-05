"""Time HairEffect.apply on cached landmarks + masks (no models), and profile it.

usage: bench.py [frame_name] [--profile]
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from makeup_cam.config import HairSettings  # noqa: E402
from makeup_cam.effects.base import FrameContext  # noqa: E402
from makeup_cam.effects.hair import HairEffect  # noqa: E402

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")

SETTINGS = {
    "default": HairSettings(),
    "all": HairSettings(gloss=1.0, smoothness=1.0, color_strength=0.8, brightness=0.2, contrast=0.2),
}


def load(name: str):
    from render import load as load_frame

    frame = load_frame(name)
    data = np.load(S / "tmp_hair" / f"cache_{name}.npz")
    return frame, data["landmarks"], data["hair"], data["face_skin"]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "frame_720p"
    frame, lm, hair, face = load(name)
    base = frame.astype(np.float32) / 255.0
    effect = HairEffect()
    for label, settings in SETTINGS.items():
        ctx = FrameContext(720, 1280, landmarks=lm, hair_mask=hair, face_skin_mask=face)
        times = []
        for _ in range(30):
            img = base.copy()
            t = time.perf_counter()
            effect.apply(img, ctx, settings)
            times.append((time.perf_counter() - t) * 1000)
        print(f"{name} {label}: median {np.median(times):.2f} ms, min {min(times):.2f} ms")
    if "--profile" in sys.argv:
        ctx = FrameContext(720, 1280, landmarks=lm, hair_mask=hair, face_skin_mask=face)
        prof = cProfile.Profile()
        prof.enable()
        for _ in range(30):
            effect.apply(base.copy(), ctx, SETTINGS["all"])
        prof.disable()
        pstats.Stats(prof).sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    main()

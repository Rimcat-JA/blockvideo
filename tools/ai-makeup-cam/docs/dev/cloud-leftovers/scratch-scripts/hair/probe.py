"""Dump landmarks + segmentation masks of the test frames (for fast iteration) and mask overlays."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from makeup_cam.segmentation import Segmenter  # noqa: E402
from makeup_cam.tracking import FaceTracker  # noqa: E402
from render import load  # noqa: E402

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
HERE = S / "tmp_hair"
OUT = S / "renders" / "hair"
NAMES = ["frame_720p", "frame_dark", "frame_tilt", "frame_far", "close", "actor_1072", "actor_1002"]


def main() -> None:
    for name in NAMES:
        frame = load(name)
        tracker = FaceTracker(S / "models" / "face_landmarker.task", 0.0)
        seg = Segmenter(S / "models" / "selfie_multiclass_256x256.tflite", interval=1, background=False)
        lm = None
        masks = None
        for i in range(8):
            lm, _ = tracker.detect(frame, 33 * (i + 1))
            masks = seg.update(frame, 33 * (i + 1), i + 1)
        tracker.close()
        seg.close()
        np.savez(HERE / f"cache_{name}.npz", landmarks=lm, hair=masks.hair, face_skin=masks.face_skin)
        over = frame.copy().astype(np.float32)
        over[..., 2] = over[..., 2] * (1 - masks.hair) + 255 * masks.hair
        over[..., 1] = over[..., 1] * (1 - masks.face_skin) + 255 * masks.face_skin
        cv2.imwrite(str(OUT / f"mask_{name}.png"), over.astype(np.uint8))
        print(name, "landmarks" if lm is not None else "NO FACE", float(masks.hair.max()))


if __name__ == "__main__":
    main()

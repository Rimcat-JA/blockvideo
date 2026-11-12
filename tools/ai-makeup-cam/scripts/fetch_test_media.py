"""Download the development test media (never committed to the repository).

Creates, under DEST (default: ``.test-media`` next to this project, git-ignored):

- ``testimgs/``: 1280x720 webcam-like frames built from the public-domain NASA
  portrait bundled with scikit-image (``frame_720p``, ``frame_dark``,
  ``frame_tilt``, ``frame_far``) - the same frames the tests' fixtures use.
- ``testvids/``: short talking-face clips of 11 women from CREMA-D
  (Open Database License / Database Contents License,
  https://github.com/CheyneyComputerScience/CREMA-D), 4 clips each,
  plus ``demographics.csv``.

Usage::

    python scripts/fetch_test_media.py [DEST]
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CREMA_D = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master"
CREMA_D_RAW = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master"
# Women aged 20-33 of different skin tones (see demographics.csv).
ACTORS = ("1002", "1012", "1013", "1029", "1046", "1060", "1072", "1073", "1074", "1076", "1082")
CLIPS = ("IEO_HAP_HI", "DFA_NEU_XX", "ITS_HAP_XX", "TIE_SAD_XX")


def build_test_frames(dest: Path) -> None:
    from skimage import data  # requirements-dev.txt

    dest.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    canvas = np.full((720, 1280, 3), (70, 75, 80), np.uint8)
    canvas[:, 280:1000] = cv2.resize(bgr, (720, 720), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dest / "frame_720p.png"), canvas)
    dark = np.clip(((canvas / 255.0) ** 1.6) * 255 * np.array([0.85, 0.95, 1.05]), 0, 255)
    cv2.imwrite(str(dest / "frame_dark.png"), dark.astype(np.uint8))
    tilt = cv2.getRotationMatrix2D((640, 300), 14, 1.0)
    cv2.imwrite(str(dest / "frame_tilt.png"), cv2.warpAffine(canvas, tilt, (1280, 720), borderMode=cv2.BORDER_REFLECT))
    far = np.full((720, 1280, 3), (70, 75, 80), np.uint8)
    far[200:600, 440:840] = cv2.resize(bgr, (400, 400))
    cv2.imwrite(str(dest / "frame_far.png"), far)
    cv2.imwrite(str(dest / "astronaut_512.png"), bgr)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def fetch_clips(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    download(f"{CREMA_D_RAW}/VideoDemographics.csv", dest / "demographics.csv")
    for actor in ACTORS:
        for clip in CLIPS:
            name = f"{actor}_{clip}.flv"
            download(f"{CREMA_D}/VideoFlash/{name}", dest / name)
            print(name)


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / ".test-media"
    build_test_frames(root / "testimgs")
    fetch_clips(root / "testvids")
    print(f"done: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

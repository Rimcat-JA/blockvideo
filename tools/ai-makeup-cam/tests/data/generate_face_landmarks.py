"""Regenerate ``face_landmarks_720p.npy`` with the real Face Landmarker.

Usage (from the project root)::

    AI_MAKEUP_CAM_MODELS_DIR=/path/to/models python tests/data/generate_face_landmarks.py [--out FILE]

The frame is the ``face_frame_720p`` test fixture. It is fed repeatedly in
VIDEO mode and the last result is kept: the first detection on a
non-square frame is vertically off by up to ~25 px, and tracking on the
same frame converges to sub-pixel stability within about ten frames, the
steady state the app sees.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS_DIR.parent), str(TESTS_DIR)]

from conftest import FACE_LANDMARKS_720P, build_face_frame_720p  # noqa: E402

CONVERGENCE_FRAMES = 15


def detect_converged_landmarks(frame_bgr: np.ndarray, model_path: Path) -> np.ndarray:
    """(478, 2) float32 pixel landmarks after tracking the still frame to convergence."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    height, width = frame_bgr.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        for frame_index in range(CONVERGENCE_FRAMES):
            result = landmarker.detect_for_video(image, 33 * (frame_index + 1))
    finally:
        landmarker.close()
    if not result.face_landmarks:
        raise RuntimeError("テスト画像から顔が検出できませんでした")
    return np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="テスト用の顔ランドマーク (face_landmarks_720p.npy) を実モデルで再生成します")
    default_model = Path(os.environ.get("AI_MAKEUP_CAM_MODELS_DIR", ".")) / "face_landmarker.task"
    parser.add_argument("--model", type=Path, default=default_model, help="face_landmarker.task のパス")
    parser.add_argument("--out", type=Path, default=FACE_LANDMARKS_720P, help="出力する .npy ファイル")
    args = parser.parse_args()
    landmarks = detect_converged_landmarks(build_face_frame_720p(), args.model)
    np.save(args.out, landmarks)
    print(f"{args.out} を書き出しました {landmarks.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

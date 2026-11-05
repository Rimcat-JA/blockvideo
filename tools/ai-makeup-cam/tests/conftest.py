"""Shared pytest fixtures.

- ``face_frame_720p``: a 1280x720 uint8 BGR "webcam" frame with a real face
  (the public-domain scikit-image astronaut portrait); skipped without
  scikit-image.
- ``face_landmarks_720p``: the (478, 2) float32 Face Landmarker points of that
  frame, precomputed with the real model (``tests/data``), so geometry tests
  run without MediaPipe.
- ``models_dir``: the model directory from ``AI_MAKEUP_CAM_MODELS_DIR``;
  skipped when unset or missing.
- ``fake_ctx``: a factory building a :class:`FrameContext` for effect tests.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from makeup_cam.effects.base import FrameContext

DATA_DIR = Path(__file__).resolve().parent / "data"
FACE_LANDMARKS_720P = DATA_DIR / "face_landmarks_720p.npy"
_BACKGROUND_BGR = (70, 75, 80)


@functools.cache
def _build_face_frame_720p() -> np.ndarray:
    import cv2
    from skimage import data

    canvas = np.full((720, 1280, 3), _BACKGROUND_BGR, np.uint8)
    face = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (720, 720), interpolation=cv2.INTER_CUBIC)
    canvas[:, 280:1000] = face
    return canvas


def build_face_frame_720p() -> np.ndarray:
    """The astronaut portrait scaled to 720x720, centered on a flat gray 1280x720 frame."""
    return _build_face_frame_720p().copy()


@pytest.fixture
def face_frame_720p() -> np.ndarray:
    pytest.importorskip("skimage")
    return build_face_frame_720p()


@pytest.fixture
def face_landmarks_720p() -> np.ndarray:
    return np.load(FACE_LANDMARKS_720P).astype(np.float32)


@pytest.fixture
def models_dir() -> Path:
    value = os.environ.get("AI_MAKEUP_CAM_MODELS_DIR")
    if not value or not Path(value).is_dir():
        pytest.skip("AI_MAKEUP_CAM_MODELS_DIR is not set to an existing directory")
    return Path(value)


MakeCtx = Callable[..., FrameContext]


@pytest.fixture
def fake_ctx() -> MakeCtx:
    """``make_ctx(frame, landmarks, hair_mask=None, face_skin_mask=None,
    intensity=1.0, body_skin_mask=None) -> FrameContext`` (size from ``frame``)."""

    def make_ctx(
        frame: np.ndarray,
        landmarks: np.ndarray | None,
        hair_mask: np.ndarray | None = None,
        face_skin_mask: np.ndarray | None = None,
        intensity: float = 1.0,
        body_skin_mask: np.ndarray | None = None,
    ) -> FrameContext:
        height, width = frame.shape[:2]
        return FrameContext(
            height=height,
            width=width,
            landmarks=None if landmarks is None else np.asarray(landmarks, dtype=np.float32),
            hair_mask=hair_mask,
            face_skin_mask=face_skin_mask,
            body_skin_mask=body_skin_mask,
            intensity=intensity,
        )

    return make_ctx

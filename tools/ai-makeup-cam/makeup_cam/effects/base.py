"""The contract every makeup effect implements.

The pipeline converts each camera frame to float32 BGR in [0, 1], builds one
``FrameContext`` (landmarks + segmentation masks + a per-frame cache), then
calls each enabled effect in ``order``. An effect edits the frame (in place
or by returning a new array) and must return a float32 BGR frame of the same
shape, values in [0, 1].

To add a new effect (for example a future hairstyle effect):
  1. add a settings section to ``config.py`` (and to ``SECTIONS``),
  2. write ``effects/<name>.py`` with a subclass decorated by
     ``@register_effect`` whose ``name`` equals the section name,
  3. import the module in ``effects/__init__.py``.
The GUI tab and preset support then come for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


@dataclass
class FrameContext:
    """Everything an effect may read about the current frame.

    Coordinates are full-frame pixel coordinates (x right, y down) of the
    *unmirrored* camera frame.

    The reshape effect (order 5) warps the frame, and then replaces ``landmarks``
    with the warped positions and ``hair_mask`` / ``face_skin_mask`` /
    ``body_skin_mask`` with copies warped by the same map, so effects that run
    after it stay aligned with the reshaped face. ``landmarks_z`` is left as it
    is. Treat these arrays as read-only; the originals stay untouched.
    """

    height: int
    width: int
    # (478, 2) float32 pixel coords of MediaPipe Face Landmarker points
    # (468 mesh points + 10 iris points), already temporally smoothed.
    # None when no face is found in this frame.
    landmarks: np.ndarray | None = None
    # (478,) float32 relative depth (MediaPipe z * width); smaller = closer.
    landmarks_z: np.ndarray | None = None
    # Segmentation confidence masks, float32 HxW in [0, 1] at full frame size,
    # temporally smoothed. None when segmentation is unavailable/not yet run.
    hair_mask: np.ndarray | None = None
    face_skin_mask: np.ndarray | None = None
    body_skin_mask: np.ndarray | None = None
    # Empty-room background plate (float32 BGR in [0, 1], same size as the
    # frame) captured with the GUI's 「背景を撮影」 button. None when the user
    # has not captured one or the camera/resolution changed since.
    background: np.ndarray | None = None
    # Master intensity (GeneralSettings.intensity); effects multiply their
    # opacity/strength by this.
    intensity: float = 1.0
    # Scratch space shared by effects within one frame (e.g. a face-oval mask
    # computed by one effect and reused by another). Keys are free-form str.
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def has_face(self) -> bool:
        return self.landmarks is not None


class Effect(ABC):
    """Base class for one makeup step.

    Subclasses set ``name`` (== settings section name), ``order`` (lower runs
    first), and what they need. ``apply`` is only called when the section is
    enabled and the requirements are met (a face for ``needs_face``, a hair
    mask for ``needs_hair_mask``).
    """

    name: ClassVar[str]
    order: ClassVar[int] = 100
    needs_face: ClassVar[bool] = True
    needs_hair_mask: ClassVar[bool] = False

    @abstractmethod
    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        """Return the frame with this effect applied.

        ``frame``: float32 HxWx3 BGR in [0, 1] (may be modified in place).
        ``settings``: this effect's section dataclass (e.g. ``LipSettings``).
        """

    def reset(self) -> None:  # pragma: no cover - default no-op
        """Drop any temporal state (called when the face is lost / camera changes)."""


_REGISTRY: dict[str, type[Effect]] = {}


def register_effect(cls: type[Effect]) -> type[Effect]:
    if not getattr(cls, "name", None):
        raise TypeError(f"{cls.__name__} must define a non-empty 'name'")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate effect name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def registered_effects() -> list[type[Effect]]:
    """All registered effect classes, sorted by ``order``."""
    return sorted(_REGISTRY.values(), key=lambda c: (c.order, c.name))

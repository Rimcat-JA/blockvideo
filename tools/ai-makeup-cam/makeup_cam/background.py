"""Background plate: a picture of the empty room behind the user.

The user steps out of view, the camera records a few frames and their median
becomes the plate (the median also removes noise and passing shadows). Hair
effects use it to fill what they remove, see ``effects/hairshape.py``.

The plate is a personal image of the user's room. It is only ever written to
``user_data_dir()/background_<W>x<H>.png`` and nowhere else, and
:func:`clear_plates` removes every copy.

A plate is valid for exactly one resolution, so each resolution has its own
file and a plate of another size is never used.

Typical use (the GUI's 「背景を撮影」 button, through the runner):

1. create a :class:`PlateCollector` and show :attr:`PlateCollector.message`;
2. call :meth:`PlateCollector.feed` with every raw camera frame until
   :attr:`PlateCollector.done`;
3. hand :attr:`PlateCollector.frames` to ``MakeupPipeline.capture_background``
   on a worker thread (never on a camera thread: it takes up to about a
   second), and show the ``message`` of the returned :class:`PlateResult`.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import user_data_dir
from .segmentation import BODY_SKIN, CLOTHES, FACE_SKIN, HAIR, DEFAULT_INPUT_LONG_SIDE, MediaPipeSegmenter, model_input

logger = logging.getLogger(__name__)

COUNTDOWN_SECONDS = 3.0
CAPTURE_FRAMES = 15  # frames in the median
CAPTURE_SPACING_S = 0.066  # kept frames are at least this far apart (about 1 s in all)
CAPTURE_TIMEOUT_S = 5.0  # after the countdown: give up if the camera delivers too few frames
MIN_FRAMES = 5  # fewer frames than this are not a reliable median
MAX_PERSON_FRACTION = 0.03  # share of the frame a person or hair may cover
MIN_MEAN_LUMINANCE = 0.03  # a darker picture is a covered lens or a switched-off light

# ---- messages shown to the user -------------------------------------------
COUNTDOWN_MESSAGE = "{seconds}秒後に背景を撮影します。カメラの前から離れてください"
CAPTURING_MESSAGE = "背景を撮影しています。そのままお待ちください"
SUCCESS_MESSAGE = "背景を撮影しました。アホ毛消しと髪型・顔の形の補正に使います"
SUCCESS_NOT_SAVED_MESSAGE = "背景を撮影しましたが、保存できませんでした。アプリを閉じるまでの間だけ使います"
PERSON_VISIBLE_MESSAGE = (
    "人や髪が映っています（画面の約{percent:.0f}%）。カメラの前から完全に離れて、もう一度撮影してください"
)
TOO_FEW_FRAMES_MESSAGE = "カメラの映像を十分に受け取れませんでした。カメラを確認して、もう一度撮影してください"
TOO_DARK_MESSAGE = "映像が暗すぎます。部屋を明るくして、レンズがふさがれていないか確認してから、もう一度撮影してください"
CHECK_FAILED_MESSAGE = "人が映っていないかの確認に失敗したため、背景を保存しませんでした。もう一度撮影してください"
CAPTURE_CANCELLED_MESSAGE = "撮影の途中で背景を削除したため、この撮影は保存も使用もしませんでした。必要なら、もう一度撮影してください"
CLEARED_MESSAGE = "撮影した背景を削除しました"
CLEAR_FAILED_MESSAGE = "撮影した背景を削除できませんでした。次のフォルダから手動で削除してください: {folder}"

_PLATE_NAME = "background_{width}x{height}.png"
_PLATE_FILE_RE = re.compile(r"^background_\d+x\d+\.png(\.tmp)?$")
# The model's classes that mean "a person": not its "others" class, which also
# fires on chairs and other furniture of an empty room.
_PERSON_CLASSES = (HAIR, BODY_SKIN, FACE_SKIN, CLOTHES)
_PERSON_CONFIDENCE = 0.5


@dataclass(frozen=True)
class PlateResult:
    """Outcome of capturing or clearing; ``message`` is ready to show to the user."""

    ok: bool
    message: str
    plate: np.ndarray | None = None  # uint8 BGR when a capture succeeded
    person_fraction: float = 0.0


class PersonProbe(Protocol):
    """Measures how much of a frame shows a person (fakes implement this in tests)."""

    def __call__(self, frame_bgr: np.ndarray) -> float:
        """Fraction of the frame (0..1) covered by a person or hair."""


class MediaPipePersonProbe:
    """:class:`PersonProbe` running its own copy of the selfie segmentation model.

    Separate from the pipeline's segmenter, whose masks lack the clothes class
    and which is busy on the live video. Call :meth:`close` when done.
    """

    def __init__(self, model_path: str | Path, input_long_side: int = DEFAULT_INPUT_LONG_SIDE) -> None:
        self._model = MediaPipeSegmenter(model_path)
        self._long_side = input_long_side
        self._ts_ms = 0

    def __call__(self, frame_bgr: np.ndarray) -> float:
        self._ts_ms += 33  # VIDEO mode needs increasing timestamps
        masks = self._model.segment(model_input(frame_bgr, self._long_side), self._ts_ms)
        person = sum(masks[index] for index in _PERSON_CLASSES)
        return float(np.mean(person > _PERSON_CONFIDENCE))

    def close(self) -> None:
        self._model.close()


class PlateCollector:
    """The frames of one capture: waits out the countdown, then keeps spaced frames.

    Not thread-safe: use it from the one thread that feeds it. The first
    :meth:`feed` does not start the clock; the constructor does.
    """

    def __init__(
        self,
        countdown_s: float = COUNTDOWN_SECONDS,
        count: int = CAPTURE_FRAMES,
        spacing_s: float = CAPTURE_SPACING_S,
        timeout_s: float = CAPTURE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._count = count
        self._spacing_s = spacing_s
        self._start_at = clock() + countdown_s
        self._deadline = self._start_at + timeout_s
        self._next_at = self._start_at
        self.frames: list[np.ndarray] = []

    def feed(self, frame_bgr: np.ndarray) -> None:
        """Offer one raw camera frame (uint8 BGR, not mirrored, without makeup)."""
        now = self._clock()
        if not self.done and now >= self._next_at:
            self.frames.append(frame_bgr.copy())
            self._next_at = now + self._spacing_s

    @property
    def seconds_left(self) -> int:
        """Whole seconds of countdown left (0 while recording), for an overlay."""
        return max(0, math.ceil(self._start_at - self._clock()))

    @property
    def message(self) -> str:
        """The overlay text for this moment."""
        left = self.seconds_left
        return COUNTDOWN_MESSAGE.format(seconds=left) if left > 0 else CAPTURING_MESSAGE

    @property
    def done(self) -> bool:
        """True when enough frames are in, or the camera did not deliver them in time."""
        return len(self.frames) >= self._count or self._clock() >= self._deadline


def median_frame(frames: Sequence[np.ndarray]) -> np.ndarray:
    """The per-pixel, per-channel median of uint8 frames (lower median for an even count)."""
    stack = np.stack(frames)
    middle = (len(frames) - 1) // 2
    return np.partition(stack, middle, axis=0)[middle]


def capture(frames: Sequence[np.ndarray], probe: PersonProbe) -> PlateResult:
    """Build a plate from ``frames`` (uint8 BGR), or reject it with a Japanese message.

    Besides the median, the first, middle and last frames are checked: the
    median of a person walking through the picture looks empty, but the plate
    would still carry their shadow. Slow (up to about a second at 1280x720).
    """
    usable = [frame for frame in frames if _is_bgr(frame)]
    usable = [frame for frame in usable if frame.shape == usable[0].shape]
    if len(usable) < MIN_FRAMES:
        return PlateResult(False, TOO_FEW_FRAMES_MESSAGE)
    plate = median_frame(usable)
    if float(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).mean()) / 255.0 < MIN_MEAN_LUMINANCE:
        return PlateResult(False, TOO_DARK_MESSAGE)
    checks = [plate, usable[0], usable[len(usable) // 2], usable[-1]]
    try:
        fraction = max(float(probe(frame)) for frame in checks)
    except Exception:
        logger.exception("person check of the background capture failed")
        return PlateResult(False, CHECK_FAILED_MESSAGE)
    if not fraction <= MAX_PERSON_FRACTION:  # also rejects NaN
        return PlateResult(False, PERSON_VISIBLE_MESSAGE.format(percent=fraction * 100.0), person_fraction=fraction)
    return PlateResult(True, SUCCESS_MESSAGE, plate=plate, person_fraction=fraction)


def plate_path(width: int, height: int) -> Path:
    """Where the plate for one resolution lives."""
    return user_data_dir() / _PLATE_NAME.format(width=width, height=height)


def save_plate(plate: np.ndarray) -> Path:
    """Save a uint8 BGR plate for its resolution, atomically; returns the path.

    Raises ``OSError`` when it cannot be written, and then leaves no temporary
    file behind (it is a picture of the room). Encoding in memory avoids
    ``cv2.imwrite``, which fails on non-ASCII paths on Windows.
    """
    if not _is_bgr(plate):
        raise ValueError("plate must be a uint8 BGR array of shape (H, W, 3)")
    ok, encoded = cv2.imencode(".png", plate)
    if not ok:
        raise OSError("could not encode the background plate")
    path = plate_path(plate.shape[1], plate.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(encoded.tobytes())
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("cannot delete a temporary background plate: %s", exc)
        raise
    return path


def load_plate(width: int, height: int) -> np.ndarray | None:
    """The saved uint8 BGR plate of this resolution; None if missing, unreadable or of another size."""
    path = plate_path(width, height)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("cannot read the background plate: %s", exc)
        return None
    try:
        plate = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        plate = None
    if plate is None or plate.shape != (height, width, 3):
        logger.warning("ignoring an invalid background plate file: %s", path.name)
        return None
    return plate


def clear_plates() -> PlateResult:
    """Delete every saved plate, of every resolution.

    All of them, because 「背景を削除」 means the picture of the room is gone,
    not just the one for the current resolution.
    """
    folder = user_data_dir()
    failed = False
    try:
        names = [entry.name for entry in os.scandir(folder) if _PLATE_FILE_RE.match(entry.name)]
    except FileNotFoundError:
        names = []
    except OSError as exc:
        logger.warning("cannot list the background plates: %s", exc)
        return PlateResult(False, CLEAR_FAILED_MESSAGE.format(folder=folder))
    for name in names:
        try:
            (folder / name).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("cannot delete a background plate: %s", exc)
            failed = True
    if failed:
        return PlateResult(False, CLEAR_FAILED_MESSAGE.format(folder=folder))
    return PlateResult(True, CLEARED_MESSAGE)


def plate_to_float(plate: np.ndarray) -> np.ndarray:
    """uint8 BGR plate -> read-only float32 BGR in [0, 1], as effects receive it."""
    image = np.multiply(plate, np.float32(1.0 / 255.0), dtype=np.float32)
    image.setflags(write=False)
    return image


def _is_bgr(frame: np.ndarray) -> bool:
    return isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3

"""Hair / skin segmentation with MediaPipe's selfie multiclass model.

The model is slow compared with the frame budget (about 20 ms on a laptop,
about 100 ms in a small CI container), so ``Segmenter``

* feeds it a downscaled copy (long side ``input_long_side``),
* runs it at most every ``interval`` frames,
* by default runs it on a background thread (MediaPipe releases the GIL), so
  the camera loop never waits for it and simply uses the latest finished
  masks, which are at most one model run old,
* smooths the class probabilities over time (EMA) at model resolution and
  upsamples them to the full frame once per run.

Everything of a run but its start (downscaling, model, smoothing, upsampling)
happens on that thread: about 2 ms of a frame's budget that the camera loop does
not spend. The thread holds on to the frame until then, so callers must not write
into a frame they passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

# Confidence-mask order of selfie_multiclass_256x256.
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)

DEFAULT_INPUT_LONG_SIDE = 384
DEFAULT_EMA_ALPHA = 0.5
# When update() has not been called for this long (segmentation was not
# needed for a while), the old masks and any run in flight are dropped.
_IDLE_RESET_MS = 500


@dataclass(frozen=True)
class SegMasks:
    """Full-frame float32 HxW masks in [0, 1] (read-only); None before the first run."""

    hair: np.ndarray | None = None
    face_skin: np.ndarray | None = None
    body_skin: np.ndarray | None = None


class SegmentationModel(Protocol):
    """What ``Segmenter`` needs from a segmentation model (fakes implement this)."""

    def segment(self, rgb: np.ndarray, ts_ms: int) -> Sequence[np.ndarray]:
        """Per-class confidence masks, HxW float32 at the size of ``rgb``."""

    def close(self) -> None: ...


class MediaPipeSegmenter:
    """Thin adapter over MediaPipe's Image Segmenter in VIDEO mode."""

    def __init__(self, model_path: str | Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        # Bytes instead of a path: MediaPipe cannot open non-ASCII paths on Windows.
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._image = mp.Image
        self._srgb = mp.ImageFormat.SRGB

    def segment(self, rgb: np.ndarray, ts_ms: int) -> list[np.ndarray]:
        result = self._segmenter.segment_for_video(self._image(image_format=self._srgb, data=rgb), ts_ms)
        # Copy out of MediaPipe-owned memory before the result is released.
        return [np.array(m.numpy_view(), dtype=np.float32).reshape(rgb.shape[:2]) for m in result.confidence_masks]

    def close(self) -> None:
        self._segmenter.close()


@dataclass(frozen=True)
class _Job:
    frame_bgr: np.ndarray
    ts_ms: int
    generation: int
    ema: np.ndarray | None  # probabilities before this run


@dataclass(frozen=True)
class _Run:
    job: _Job
    ema: np.ndarray
    masks: SegMasks


def model_input(frame_bgr: np.ndarray, long_side: int) -> np.ndarray:
    """Contiguous RGB copy of the frame, downscaled to at most ``long_side``."""
    height, width = frame_bgr.shape[:2]
    scale = long_side / max(height, width)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _full_frame_mask(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask.setflags(write=False)
    return mask


class Segmenter:
    """Hair / face-skin / body-skin masks for the current frame.

    Pass ``model`` to inject a ``SegmentationModel`` (tests); otherwise the
    MediaPipe model at ``model_path`` is loaded. With ``background=False``
    the model runs inline in ``update`` (deterministic, for tests/tools).
    """

    def __init__(
        self,
        model_path: str | Path | None,
        interval: int = 2,
        input_long_side: int = DEFAULT_INPUT_LONG_SIDE,
        *,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        background: bool = True,
        model: SegmentationModel | None = None,
    ) -> None:
        if model is None:
            if model_path is None:
                raise ValueError("model_path or model is required")
            model = MediaPipeSegmenter(model_path)
        self._model = model
        self._input_long_side = input_long_side
        self._ema_alpha = ema_alpha
        self._interval = 1
        self.set_interval(interval)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmenter") if background else None
        self._pending: Future[_Run] | None = None
        self._generation = 0
        self._last_ts = -1
        self._last_update_ms: int | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._last_run_index: int | None = None
        self._ema: np.ndarray | None = None
        self._masks = SegMasks()
        self._closed = False

    @property
    def interval(self) -> int:
        return self._interval

    def set_interval(self, interval: int) -> None:
        """Run the model at most every ``interval`` frames (>= 1)."""
        self._interval = max(1, int(interval))

    def update(self, frame_bgr: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        """Start a model run if one is due and return the latest masks.

        On frames without a finished run this returns the previous masks
        (the very same arrays). Exceptions from a background run are raised
        here, once. ``frame_bgr`` must not be written to afterwards.
        """
        height, width = frame_bgr.shape[:2]
        if self._frame_shape != (height, width) or self._was_idle(int(ts_ms)):
            self.reset()
            self._frame_shape = (height, width)
        self._last_update_ms = int(ts_ms)
        self._collect()
        if self._pending is None and self._is_due(frame_index):
            self._last_run_index = frame_index
            job = _Job(frame_bgr, self._next_timestamp(ts_ms), self._generation, self._ema)
            if self._executor is None:
                self._adopt(self._run(job))
            else:
                self._pending = self._executor.submit(self._run, job)
        return self._masks

    def reset(self) -> None:
        """Forget masks and smoothing; a run still in flight is discarded."""
        self._generation += 1
        self._last_run_index = None
        self._ema = None
        self._masks = SegMasks()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._model.close()

    # ---- internals --------------------------------------------------------

    def _was_idle(self, ts_ms: int) -> bool:
        return self._last_update_ms is not None and ts_ms - self._last_update_ms > _IDLE_RESET_MS

    def _is_due(self, frame_index: int) -> bool:
        if self._last_run_index is None:
            return True
        # A frame counter that went backwards (restarted) also triggers a run.
        return not 0 <= frame_index - self._last_run_index < self._interval

    def _next_timestamp(self, ts_ms: int) -> int:
        # MediaPipe VIDEO mode rejects timestamps that do not strictly increase.
        self._last_ts = max(int(ts_ms), self._last_ts + 1)
        return self._last_ts

    def _run(self, job: _Job) -> _Run:
        """One model run and everything that follows from it. Touches no state of ``self``
        that changes: this runs on the worker thread, while ``reset`` may be called."""
        rgb = model_input(job.frame_bgr, self._input_long_side)
        masks = self._model.segment(rgb, job.ts_ms)
        probabilities = np.stack([masks[HAIR], masks[FACE_SKIN], masks[BODY_SKIN]]).astype(np.float32, copy=False)
        ema = probabilities if job.ema is None else job.ema + self._ema_alpha * (probabilities - job.ema)
        height, width = job.frame_bgr.shape[:2]
        return _Run(job, ema, SegMasks(*(_full_frame_mask(p, width, height) for p in ema)))

    def _collect(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        future, self._pending = self._pending, None
        self._adopt(future.result())

    def _adopt(self, run: _Run) -> None:
        if run.job.generation == self._generation:
            self._ema, self._masks = run.ema, run.masks

> **注意（引き継ぎ時に追記）**: この文書はクラウド環境で書いた開発用の約束ごとです。`/tmp/claude-0/...scratchpad` のパスは当時の作業場所で、今は存在しません。読み替え: VENV → プロジェクトの `.venv`、モデル → `AI_MAKEUP_CAM_MODELS_DIR`（未設定ならユーザーデータ内に自動ダウンロード）、テスト素材 → `scripts/fetch_test_media.py` で作る `.test-media/`、renders → `.test-media/renders/` など。詳しくは `docs/dev/HANDOFF.md`。

# AI Makeup Cam — build contract (read fully before writing code)

## Goal
A Python app that applies **natural, finely adjustable real-time makeup** (skin, lips, blush,
eyeshadow, eyeliner, brows, hair gloss/smoothing/color/brightness) to a webcam feed and outputs
it as a **virtual camera** (pyvirtualcam → OBS Virtual Camera on Windows/macOS, v4l2loopback on
Linux) so it can be selected in Zoom / Teams / Google Meet for **job-interview video calls**.
The user is a Python programmer and wants full control of every parameter via a PySide6
settings window with live preview. Must run on **Windows and macOS** (Linux nice-to-have),
CPU only, real-time (target ≥ 24 fps at 1280x720 on a typical laptop; total effect time budget
≈ 15 ms/frame, landmarks+segmentation excluded). Naturalness matters more than drama: an
interviewer must not notice a filter. It must **never crash or freeze mid-interview**: per-effect
exceptions are caught, logged once, and that effect is skipped; lost face → pass frames through
(with makeup fading out over a few frames rather than popping).

Future (NOT now): a hairstyle-change effect may be added later — keep the effect plugin
architecture clean so it can drop in (see effects/base.py docstring).

## Location & environment
- Project root: `/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/` (call it ROOT).
  It is an independent sub-project inside an unrelated repo; touch nothing outside ROOT.
- Package: `ROOT/makeup_cam/`, tests: `ROOT/tests/` (pytest, `pythonpath=["."]` set in pyproject).
- Python venv with all deps (mediapipe 1.2.x tasks API, opencv-python-headless 5.0, numpy 2.x,
  PySide6 6.12, pyvirtualcam 0.15, pytest, scikit-image):
  `VENV=/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/venv`
  Run tests: `cd ROOT && $VENV/bin/python -m pytest -q tests/<your tests>`.
  Code must stay compatible with Python 3.10+ (use `from __future__ import annotations`).
- Pre-downloaded models (use via env `AI_MAKEUP_CAM_MODELS_DIR`):
  `SCRATCH=/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad`
  `$SCRATCH/models/face_landmarker.task` (sha256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff,
   URL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
  `$SCRATCH/models/selfie_multiclass_256x256.tflite` (sha256 c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0,
   URL https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite;
   output = 6 confidence masks in order: background, hair, body-skin, face-skin, clothes, others)
- Test images (public-domain NASA portrait, never commit them): `$SCRATCH/testimgs/` —
  `frame_720p.png` (1280x720 webcam-like), `frame_dark.png` (dim warm light), `frame_tilt.png`
  (head tilted 14°), `frame_far.png` (small face), `astronaut_512.png`. In repo tests, obtain the
  portrait via `skimage.data.astronaut()` and `pytest.importorskip("skimage")`; skip model-based
  tests when the models are not available (env var unset/missing files and no network).
- Visual check renders go to `$SCRATCH/renders/<your-area>/` (never into ROOT). **Look at them**
  with the Read tool (it displays images) and iterate until the result looks natural.
- MediaPipe in this container needs EGL libs (installed). Teardown prints harmless
  "NoneType is not callable" noise at interpreter exit; ignore. Call `.close()` explicitly.
- MediaPipe tasks API (no `mp.solutions` in this version):
  `from mediapipe.tasks.python import vision, BaseOptions`;
  `vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=...), running_mode=vision.RunningMode.VIDEO, num_faces=1))`,
  `.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8_contiguous), ts_ms)` (ts strictly increasing);
  `vision.ImageSegmenter...(output_confidence_masks=True, output_category_mask=False)`, `.segment_for_video(img, ts)`
  → `.confidence_masks[i].numpy_view()` (HxWx1 float32 at input size).
  Measured on this (slow) container: landmarks+multiclass seg on 512x512 ≈ 120 ms; laptops are much faster,
  but keep segmentation cheap (downscale input, run every N frames, async-friendly).

## Already written (do NOT rewrite; read them): 
- `makeup_cam/config.py` — all settings dataclasses with GUI metadata (`kind`, `min`, `max`, `label`…),
  `MakeupSettings` (to/from dict/json, tolerant + clamping), `SECTIONS`, `SECTION_LABELS`,
  `section_fields()`, `hex_to_bgr("#RRGGBB") -> (b,g,r) floats`, `user_data_dir()`,
  `user_presets_dir()`, `BUILTIN_PRESETS_DIR` (= makeup_cam/presets/).
  If you truly need a new settings field, add it to config.py in your own section only, with
  metadata, and report it.
- `makeup_cam/effects/base.py` — `FrameContext`, `Effect` ABC (`name`, `order`, `needs_face`,
  `needs_hair_mask`, `apply(frame_f32_bgr01, ctx, section_settings) -> frame`, `reset()`),
  `register_effect`, `registered_effects()`.
- `makeup_cam/effects/__init__.py` exposes `load_builtin_effects()` which imports modules
  `hair, skin, brows, eyes, blush, lips` and returns `registered_effects()`. Importing
  `makeup_cam.effects.base` alone does NOT import the effect modules (so each effect can be
  developed/tested on its own).

Conventions: frames inside the pipeline are float32 BGR in [0,1], HxWx3, C-contiguous.
Landmarks are full-frame pixel coords of the UNMIRRORED frame; "LEFT_*" constants mean the
**subject's** left (which appears on the image's right side). Effects must scale all sizes by
face size (`regions.face_scale`), never by fixed pixels, so they work near and far.
Effects should work inside a ROI around the face (crop → process → write back), not on the full
frame, for speed. Every opacity/strength is multiplied by `ctx.intensity`. With a slider at 0 the
effect must be an exact no-op. Comments/docstrings in English; user-facing strings (GUI, CLI
messages, errors shown to the user, README) in Japanese.

## Module ownership & APIs (implement exactly these names; you may add more)

### regions.py + blend.py  (owner: geometry agent)
regions.py:
- Ordered landmark index lists (closed polygons unless noted): `FACE_OVAL`, `LIPS_OUTER`,
  `LIPS_INNER`, `LEFT_EYE`, `RIGHT_EYE`, `LEFT_BROW`, `RIGHT_BROW`, `LEFT_IRIS`(468-472 range),
  `RIGHT_IRIS`, open polylines ordered inner corner → outer corner: `LEFT_EYE_UPPER`,
  `LEFT_EYE_LOWER`, `RIGHT_EYE_UPPER`, `RIGHT_EYE_LOWER`; `LEFT_CHEEK_CENTER`,
  `RIGHT_CHEEK_CENTER` (single indices or small index lists on the cheekbone apple),
  `NOSE_TIP`, and anything else useful (under-eye area, nose wings, eye-brow crease points).
  Verify them visually by drawing them on the test images.
- `@dataclass(frozen=True) class Roi: x0,y0,x1,y1` with `.h`, `.w`, `.slices` (for frame[roi.slices]), `.offset` (x0,y0).
- `roi_around(points: ndarray(N,2), shape: (H,W), pad: float = 0.25) -> Roi | None` (pad as fraction of bbox size, clipped; None if empty).
- `face_scale(landmarks) -> float` — inter-ocular distance (eye-corner to eye-corner or iris centers) in px.
- `polygon_mask(points, roi, feather_px=0.0) -> float32 (roi.h, roi.w)` in [0,1] (anti-aliased, supersample or cv2.LINE_AA).
- `polyline_mask(points, thickness_px, roi, feather_px=0.0, thickness_profile=None)` (profile optional per-vertex thickness multipliers for tapering eyeliner).
- `ellipse_mask(center_xy, axes_xy, angle_deg, roi, feather_px=0.0)`.
- `feather(mask, sigma_px) -> mask` (gaussian; sigma≤0 → unchanged).
- `expand_polygon(points, amount_px)` (offset outward from centroid; negative shrinks).
- Region helpers returning ROI-sized masks: `lips_mask(landmarks, roi, outline=0.0, feather_px=…)`
  (outer minus inner opening/teeth), `eye_opening_mask(landmarks, side, roi, …)`,
  `brow_mask(landmarks, side, roi, thickness=0.0, …)`, `face_oval_mask(landmarks, roi, …)`.
blend.py (all float32, inputs [0,1], masks broadcast HxW):
- `mix(base, other, mask)` = base*(1-m)+other*m.
- `blend_multiply`, `blend_screen`, `blend_overlay`, `blend_soft_light` (pegtop/W3C), each
  `(base, color_bgr_tuple_or_array) -> result` (no masking; combine with `mix`).
- `apply_color(base, color_bgr, mask, opacity, mode="soft_light"|"multiply"|"normal"|"color"|"overlay"|"screen")`.
  `"color"` = keep base luminance, take hue/chroma of color (Lab a/b) — the most natural mode for
  lipstick/hair tint.
- `bgr_to_lab(img) / lab_to_bgr(lab)` (cv2 float32 conversion, L in 0..100).
- `adjust_brightness_contrast(img, brightness, contrast, mask=None, pivot=None)`.
- `luminance(img) -> HxW`.
Tests: `tests/test_regions.py`, `tests/test_blend.py` (synthetic landmarks are fine; plus an
optional model-based test that indices land where expected on the astronaut portrait).

### models.py, tracking.py, segmentation.py, pipeline.py  (owner: tracking agent)
- models.py: `MODEL_SPECS` (name → url, sha256, filename); `models_dir() -> Path`
  (env `AI_MAKEUP_CAM_MODELS_DIR`, else `config.user_data_dir()/"models"`);
  `ensure_model(name, progress=None) -> Path` (download to temp + verify sha256 + atomic rename;
  clear Japanese error on failure/offline); `ensure_all_models(progress=None)`.
- tracking.py: `class OneEuroFilter` (vectorized over (N,2) arrays, min_cutoff/beta/d_cutoff);
  `class FaceTracker(model_path, smoothing=0.5)` with `detect(frame_bgr_uint8, ts_ms) -> (landmarks (478,2) f32 | None, z (478,) | None)`,
  `set_smoothing(0..1)` (maps to filter params: 0 = raw/fast, 1 = very stable), `reset()`, `close()`.
  Feed MediaPipe a downscaled copy if the frame is large (e.g. longest side 640) and scale points back.
  Handle the 478-point output; one face only; ts must be strictly increasing (guard it).
- segmentation.py: `class Segmenter(model_path, interval=2, input_long_side=384)` with
  `update(frame_bgr_uint8, ts_ms, frame_index) -> SegMasks` where `SegMasks(hair, face_skin, body_skin)`
  are float32 full-frame HxW masks (None before first run). Runs the model only every `interval`
  frames, upsamples (cv2.resize INTER_LINEAR), temporal EMA, `set_interval()`, `reset()`, `close()`.
  Optional: restrict to a region around the face+hair to save time (keep simple & correct first).
- pipeline.py:
  ```python
  @dataclass
  class ProcessResult:
      output: np.ndarray      # uint8 BGR HxWx3 sent to the virtual camera (mirrored iff general.mirror_output)
      preview: np.ndarray     # uint8 BGR for the GUI (mirrored iff general.mirror_preview; landmark dots iff show_landmarks)
      original_preview: np.ndarray  # unprocessed frame, mirrored like preview (for before/after compare)
      face_found: bool
      timings_ms: dict[str, float]  # "landmarks", "segmentation", each effect name, "total"
  class MakeupPipeline:
      def __init__(self, settings: MakeupSettings | None = None, *, tracker=None, segmenter=None,
                   model_dir: Path | None = None, effects: list[Effect] | None = None): ...
          # tracker/segmenter injectable for tests; otherwise built from models.ensure_model(...)
      def set_settings(self, settings: MakeupSettings) -> None  # thread-safe (lock + copy)
      def get_settings(self) -> MakeupSettings
      bypass: bool  # property; True = pass frames through untouched (GUI "メイクOFF")
      def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ProcessResult
      def close(self) -> None
  ```
  process(): build FrameContext (intensity, masks, landmarks), run `registered_effects()`
  instances in order where section `.enabled` and requirements met; catch & log per-effect
  exceptions (log once per effect per message, keep running); fade makeup in/out over ~6 frames
  when the face appears/disappears (blend processed vs original by a fade factor) to avoid
  popping; skip segmentation work entirely when no enabled effect needs it (hair effect off →
  no segmentation unless skin uses face_skin_mask — skin effect MAY use face_skin_mask if
  available, so run segmentation when hair OR skin is enabled). Default effects:
  call `makeup_cam.effects.load_builtin_effects()` lazily inside `__init__` only when `effects=None`. Until the effect modules exist, test with fake
  effects injected via `effects=`.
  Tests: one-euro behavior, segmentation EMA/interval with a fake model, pipeline with fake
  tracker/segmenter/effects (exception isolation, bypass, mirroring, fade, no-face path),
  and an optional real-model test on the astronaut image.

### camera.py + runner.py  (owner: camera agent)
- camera.py: `@dataclass CameraInfo(index, name, width, height)`; `list_cameras(max_index=8) -> list[CameraInfo]`
  (probe quickly with the right backend; on Windows `cv2.CAP_DSHOW` (fallback MSMF), macOS
  `cv2.CAP_AVFOUNDATION`, Linux `cv2.CAP_V4L2`; names via platform tricks if cheap, else "カメラ 0");
  `class CameraSource(index=0, width=1280, height=720, fps=30)` with `open()`, `read() -> uint8 BGR | None`,
  `close()`, auto-reopen after repeated failures, actual resolution properties; MJPG fourcc hint for 720p30 on Windows.
  `class VirtualCameraOutput(width, height, fps)` wrapping `pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR)`,
  `.device` name, `send(frame_bgr)` (resize if size differs), `close()`, and
  `VirtualCameraError` with **Japanese, actionable** messages per OS (Windows: install OBS Studio ≥ 26 /
  restart apps; macOS: install OBS Studio ≥ 30, open OBS once and click "仮想カメラ開始" once to
  install the system extension and allow it in System Settings; Linux: `sudo modprobe v4l2loopback`).
- runner.py: UI-agnostic worker:
  ```python
  @dataclass
  class RunnerStatus:
      fps: float; camera_ok: bool; camera_error: str | None; virtual_camera: str | None
      virtual_camera_error: str | None; face_found: bool; processing_ms: float
  class CameraRunner:
      def __init__(self, pipeline: MakeupPipeline, *, camera_index=0, width=1280, height=720, fps=30,
                   virtual_camera=True, on_frame=None, on_status=None,
                   camera_factory=CameraSource, vcam_factory=VirtualCameraOutput): ...
      def start(self) / stop(self, timeout=2.0)   # background thread; stop is idempotent
      def set_settings(self, s) ; set_bypass(self, b: bool)
      def set_camera(self, index, width=None, height=None)   # reopen in worker thread
      def set_virtual_camera(self, enabled: bool)            # open/close vcam in worker thread
      @property status -> RunnerStatus
  ```
  on_frame(ProcessResult) and on_status(RunnerStatus) are called from the worker thread (GUI
  marshals via Qt signals). If the camera fails, keep the vcam alive sending a neutral "カメラ未接続"
  frame (don't let the call show a frozen image), and retry opening the camera every ~2 s.
  The virtual camera keeps receiving frames at a steady rate even if processing is slow (send the
  latest frame). Never let an exception kill the thread; report via status.
  Tests with fake camera / fake vcam / fake pipeline (no real hardware here).

### presets.py + gui.py  (owner: gui agent)
- presets.py: `list_presets() -> list[PresetInfo(name, path, builtin: bool)]` (builtin from
  `BUILTIN_PRESETS_DIR/*.json`, user from `user_presets_dir()`; user overrides builtin of same name),
  `load_preset(name) -> MakeupSettings`, `save_preset(name, settings)` (user dir; sanitize file
  names but keep Japanese names working), `delete_preset(name)` (user only), `last_session_path()`,
  `save_last_session(s)`, `load_last_session() -> MakeupSettings | None`.
- gui.py: PySide6 `MainWindow` + `run_gui(args) -> int`. Layout: left = live preview (scaled
  QLabel or custom widget, keeps aspect), right = QTabWidget with one tab per `SECTIONS` entry
  (labels from `SECTION_LABELS`), **generated from `section_fields()` metadata**: float →
  slider + spinbox (2 decimals) synced; int → spinbox/slider; bool → checkbox; color → swatch
  button opening QColorDialog (+ hex text); choice → QComboBox. Each widget shows `help` as
  tooltip and has a small "↺" reset-to-default. Top bar: camera selector (from list_cameras,
  run in a background thread because probing is slow), resolution (1280x720 / 960x540 / 640x480),
  "仮想カメラ出力" toggle, "メイクOFF" toggle, press-and-hold "比較（すっぴん）" button showing
  original_preview, preset combo + 保存 / 名前を付けて保存 / 削除 / 初期値に戻す, status bar
  (FPS, processing ms, 顔検出 OK/NG, virtual camera device or Japanese error). Settings changes
  apply live (debounce ~30 ms) via runner.set_settings. Autosave last session on change
  (debounced) and restore on start. Keyboard: Space = hold compare. Window title "AI メイクカメラ".
  Frames arrive on a worker thread → emit a Qt signal with the numpy array → convert to QImage
  in the GUI thread (copy!). Drop frames if the GUI is behind (keep only latest).
  Must be testable offscreen (`QT_QPA_PLATFORM=offscreen`) with a fake runner: tests build the
  window, change a slider, check the settings object, save/load preset roundtrip. Write a
  screenshot `$SCRATCH/renders/gui/window.png` with fake frames to review the layout (install
  `fonts-noto-cjk` via apt if Japanese renders as boxes — that is only for the screenshot).

### effects/*.py  (owners: effect agents; written AFTER regions/blend/tracking exist)
- `effects/hair.py` — `HairEffect` name "hair", order 10, needs_hair_mask=True, needs_face=False.
- `effects/skin.py` — `SkinEffect` name "skin", order 20.
- `effects/brows.py` — `BrowEffect` name "brows", order 30.
- `effects/eyes.py` — `EyeshadowEffect` "eyeshadow" order 40, `EyelinerEffect` "eyeliner" order 50.
- `effects/blush.py` — `BlushEffect` "blush" order 60.
- `effects/lips.py` — `LipEffect` "lips" order 70.

### Integration (owner: integrator, last)
`makeup_cam/__main__.py` (CLI), built-in presets JSON, README.md (Japanese), launch scripts,
end-to-end tests, final before/after gallery.

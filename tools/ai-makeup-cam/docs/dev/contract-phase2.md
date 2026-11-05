> **注意（引き継ぎ時に追記）**: この文書はクラウド環境で書いた開発用の約束ごとです。`/tmp/claude-0/...scratchpad` のパスは当時の作業場所で、今は存在しません。読み替え: VENV → プロジェクトの `.venv`、モデル → `AI_MAKEUP_CAM_MODELS_DIR`（未設定ならユーザーデータ内に自動ダウンロード）、テスト素材 → `scripts/fetch_test_media.py` で作る `.test-media/`、renders → `.test-media/renders/` など。詳しくは `docs/dev/HANDOFF.md`。

# AI Makeup Cam — phase 2 contract (read fully; also skim CONTRACT.md for environment details)

## Context
Phase 1 is done and green (547 tests): a working real-time makeup virtual camera at
`ROOT=/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam` (read `README.md` there first for
the architecture). The user is a Japanese woman job-hunting; she wants to skip makeup for ONLINE
interviews (Zoom/Teams/Meet) and look like she does in person with full makeup. Target motion
envelope = a typical Japanese online interview: seated, talking, nodding, smiling with teeth, small
head turns (yaw ±20°, pitch ±15°, roll ±10°), glancing down at notes, small hand gestures below the
face, and BOWS (お辞儀) at start/end where the face drops or leaves the frame. Out of scope: raising
hands, long hand-over-face, standing up. Everything must be natural (an interviewer must not notice
a filter) and must NEVER glitch visibly within that envelope; when unsure, fade an effect out rather
than show an artifact.

Phase 2 adds: teeth, contour+tear bag, face reshape (geometric warp), hair shape (flyaways,
silhouette, background plate), then integration, then video QA on real talking-face footage.

## Environment (same as phase 1)
- VENV python: `/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/venv/bin/python`
- `SCRATCH=/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad`
- Models: `AI_MAKEUP_CAM_MODELS_DIR=$SCRATCH/models`. GUI tests: `QT_QPA_PLATFORM=offscreen`.
- Still test frames: `$SCRATCH/testimgs/frame_{720p,dark,tilt,far}.png` (NASA portrait, public domain).
- **Talking-face videos (all women, as the user asked)**: `$SCRATCH/testvids/<actor>_<sentence>_<emotion>_<level>.flv`
  from CREMA-D (ODbL/DbCL licence; 480x360, ~30 fps, 2-4 s each, green screen). Actors:
  1002,1012,1013,1029,1046,1060,1072,1073,1074,1076,1082 (Caucasian, Hispanic, African American,
  Asian; age 20-33); clips per actor: `IEO_HAP_HI` (big smile, teeth), `DFA_NEU_XX`, `ITS_HAP_XX`,
  `TIE_SAD_XX`. `demographics.csv` is there too. Upscale to 960x720 (INTER_CUBIC) and pad to
  1280x720 to mimic a webcam when useful. NEVER copy any test image/video into ROOT.
- Renders go to `$SCRATCH/renders/<area>/`; Read them (the Read tool shows images) and iterate.
- `ffmpeg` is installed (use it to make phone-friendly H.264 mp4/gif from cv2 `mp4v` output).

## Already done for you (do not redo)
- `makeup_cam/config.py` now has sections with GUI metadata: `contour` (ContourSettings),
  `tearbag` (TearbagSettings), `teeth` (TeethSettings), `reshape` (ReshapeSettings),
  `hairshape` (HairShapeSettings); `SECTIONS`/`SECTION_LABELS` include them (GUI tabs auto-generate).
  You may tune defaults/ranges/help text **only in your own section(s)**; keep every slider an exact
  no-op at 0 (reshape: at 0 for every shape slider).
- `effects/base.py` `FrameContext` has `background: np.ndarray | None` (float32 BGR [0,1] full-frame
  empty-room plate, or None).
- Built-in presets contain the new sections with provisional values.
- Effect modules are NOT yet in `effects/__init__.py:BUILTIN_EFFECT_MODULES` — the integrator adds
  them. Until then, test with `MakeupPipeline(..., effects=[YourEffect(), ...])`.

## Effect order (Effect.order) and modules
| order | name (= section) | module | class |
|---|---|---|---|
| 5 | reshape | effects/reshape.py | ReshapeEffect |
| 8 | hairshape | effects/hairshape.py | HairShapeEffect |
| 10 | hair | (exists) | |
| 20 | skin | (exists) | |
| 25 | contour | effects/contour.py | ContourEffect |
| 30 | brows | (exists) | |
| 40 | eyeshadow | (exists) | |
| 45 | tearbag | effects/contour.py | TearbagEffect |
| 50 | eyeliner | (exists) | |
| 60 | blush | (exists) | |
| 70 | lips | (exists) | |
| 75 | teeth | effects/teeth.py | TeethEffect |

## Rules for all effects (same as phase 1)
Exact no-op at 0 strength / intensity 0 (bit-identical); float32 BGR [0,1] in/out; ROI-only work;
sizes scaled by `regions.face_scale`; strengths × `ctx.intensity`; ≤ 4 ms per effect at 1280x720 in
this container (reshape/hairshape may use up to 6 ms); temporal stability (no per-frame random
noise; any per-frame statistics must be smoothed with state reset in `reset()`); reuse
`regions.py`/`blend.py` helpers (read them); tests in your own test file(s) including a real-landmark
test with the conftest fixtures; visual tuning loop on the still frames AND on several CREMA-D clips
(render before/after strips over time, e.g. every 5th frame, to judge temporal behaviour).

## Special contracts
- **reshape may mutate the context**: after warping the frame it MUST set `ctx.landmarks` (and
  `ctx.landmarks_z` unchanged) to the warped landmark positions and warp `ctx.hair_mask`,
  `ctx.face_skin_mask`, `ctx.body_skin_mask` with the same map, so later effects (lips, liner…)
  align with the reshaped face. Document this in the `FrameContext` docstring (only that edit in
  base.py). Use a backward map (out(p) = in(p + b(p))) built so that landmark targets land exactly
  where intended (e.g. RBF/thin-plate or Gaussian-kernel interpolation of (source − target) offsets
  from target positions, on a coarse grid upsampled, then `cv2.remap` within a padded ROI). Keep all
  displacements small and smooth (max ≈ 6-8 % of face width at slider 1.0); fade the field to zero
  outside the person according to `protect_background`; if `ctx.background` is available, use it to
  fill background exposed by slimming. Must stay stable with landmark jitter (smoothness of the field
  is key) and during bows (the pipeline fade handles face loss).
- **hairshape + background plate**: new `makeup_cam/background.py` (capture: median of ~15 frames
  after a 3 s countdown, reject if a person is still visible per segmentation; save/load
  `user_data_dir()/background_<W>x<H>.png`; invalid when resolution changes). The hairshape agent
  ALSO owns minimal, well-tested edits to `pipeline.py` (hold the plate, expose
  `capture_background(...)`/`clear_background()`, put it in `ctx.background`), `runner.py`
  (thread-safe command to capture), and `gui.py` (button 「背景を撮影」 with a Japanese countdown
  overlay 「3秒後に背景を撮影します。カメラの前から離れてください」, success/failure message,
  「背景を削除」). No other agent edits pipeline.py, runner.py or gui.py in the build phase.

## Ownership in the build phase (only touch your files + your config section)
- teeth agent: `effects/teeth.py`, `tests/test_effect_teeth.py`, TeethSettings.
- contour agent: `effects/contour.py`, `tests/test_effect_contour.py`, ContourSettings, TearbagSettings.
- reshape agent: `effects/reshape.py`, `tests/test_effect_reshape.py`, ReshapeSettings, FrameContext docstring note.
- hairshape agent: `effects/hairshape.py`, `makeup_cam/background.py`, `tests/test_effect_hairshape.py`,
  `tests/test_background.py`, HairShapeSettings, and the pipeline/runner/gui edits above (+ their tests).

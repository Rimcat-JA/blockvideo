// NOTE (handoff): workflow script used in the cloud session. Paths under /tmp/claude-0/... no longer exist;
// adapt SCRATCH/ROOT/VENV to the local checkout (see docs/dev/HANDOFF.md) before re-running.
export const meta = {
  name: 'makeup-cam-plan-a',
  description: 'Plan A: build a preset from a photo of the user wearing her real makeup (analyzer + round-trip verification)',
  phases: [
    { title: 'Build', detail: 'analyze.py: measure makeup from a photo (optional bare photo) and emit a preset' },
    { title: 'Verify', detail: 'independent adversarial round-trip verification and fixes' },
  ],
}

const SCRATCH = '/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad'
const ROOT = '/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam'
const VENV = `${SCRATCH}/venv/bin/python`

const REPORT = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string' },
    accuracy: { type: 'string', description: 'Round-trip accuracy table (preset -> render -> analyze -> compare)' },
    renders: { type: 'array', items: { type: 'string' } },
    open_issues: { type: 'string' },
  },
  required: ['summary', 'files', 'tests', 'accuracy', 'open_issues'],
}

const CONTEXT = `Project: a real-time AI makeup virtual camera at ${ROOT} (read README.md, makeup_cam/config.py, makeup_cam/regions.py, makeup_cam/blend.py, makeup_cam/tracking.py, makeup_cam/segmentation.py, makeup_cam/pipeline.py, makeup_cam/presets.py, makeup_cam/__main__.py (render_still), and the effects in makeup_cam/effects/ — especially lips.py, blush.py, eyes.py, brows.py, skin.py — to understand EXACTLY how each setting maps to pixels). Environment notes: ${SCRATCH}/CONTRACT.md and ${SCRATCH}/CONTRACT2.md (read the "Environment" parts). Use ${VENV}; models in AI_MAKEUP_CAM_MODELS_DIR=${SCRATCH}/models; test stills in ${SCRATCH}/testimgs; female talking-face clips in ${SCRATCH}/testvids (CREMA-D, ODbL). Never copy images/videos into ${ROOT}. Do not touch git.
IMPORTANT concurrency rule: another workflow is concurrently editing effects (teeth, contour, reshape, hairshape), pipeline.py, runner.py, gui.py, __main__.py, config.py, presets and README. You must NOT edit any existing file. You only create: makeup_cam/analyze.py, tests/test_analyze.py, and (optionally) docs/plan-a-photo-preset.md. Integration into the CLI/GUI happens later.
The user: a Japanese woman job-hunting who does full makeup for in-person interviews and wants the app to reproduce THAT look for online interviews without makeup. "Plan A" = she takes a photo of herself with her usual makeup (ideally with the webcam under her interview lighting; optionally also a bare-face photo in the same setup, see docs/capture-guide.md §2) and the app builds a preset that reproduces it.`

phase('Build')
const built = await agent(`${CONTEXT}

Build makeup_cam/analyze.py:
- API: \`analyze_makeup(makeup_bgr: np.ndarray, bare_bgr: np.ndarray | None = None, *, base: MakeupSettings | None = None, models_dir=None) -> AnalysisResult\` where AnalysisResult has \`settings: MakeupSettings\` (base preset — default natural — with the measured sections overwritten), \`measurements: dict\` (raw per-region numbers for debugging), \`confidence: dict[str, float]\` per section, and \`notes: list[str]\` (Japanese, user-facing: e.g. 「すっぴん写真がないため、リップの濃さは推定です」). Plus \`save_as_preset(result, name)\` using presets.save_preset, and a module CLI: \`python -m makeup_cam.analyze MAKEUP.jpg [--bare BARE.jpg] [--base natural] --name いつものメイク [--json OUT.json]\` with Japanese messages.
- Landmarks: the first Face Landmarker frame is inaccurate (stretched); feed the still repeatedly in VIDEO mode (like __main__.render_still) until stable. Use the segmenter for hair/face-skin masks. Fail with a clear Japanese error if no face / more than one face / face too small.
- Measure robustly in Lab (trimmed medians, exclude speculars/shadows), per region using regions.py masks: lips (color, intensity vs bare or vs a skin-derived natural-lip baseline, finish from specular ratio, outline over/under from color extent beyond the lip line if bare is given), blush (cheek-apple chroma excess vs reference skin, hue → color, extent → size, offset), eyeshadow (lid color vs brow-bone skin, extent → height), eyeliner (dark band above the upper lash line → thickness; dark pixels beyond the outer corner → wing; lower line), brows (fill color/darkness vs bare or vs neighbouring skin), skin (foundation tone from the face's median skin; smoothing/texture from high-frequency energy vs bare), and — when a bare photo is given — contour/highlight and tear bag from luminance differences in those regions (set only fields that exist in config.py; read the current config first). Leave teeth/reshape/hair/hairshape at base values.
- Invert the effect models: choose setting values so that APPLYING the app's effect to the bare face reproduces the measured made-up appearance. Best approach: a small per-section optimisation (e.g. coordinate search / golden-section on opacity, direct color estimate) that renders the single effect on the bare photo (or on the makeup photo with the region "un-made-up" approximated if no bare photo) and minimises ΔE to the made-up region. Keep runtime < ~10 s on this container.
- White balance: if both photos are given, normalise them to each other using the background/clothing (non-face) region; otherwise gray-world on the background. Document limitations.
- Tests (tests/test_analyze.py): unit tests for the measurement helpers with synthetic images; and ROUND-TRIP tests (skip without models/skimage): take face_frame_720p as the "bare" face, render it with a known preset (polished, plus 2-3 custom settings: e.g. red matte lips, coral blush, brown smoky shadow, winged liner) via the real pipeline, run analyze_makeup(rendered, bare=face_frame_720p) and assert the recovered settings are close (lip color ΔE00 < 8, opacities ±0.15, eyeliner wing presence, blush present/absent, etc.); and the same WITHOUT the bare photo with looser tolerances. Also run the round-trip on 2-3 CREMA-D women (extract a frame) and report.
- Report an accuracy table. Render ${SCRATCH}/renders/plan_a/: target (made-up) | re-rendered from recovered preset | difference, and Read them.
Run: cd ${ROOT} && AI_MAKEUP_CAM_MODELS_DIR=${SCRATCH}/models ${VENV} -m pytest -q tests/test_analyze.py`, { label: 'plan-a:build', phase: 'Build', schema: REPORT })

phase('Verify')
const verified = await agent(`${CONTEXT}

You are an independent, skeptical verifier of makeup_cam/analyze.py (Plan A). The builder reported: ${JSON.stringify(built)}
Try hard to break it, then fix what is real (you may edit only makeup_cam/analyze.py, tests/test_analyze.py, docs/plan-a-photo-preset.md):
1. Re-run the round-trip yourself with NEW targets the builder did not use (e.g. nude/beige lips, very dark berry lips, no blush, heavy blush, no liner, thick liner without wing, light vs dark brows), and on CREMA-D women of different skin tones (1029/1060/1074 dark skin, 1072 Asian, 1013 Hispanic) — extract frames, render targets with the real pipeline, analyze with and without the bare frame. Also test robustness: different white balance between bare and makeup photos (apply a color cast), JPEG compression, slight head pose difference between the two photos (use two different frames of the same clip), smiling vs neutral.
2. Check failure handling: no face, two faces, tiny face, closed eyes, open mouth — clear Japanese errors or graceful confidence drops, never nonsense presets (e.g. opacity 1.0 garbage).
3. Check that a preset produced from a real photo of someone WITHOUT makeup (use the bare frames as "makeup" photos) yields near-zero makeup (not hallucinated makeup).
4. Fix real problems, add regression tests, update the accuracy table, and write docs/plan-a-photo-preset.md (Japanese, short: how to take the photo, how to run, what is measured, limitations).
Run the tests at the end.`, { label: 'plan-a:verify', phase: 'Verify', schema: REPORT })

return { built, verified }

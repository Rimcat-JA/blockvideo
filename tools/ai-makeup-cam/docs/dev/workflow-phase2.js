// NOTE (handoff): workflow script used in the cloud session. Paths under /tmp/claude-0/... no longer exist;
// adapt SCRATCH/ROOT/VENV to the local checkout (see docs/dev/HANDOFF.md) before re-running.
export const meta = {
  name: 'makeup-cam-phase2',
  description: 'Add teeth, contour/tear bag, face reshape, hair shape; integrate; video QA on female talking-face clips; fix',
  phases: [
    { title: 'Build', detail: 'teeth, contour+tearbag, reshape, hairshape+background plate' },
    { title: 'Integrate', detail: 'register effects, presets, GUI tabs, README, perf' },
    { title: 'Video QA', detail: 'render female talking-face clips, bows, metrics, honest findings' },
    { title: 'Fix', detail: 'fix QA findings, re-verify, final sample videos' },
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
    tests: { type: 'string', description: 'Exact command and result line' },
    perf: { type: 'string' },
    renders: { type: 'array', items: { type: 'string' } },
    api_deviations: { type: 'string' },
    open_issues: { type: 'string' },
  },
  required: ['summary', 'files', 'tests', 'api_deviations', 'open_issues'],
}

const QA_REPORT = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'Honest overall verdict per preset: would an interviewer notice?' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          effect: { type: 'string' },
          description: { type: 'string', description: 'What looks wrong, where (clip/frame), and how to reproduce' },
          evidence: { type: 'array', items: { type: 'string' }, description: 'Render paths' },
          suspected_cause: { type: 'string' },
        },
        required: ['id', 'severity', 'effect', 'description', 'evidence', 'suspected_cause'],
      },
    },
    metrics: { type: 'string', description: 'Temporal metrics per preset/effect' },
    tooling: { type: 'string', description: 'Scripts/CLI added and how to run them' },
    files: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string' },
  },
  required: ['summary', 'findings', 'metrics', 'tooling', 'files', 'tests'],
}

const COMMON = `You are one of several agents extending a working Python app. FIRST read ${SCRATCH}/CONTRACT2.md completely (phase 2 contract), then skim ${SCRATCH}/CONTRACT.md for environment notes, then read ${ROOT}/README.md, makeup_cam/config.py, makeup_cam/effects/base.py, makeup_cam/regions.py, makeup_cam/blend.py, makeup_cam/pipeline.py and one existing effect (e.g. makeup_cam/effects/lips.py) plus its tests to match conventions.
Rules: only touch the files you own (per CONTRACT2.md). Do not touch git. Never put test images/videos into ${ROOT}. Use ${VENV}. Run only your tests during the build phase (others are editing concurrently). Production quality, English comments, Japanese user-facing strings. Your final answer is a structured report.`

phase('Build')
const BUILD = [
  {
    key: 'teeth',
    prompt: `YOUR AREA: teeth effect (effects/teeth.py, TeethEffect, name "teeth", order 75; tests/test_effect_teeth.py; TeethSettings).
Mask: inner-mouth polygon (regions LIPS_INNER, slightly eroded, feathered by "feather") ∩ a soft "teeth-likeness" weight from Lab: high L relative to the mouth interior, low chroma, not red (exclude gums/tongue/lips) — compute adaptive thresholds per frame from mouth-interior statistics, smooth them over time (EMA, reset() clears). Closed mouth → no-op quickly.
whitening: lift L with a soft shoulder (never exceed ~92 L or flatten texture; keep enamel translucency at the edges), yellow_reduction: pull b* toward a slightly warm neutral (never blue/grey), evenness: edge-preserving smoothing of L/a/b inside the teeth mask + partially lift dark interdental gaps/overlap shadows while keeping faint separations so teeth never merge into one white block. Natural defaults (interview-appropriate, "clean teeth", not veneers).
Visual loop: the CREMA-D IEO_HAP_HI clips (big smiles with teeth) for several actors of different skin tones + frame_720p; render zoomed mouth crops before/after over time. Check temporal stability while talking (mouth opening/closing).`,
  },
  {
    key: 'contour',
    prompt: `YOUR AREA: effects/contour.py with ContourEffect (name "contour", order 25) and TearbagEffect (name "tearbag", order 45); tests/test_effect_contour.py; ContourSettings, TearbagSettings.
CONTOUR shading (multiply/soft-light with shading_color, × shading × per-area weight): jawline band just inside the lower face oval (from below the ear to the chin, fading toward the chin center), cheek hollow (soft diagonal band under the cheekbone from near the ear toward the mouth corner, stopping well before it), nose shadow (soft lines along both sides of the nose bridge from brow inner corner down, fading), forehead hairline band (top of face oval, restricted to skin; skip where hair/bangs cover). HIGHLIGHT (screen/soft-light with highlight_color, × highlight × weight): nose bridge line, top of cheekbones (C-shape toward temple), chin center, cupid's bow. Everything follows head yaw/roll from landmarks, scaled by face size, heavily feathered (softness), restricted to skin (face oval ∩ face_skin_mask when present), never on eyes/brows/lips/nostrils.
TEARBAG: soft highlight band directly under the lower lash line (lower eyelid line), from about the inner third to "width" toward the outer corner, plus a faint thin shadow line under it (× shadow), feathered; must follow blinking/smiling (the bag bulges when smiling) and never paint the eyeball.
Natural defaults: real makeup contour should be invisible as "makeup" but make the face look smaller/more defined. Visual loop on stills + several CREMA-D clips across skin tones (incl. dark skin: shading must not look grey/ashy — adapt shading color toward the skin's own darker tone or use multiply in a way that preserves undertone).`,
  },
  {
    key: 'reshape',
    prompt: `YOUR AREA: effects/reshape.py (ReshapeEffect, name "reshape", order 5); tests/test_effect_reshape.py; ReshapeSettings; and ONLY a docstring note in effects/base.py FrameContext about reshape mutating ctx (see CONTRACT2 "Special contracts").
Implement a smooth, small, stable geometric warp (beauty-app style) of real pixels: face_slim (pull lower face oval inward toward the face center line, stronger at the cheeks/jaw, not at the temples), jaw (narrow the gonial angle region), chin_length (move chin points down/up), eye_size (scale each eye region around its center incl. iris; gentle falloff to brows/cheeks), nose_width (narrow nose wings), nose_length (move nose tip/base up), mouth_width (scale mouth around center horizontally), mouth_corner (lift corners), lip_fullness (expand lips vertically from the lip line), brow_lift (move brows up). Build target landmark positions, then a backward displacement field b on a coarse grid (e.g. 1/4 res of a padded face ROI) via Gaussian-RBF / thin-plate interpolation of (source−target) at target positions + fixed zero anchors on an outer ring (expanded face oval/head box) so the field decays smoothly; upsample, then cv2.remap the ROI. Set ctx.landmarks to the targets and warp ctx masks with the same map (contract). protect_background: fade the field outside the person (segmentation body/hair/face masks if present, else an expanded face oval); if ctx.background is present, fill background exposed by inward motion from it. Exactly no-op when all shape sliders are 0 (return the frame object unchanged, don't touch ctx).
Stability: landmark jitter must not cause wobble (the field must be low-frequency); measure frame-to-frame displacement-field variation on a CREMA-D clip vs. landmark jitter. Look for background bending (straight edges near the face) — create a test frame with straight vertical/horizontal lines behind the face (draw lines on the frame_720p background) and show the protection working. Check that lips/eyeliner from later effects still align (run pipeline with [ReshapeEffect(), LipEffect(), EyelinerEffect()]). Visual loop: stills + CREMA-D clips at moderate (0.3-0.5) and max settings; maximum must still look like a plausible human face (cap displacements accordingly).`,
  },
  {
    key: 'hairshape',
    prompt: `YOUR AREA: effects/hairshape.py (HairShapeEffect, name "hairshape", order 8, needs_hair_mask=True, needs_face=False), makeup_cam/background.py, tests/test_effect_hairshape.py, tests/test_background.py, HairShapeSettings, plus the minimal pipeline/runner/gui edits for the background plate described in CONTRACT2 (update their existing tests; keep all existing tests green: run tests/test_pipeline.py tests/test_runner.py tests/test_gui.py too).
background.py: capture (median of ~15 frames after a 3 s countdown; reject with a Japanese message if segmentation still sees a person/hair covering > ~3% of the frame), save/load per resolution under user_data_dir(), clear. Pipeline converts the plate to float32 once and puts it in ctx.background when sizes match.
HairShapeEffect: flyaway (アホ毛消し): find thin, low-confidence strands outside the hair's smooth silhouette (e.g. silhouette = morphological opening/closing + blur of the hair mask at a face-scale-relative radius; strands = soft hair confidence outside the silhouette and thin structures detected via a black-hat / ridge filter on luminance near the boundary) and replace them with the background plate (if available and use_background_plate) or else with a locally estimated background (inpainting on a downscaled ROI or directional fill from just outside) — strength × flyaway; must not eat into the main hair mass or the face; temporal stability (smooth masks over time). volume_top / width_sides: small smooth warp of the hair region only (push crown outward/inward, compress/expand sides) — when shrinking, fill exposed area from the plate (if no plate, cap shrinking to very small amounts and say so in the help text). No-op at defaults of 0 except flyaway (0 = no-op too).
Visual loop: CREMA-D clips (green screen backgrounds make plates easy: use the first frames? No — synthesize a plate by inpainting the person out or by taking the median of frames with the person masked; document) and the stills; zoom on hair edges. GUI: screenshot the new button + countdown overlay offscreen to ${SCRATCH}/renders/hairshape/.`,
  },
]

const built = await parallel(BUILD.map(b => () => agent(`${COMMON}\n\n${b.prompt}\nRun your tests: cd ${ROOT} && AI_MAKEUP_CAM_MODELS_DIR=${SCRATCH}/models QT_QPA_PLATFORM=offscreen ${VENV} -m pytest -q <your test files>`, { label: `build:${b.key}`, phase: 'Build', schema: REPORT })))
const buildReports = {}
BUILD.forEach((b, i) => { buildReports[b.key] = built[i] })

phase('Integrate')
const integ = await agent(`${COMMON}

YOUR AREA (integrator): all build agents are done; you may now edit ANY file under ${ROOT} (minimal, justified edits).
Build reports: ${JSON.stringify(buildReports)}
Tasks:
1. Register the new modules in effects/__init__.py BUILTIN_EFFECT_MODULES (reshape, hairshape, contour, teeth — tearbag lives in contour.py) and make sure registered_effects() ordering matches CONTRACT2. Fix mismatches. Full suite green: cd ${ROOT} && QT_QPA_PLATFORM=offscreen AI_MAKEUP_CAM_MODELS_DIR=${SCRATCH}/models ${VENV} -m pytest -q.
2. Presets: re-tune natural (就活ナチュラル: subtle contour + tearbag + clean teeth, reshape all 0, flyaway gentle), bare (すっぴん風補正: skin/under-eye/teeth/hair tidy; no color makeup, no contour), polished (しっかりメイク: clearly made-up, office-appropriate, contour/tearbag visible but soft, reshape 0), and ADD strong.json (しっかりメイク＋小顔補正: polished + moderate reshape: face_slim≈0.4, jaw≈0.3, eye_size≈0.25, nose_width≈0.3, mouth_corner≈0.4, brow_lift≈0.1 — tune visually so it still looks like the same person) and keep off.json all disabled. Render each preset on stills and on 3+ CREMA-D women (different skin tones) and Read them; iterate.
3. Preset display names: add optional "display_name" (Japanese) support in presets.py + GUI combo (built-ins show e.g. 「就活ナチュラル（標準）」), keep file names ASCII; adjust tests (test_cli preset completeness must ignore display_name).
4. GUI now has 13 tabs: make navigation comfortable (e.g. replace the QTabWidget bar with a left-side QListWidget + QStackedWidget, or group tabs: 「肌・立体感」「目元」「口元」「顔の形」「髪」 with sub-sections) — keep the auto-generation from config metadata and the tests; screenshot ${SCRATCH}/renders/integrate2/gui_*.png and Read them.
5. Performance: the pipeline had ~9 ms overhead beyond landmarks+effects (uint8↔float32 conversion of the whole frame, mirroring, clipping). Optimize safely (e.g. convert/clip only the union ROI of enabled effects, keep the rest as uint8; avoid redundant copies), measure before/after with all effects on natural and strong presets at 1280x720, report the per-stage table.
6. README.md: document the new tabs/settings (tables), 背景を撮影, presets incl. strong, and replace any line that says not to reshape the face with balanced guidance (keep reshape subtle so the online look matches how you look in person with makeup). Mention the capture guide docs/capture-guide.md and the 3D spec docs/3d-model-commission-spec.md.
7. Leave the tree clean (no caches).`, { label: 'integrate', phase: 'Integrate', schema: REPORT })

phase('Video QA')
const qa = await agent(`${COMMON}

YOUR AREA (video QA): the app is fully integrated. Be an honest, demanding reviewer (a makeup artist + a video engineer). You own: a new CLI option \`--video IN --out OUT [--compare] [--preset/--settings]\` in makeup_cam/__main__.py (process a video file frame by frame through the real pipeline with real timestamps; write mp4; compare = side-by-side BEFORE|AFTER; tests in tests/test_cli.py with a tiny synthetic video), and any QA scripts under ${SCRATCH}/qa/ (not in ROOT).
Integration report: ${JSON.stringify(integ)}
Do:
1. Prepare inputs: every CREMA-D clip in ${SCRATCH}/testvids (11 women x 4 clips) upscaled to 960x720 and padded to 1280x720. Also synthesize from 2-3 clips + frame_720p: (a) a BOW sequence (over ~2 s the frame content moves down/scales/tilts so the face drops and partly leaves the frame, holds, then returns), (b) a GLANCE-DOWN sequence (perspective squash suggesting looking down), (c) a slow head-turn by concatenating clips is not possible — instead use the clips' natural motion; and (d) a static background-with-straight-lines variant for the reshape preset.
2. Run presets natural, polished, strong through --video. Produce side-by-side mp4s (convert to H.264 with ffmpeg) in ${SCRATCH}/renders/qa/, plus contact sheets (every 5th frame, zoomed on the face) and Read many of them.
3. Metrics per preset & clip: (i) temporal flicker of the makeup delta: mean |Δ_t − Δ_{t−1}| inside the face where Δ = out − in (high-pass in time), compared to the input's own frame-to-frame change; (ii) landmark jitter; (iii) count of frames with sudden jumps (z-score spikes) of the delta; (iv) teeth/lip bleed: delta energy outside the intended masks; (v) bow handling: no pops (delta fades smoothly). Report numbers.
4. Judge honestly for each preset: would an interviewer notice? List concrete findings with severity (blocker = visible glitch in the envelope; major = noticeable unnatural look; minor), evidence paths (frames/videos), suspected cause. Include skin-tone-specific issues (dark skin ashy contour, foundation mismatch, etc.).
Run the full suite at the end.`, { label: 'video-qa', phase: 'Video QA', schema: QA_REPORT })

phase('Fix')
const fix = await agent(`${COMMON}

YOUR AREA (fixer): you may edit any file under ${ROOT}. Fix the QA findings below, highest severity first (all blockers and majors; minors when cheap). For each: reproduce with the QA tooling (${SCRATCH}/qa/ and the --video CLI), fix the root cause (no hacks that hide symptoms), add a regression test where feasible, re-run the relevant metrics and show before/after numbers and renders.
QA report: ${JSON.stringify(qa)}
Then produce the FINAL deliverables for the user (a Japanese woman; use only female clips): ${SCRATCH}/renders/qa_final/ containing, as H.264 mp4 (phone-friendly, ≤ 15 s each, side-by-side BEFORE | AFTER with small ASCII labels): natural on 3 different women (incl. the Asian actor 1072 and a dark-skinned actor), polished on 2, strong on 2, and one bow sequence; plus one PNG contact sheet per preset. Read them and describe honestly. Full suite green at the end; tree clean.`, { label: 'fix', phase: 'Fix', schema: REPORT })

return { buildReports, integ, qa, fix }

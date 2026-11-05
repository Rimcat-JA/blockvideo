import sys, time, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests")
from makeup_cam.effects import skin as K
from makeup_cam.effects.base import FrameContext
from makeup_cam.config import SkinSettings
from conftest import build_face_frame_720p
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)
frame = build_face_frame_720p().astype(np.float32) / 255
if len(sys.argv) > 1:
    import render
    from makeup_cam.tracking import FaceTracker
    fr = render.close_frame(); frame = fr.astype(np.float32) / 255
    tr = FaceTracker("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/models/face_landmarker.task", 0.0)
    for i in range(3): lm, _ = tr.detect(fr, 33 * (i + 1))
    tr.close()
seg = np.ones((720, 1280), np.float32)
st = SkinSettings(); amounts = K._Amounts.from_settings(st, 1.0)
acc = {}
def tick(name, t0):
    acc[name] = acc.get(name, 0) + time.perf_counter() - t0
    return time.perf_counter()
N = 60
for i in range(N + 3):
    if i == 3: acc.clear()
    ctx = FrameContext(720, 1280, landmarks=lm, face_skin_mask=seg)
    t = time.perf_counter()
    ws = K._workspace(lm, frame.shape); patch = frame[ws.roi.slices]; t = tick("workspace", t)
    small = K._resize(patch, ws.small_size, cv2.INTER_AREA); t = tick("downscale", t)
    lmw, s = ws.landmarks, ws.scale
    oval = K.expand_polygon(lmw[K.FACE_OVAL], -K._OVAL_SHRINK * s)
    area = K._shapes_mask(ws.small_size, [oval], sigma=K._OVAL_FEATHER * s); t = tick("oval", t)
    sw = K._segmentation_weight(ctx, ws, area); area *= sw; t = tick("seg", t)
    prot = K._protect_mask(ws); t = tick("protect", t)
    masks = K._SkinMasks(area, area * (1 - prot))
    model = K._Model(*small.shape[:2]); level = K._skin_level(small, masks.skin); t = tick("level", t)
    gain, offset, detail = K._smoothing_terms(small, ws.scale, level, amounts.texture); t = tick("gf", t)
    model.mix_toward(masks.skin * np.float32(amounts.smoothing), offset, gain, detail); t = tick("mix1", t)
    spots, clean = K._blemishes(small, ws.scale, level); t = tick("blemish", t)
    model.mix_toward(spots * masks.skin * np.float32(amounts.blemish), clean); t = tick("mix2", t)
    ue = K._under_eye_zone(ws, masks.area); t = tick("ue-zone", t)
    d = K._color_delta(small, masks, ws, amounts, st.tone_color); t = tick("color(incl ue)", t)
    model.lift += d
    out = model.render(patch, ws); t = tick("render", t)
    frame[ws.roi.slices] = out; t = tick("writeback", t)
print("roi", ws.roi, "small", ws.small_size)
tot = 0
for k, v in acc.items():
    print(f"{k:14s} {v / N * 1000:.3f} ms"); tot += v
print(f"total {tot / N * 1000:.3f} ms (ue-zone counted twice)")

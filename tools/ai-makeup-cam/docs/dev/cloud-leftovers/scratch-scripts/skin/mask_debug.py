import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.effects import skin
from makeup_cam.effects.base import FrameContext
from makeup_cam.segmentation import Segmenter
S = "/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad/"
rows = []
for name in sys.argv[1].split(","):
    u8 = cv2.imread(S + f"testimgs/{name}.png")
    img = u8.astype(np.float32) / 255
    lm = np.load(S + f"tmp_skin/lm_{name}.npy")
    seg = Segmenter(S + "models/selfie_multiclass_256x256.tflite", interval=1, background=False)
    m = seg.update(u8, 1, 1); seg.close()
    ws = skin._workspace(lm, img.shape)
    ctx = FrameContext(720, 1280, landmarks=lm, face_skin_mask=m.face_skin, hair_mask=m.hair)
    masks = skin._skin_masks(ws, ctx)
    ue = skin._under_eye_masks(ws, masks.area)
    small = skin._resize(img[ws.roi.slices], ws.small_size, cv2.INTER_AREA)
    f = 6
    def up(x): return cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    base = up(small)
    ov = base.copy()
    ov[..., 1] = ov[..., 1] * 0.4 + masks.skin.repeat(f, 0).repeat(f, 1)[: ov.shape[0], : ov.shape[1]] * 0.6 if False else ov[..., 1]
    skin_v = up(masks.skin)[..., None]
    ue_v = up(ue)[..., None]
    tint = base * (1 - 0.5 * skin_v) + np.array([0, 1, 0], np.float32) * 0.5 * skin_v
    tint = tint * (1 - 0.6 * ue_v) + np.array([1, 0, 1], np.float32) * 0.6 * ue_v
    for side in ("left", "right"):
        hull = np.zeros((0,2))
        cv2.polylines(tint, [np.round(hull).astype(np.int32)], True, (0, 0, 1), 1, cv2.LINE_AA)
    rows.append(np.hstack([base, tint]))
    print(name, "fs coverage", None if m.face_skin is None else float(m.face_skin[ws.roi.slices].mean()))
w = max(r.shape[1] for r in rows)
cv2.imwrite(S + "tmp_skin/mask_debug.png", (np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]) * 255).astype(np.uint8))

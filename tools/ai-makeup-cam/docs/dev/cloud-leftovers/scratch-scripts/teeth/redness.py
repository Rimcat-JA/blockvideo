from common import *
from makeup_cam import blend
from makeup_cam.config import LipSettings, TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import lips as lips_mod, teeth as T
rows = []
for name, idx in [("1073_IEO_HAP_HI", 46), ("1074_IEO_HAP_HI", 42), ("1029_IEO_HAP_HI", 38), ("1060_IEO_HAP_HI", 19), ("1072_IEO_HAP_HI", 33)]:
    frames, lms = load(name)
    eff, lip = T.TeethEffect(), lips_mod.LipEffect()
    for i in range(idx + 1):
        img = frames[i].astype(np.float32) / 255
        ctx = FrameContext(height=720, width=1280, landmarks=lms[i])
        img = lip.apply(img, ctx, LipSettings())
        base = img.copy(); eff.apply(img, ctx, TeethSettings())
    lm = lms[idx]; st = eff.stats; scale = regions.face_scale(lm)
    lab = blend.bgr_to_lab(base)
    ch = cv2.GaussianBlur(lab[..., 1:], (0, 0), T._CHROMA_SIGMA * scale)
    box = mouth_box(lm)
    tiles = [label(crop((base * 255).astype(np.uint8), box, (300, 175)), name[:4])]
    for k in (0.0, 0.3, 0.6):
        r = ch[..., 0] - k * ch[..., 1]
        sr = st.skin_lab[1] - k * st.skin_lab[2]
        # map: skin_r - 12 -> 0 (black) ... skin_r + 4 -> 1 (white)
        v = np.clip((r - (sr - 12)) / 16, 0, 1)
        t = crop(cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET), box, (300, 175))
        tiles.append(label(t, f"k={k} skin_r={sr:.1f}"))
    rows.append(np.hstack(tiles))
    print(name, "skin", np.round(st.skin_lab, 1), "teeth_a", round(st.teeth_a, 1))
cv2.imwrite(str(RENDERS / "redness.png"), np.vstack(rows))

import sys
from common import *
from makeup_cam import blend
from makeup_cam.config import LipSettings, TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import lips as lips_mod, teeth as T
name, idx, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
frames, lms = load(name)
eff, lip = T.TeethEffect(), lips_mod.LipEffect()
for i in range(idx + 1):
    img = frames[i].astype(np.float32) / 255
    ctx = FrameContext(height=720, width=1280, landmarks=lms[i])
    img = lip.apply(img, ctx, LipSettings())
    base = img.copy()
    eff.apply(img, ctx, TeethSettings())
lm = lms[idx]; st = eff.stats; scale = regions.face_scale(lm)
full = regions.Roi(0, 0, 1280, 720)
lab = blend.bgr_to_lab(base)
chroma = cv2.GaussianBlur(lab[..., 1:], (0, 0), T._CHROMA_SIGMA * scale)
L = lab[..., 0]
filled = cv2.morphologyEx(L, cv2.MORPH_CLOSE, T._gap_kernel(scale))
geom = regions.polygon_mask(lm[regions.LIPS_INNER], full, scale * T._feather(TeethSettings()))
f1 = T._smoothstep(L / st.teeth_l, *T._L_RATIO)
f2 = T._smoothstep(L / np.maximum(filled, 1), *T._GAP_RATIO)
f3 = (1 - T._smoothstep(chroma[..., 0], st.red_threshold(), st.red_threshold() + T._RED_WIDTH)) * (1 - T._smoothstep(chroma[..., 0] - T._HUE_B_WEIGHT * chroma[..., 1], st.hue_threshold(), st.hue_threshold() + T._HUE_WIDTH))
print("teeth_l", st.teeth_l, "teeth_a", st.teeth_a)
box = mouth_box(lm)
g = lambda m: crop(cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), box, (330, 190))
ov = (base * 255).astype(np.uint8).copy()
cv2.polylines(ov, [np.round(lm[regions.LIPS_INNER] * 4).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA, shift=2)
a_vis = T._smoothstep(L / st.teeth_l, *T._L_RATIO) * T._smoothstep(L / np.maximum(filled, 1), *T._GAP_RATIO) * f3 * geom
row1 = np.hstack([crop(ov, box, (330, 190)), label(g(geom), "geom"), label(g(f1), "L/teeth")])
row2 = np.hstack([label(g(f2), "L/filled"), label(g(f3), "not red"), label(g(a_vis), "likeness x geom")])
cv2.imwrite(str(RENDERS / f"{tag}.png"), np.vstack([row1, row2]))

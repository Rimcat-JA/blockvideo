import sys, numpy as np, cv2
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import SkinSettings
from makeup_cam.effects.skin import SkinEffect
from makeup_cam.effects.base import FrameContext
lm = np.load("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam/tests/data/face_landmarks_720p.npy").astype(np.float32)
s = regions.face_scale(lm)
SKIN = (0.55, 0.65, 0.80)
def flat():
    f = np.full((720, 1280, 3), 0.3, np.float32)
    roi = regions.Roi(0, 0, 1280, 720)
    m = regions.face_oval_mask(lm, roi, expand_px=0.3 * s)
    f[:] = f * (1 - m[..., None]) + np.array(SKIN, np.float32) * m[..., None]
    return f
def only(**kw):
    base = dict(smoothing=0.0, blemish=0.0, redness=0.0, under_eye=0.0, tone_strength=0.0, brightness=0.0)
    base.update(kw); return SkinSettings(**base)
def run(frame, st, **ctxkw):
    out = SkinEffect().apply(frame.copy(), FrameContext(720, 1280, landmarks=lm, **ctxkw), st)
    return out
# blemish: dot vs line
f = flat()
cl = regions.cheek_center(lm, "right"); cr = regions.cheek_center(lm, "left")
r = max(1, int(round(0.04 * s)))
cv2.circle(f, (int(cl[0]), int(cl[1])), r, (0.45, 0.52, 0.68), -1, cv2.LINE_AA)
p0 = (int(cr[0] - 0.3 * s), int(cr[1])); p1 = (int(cr[0] + 0.3 * s), int(cr[1]))
cv2.line(f, p0, p1, (0.45, 0.52, 0.68), max(1, r), cv2.LINE_AA)
out = run(f, only(blemish=1.0))
def lum(img, xy): return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[int(xy[1]), int(xy[0])])
base = lum(flat(), cl)
print("dot before/after deficit", base - lum(f, cl), base - lum(out, cl))
print("line before/after deficit", base - lum(f, cr), base - lum(out, cr))
# redness
f = flat(); red = f.copy()
cv2.circle(red, (int(cl[0]), int(cl[1])), int(0.25 * s), (0.50, 0.55, 0.85), -1, cv2.LINE_AA)
red = cv2.GaussianBlur(red, (0, 0), 0.05 * s)
out = run(red, only(redness=1.0))
lab = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
print("redness a* at patch before/after", lab(red)[int(cl[1]), int(cl[0])], lab(out)[int(cl[1]), int(cl[0])], "plain skin a*", lab(flat())[int(cr[1]), int(cr[0])])
# under eye
f = flat()
ind = regions.side_indices("right")
top = lm[ind.under_eye_top]; bot = lm[ind.under_eye_bottom]
mid = ((top + bot) / 2).mean(axis=0)
dark = f.copy(); cv2.ellipse(dark, (int(mid[0]), int(mid[1])), (int(0.3 * s), int(0.08 * s)), 0, 0, 360, (0.45, 0.5, 0.6), -1, cv2.LINE_AA)
dark = cv2.GaussianBlur(dark, (0, 0), 0.03 * s)
out = run(dark, only(under_eye=1.0))
print("under-eye L before/after", lab(dark)[int(mid[1]), int(mid[0])], lab(out)[int(mid[1]), int(mid[0])])
iris = lm[regions.RIGHT_IRIS_CENTER]
print("iris changed", np.abs(out - dark)[int(iris[1]) - 2:int(iris[1]) + 3, int(iris[0]) - 2:int(iris[0]) + 3].max())
# brightness / tone
for b in (0.2, -0.2):
    out = run(flat(), only(brightness=b)); print("brightness", b, lum(flat(), cr), lum(out, cr))
out = run(flat(), only(tone_strength=1.0)); print("tone", lab(flat())[int(cr[1]), int(cr[0])], lab(out)[int(cr[1]), int(cr[0])], "target", lab(np.array([[[0.75, 0.83, 0.945]]], np.float32)))
# monotonic intensity on real-ish frame
f = flat() + (np.random.default_rng(0).standard_normal((720, 1280, 3)).astype(np.float32) * 0.02)
f = np.clip(f, 0, 1)
for k in (0.25, 0.5, 1.0):
    out = SkinEffect().apply(f.copy(), FrameContext(720, 1280, landmarks=lm, intensity=k), SkinSettings())
    print("intensity", k, float(np.abs(out - f).mean()))

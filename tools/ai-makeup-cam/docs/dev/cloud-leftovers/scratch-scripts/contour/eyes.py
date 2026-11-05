"""Tear-bag close-ups: skin only | default | strong | max (+ smile clip frames)."""
import sys
from harness import *
from makeup_cam.config import TearbagSettings
tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
VARIANTS = [("skin only", None), ("default", TearbagSettings()), ("strong", TearbagSettings(strength=0.5, shadow=0.6)), ("max", TearbagSettings(strength=1.0, shadow=1.0, width=1.0))]
SOURCES = [("frame_near", None), ("1029_IEO_HAP_HI", 30), ("1072_IEO_HAP_HI", 30), ("1074_DFA_NEU_XX", 20), ("1002_IEO_HAP_HI", 40), ("1076_DFA_NEU_XX", 20)]
rows = []
for src, idx in SOURCES:
    frames = [still(src)] * 4 if idx is None else load_clip(src)[: idx + 1]
    effects = {"tearbag": C.TearbagEffect()}
    for f, ctx in contexts(frames):
        if ctx.landmarks is not None:
            made_up(f, ctx, None, TearbagSettings(), effects=effects)  # warm up smile state
    lm = ctx.landmarks
    box = regions.roi_around(lm[regions.LEFT_EYE + regions.RIGHT_EYE + regions.RIGHT_UNDER_EYE_BOTTOM + regions.LEFT_UNDER_EYE_BOTTOM], f.shape, pad=0.12)
    tiles = []
    for name, t in VARIANTS:
        img = made_up(f, ctx, None, t, effects=effects)[box.slices]
        k = 640 / img.shape[1]
        tiles.append(label(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), f"{src} {name} smile={effects['tearbag']._smile:.2f}"))
    rows.append(vstack([hstack(tiles[:2]), hstack(tiles[2:])]))
for i in range(0, len(rows), 3):
    cv2.imwrite(str(OUT / f"eyes_{tag}_{i // 3}.png"), vstack(rows[i:i + 3]))

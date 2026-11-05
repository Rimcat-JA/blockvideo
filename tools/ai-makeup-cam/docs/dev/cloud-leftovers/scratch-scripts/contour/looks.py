"""Before/after looks: skin only | default contour+tearbag | strong | x-strong (placement)."""
import sys
from harness import *
from makeup_cam.config import ContourSettings, TearbagSettings
tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
which = sys.argv[2] if len(sys.argv) > 2 else "all"
VARIANTS = [
    ("skin only", None, None),
    ("default", ContourSettings(), TearbagSettings()),
    ("x2", ContourSettings(shading=min(2 * ContourSettings().shading, 1), highlight=min(2 * ContourSettings().highlight, 1)), TearbagSettings(strength=0.5)),
    ("max", ContourSettings(shading=1.0, highlight=1.0), TearbagSettings(strength=1.0, shadow=1.0)),
]
SOURCES = [("frame_near", None), ("frame_dark", None), ("1029_DFA_NEU_XX", 20), ("1060_ITS_HAP_XX", 40), ("1074_DFA_NEU_XX", 20), ("1072_DFA_NEU_XX", 20), ("1002_DFA_NEU_XX", 20), ("1013_IEO_HAP_HI", 40)]
if which != "all":
    SOURCES = [s for s in SOURCES if s[0] in which.split(",")]
rows = []
for src, idx in SOURCES:
    frames = [still(src)] * 4 if idx is None else load_clip(src)[: idx + 1]
    for f, ctx in contexts(frames):
        pass
    tiles = [label(crop(made_up(f, ctx, c, t), ctx.landmarks, height=360), f"{src} {name}") for name, c, t in VARIANTS]
    rows.append(hstack(tiles))
for i in range(0, len(rows), 4):
    cv2.imwrite(str(OUT / f"looks_{tag}_{i // 4}.png"), vstack(rows[i:i + 4]))

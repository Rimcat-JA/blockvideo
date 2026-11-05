"""Weight maps on stills and a few clip frames (placement check)."""
import sys
from harness import *
from makeup_cam.config import ContourSettings
tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
s = ContourSettings(shading=1.0, highlight=1.0)
rows = []
for name in ["frame_near", "frame_tilt"]:
    frames = [still(name)] * 4
    for f, ctx in contexts(frames):
        pass
    m, roi = weight_map(f, ctx, s)
    rows.append(label(crop(m, ctx.landmarks, height=480), name))
for clip, idx in [("1029_IEO_HAP_HI", 30), ("1072_DFA_NEU_XX", 20), ("1060_ITS_HAP_XX", 40)]:
    frames = load_clip(clip)[: idx + 1]
    for f, ctx in contexts(frames):
        pass
    m, roi = weight_map(f, ctx, s)
    rows.append(label(crop(m, ctx.landmarks, height=480), clip))
cv2.imwrite(str(OUT / f"maps_{tag}.png"), hstack(rows))

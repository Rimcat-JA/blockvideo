"""Per-area weight maps + stroke center lines."""
import sys
from harness import *
from makeup_cam.config import ContourSettings
tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
AREAS = ["jawline", "cheek_hollow", "nose_shadow", "forehead", "nose_bridge", "cheekbone", "chin", "cupids_bow"]
def only(area):
    vals = {a: (1.0 if a == area else 0.0) for a in AREAS}
    return ContourSettings(shading=1.0, highlight=1.0, **vals)
cols = []
for src in ["frame_near", ("1029_IEO_HAP_HI", 30), ("1072_DFA_NEU_XX", 20)]:
    frames = [still(src)] * 4 if isinstance(src, str) else load_clip(src[0])[: src[1] + 1]
    for f, ctx in contexts(frames):
        pass
    tiles = []
    for area in AREAS:
        m, roi = weight_map(f, ctx, only(area))
        lm = ctx.landmarks
        scale = regions.face_scale(lm)
        stroke = next(s for s in C._STROKES if s.area == area)
        for side in (regions.SIDES if stroke.mirrored else ("right",)):
            p = C._stroke_path(stroke, lm, side, scale, 1.0)
            pts = np.round(p.points).astype(np.int32)
            cv2.polylines(m, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(label(crop(m, lm, height=300), area))
    cols.append(vstack([hstack(tiles[:4]), hstack(tiles[4:])]))
cv2.imwrite(str(OUT / f"areas_{tag}.png"), vstack(cols))

from common import *
from makeup_cam import blend
LB = SCRATCH / "tmp_lips_blush"
tiles = []
for n in ["720p", "dark", "far", "near"]:
    p = SCRATCH / "testimgs" / f"frame_{n}.png"
    if not p.exists():
        p = LB / f"frame_{n}.png"
    f = cv2.imread(str(p))
    lm = np.load(LB / f"lm_frame_{n}.npy")
    print(n, f.shape, "scale", round(regions.face_scale(lm), 1), "open", round(opening(lm), 1))
    box = mouth_box(lm)
    ov = f.copy()
    cv2.polylines(ov, [np.round(lm[regions.LIPS_INNER] * 4).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA, shift=2)
    tiles.append(np.hstack([label(crop(f, box), n), crop(ov, box)]))
cv2.imwrite(str(RENDERS / "stills_raw.png"), np.vstack(tiles))

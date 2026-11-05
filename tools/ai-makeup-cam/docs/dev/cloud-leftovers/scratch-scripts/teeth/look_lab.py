from common import *
from makeup_cam import blend

def norm(ch, lo, hi):
    return cv2.cvtColor(np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

rows = []
for a in ["1012", "1029", "1060", "1072", "1074", "1082"]:
    frames, lms = load(f"{a}_IEO_HAP_HI")
    ops = [opening(l) if np.isfinite(l).all() else 0 for l in lms]
    i = int(np.argmax(ops))
    f, lm = frames[i], lms[i]
    box = mouth_box(lm)
    ov = f.copy()
    pts = np.round(lm[regions.LIPS_INNER] * 4).astype(np.int32)
    cv2.polylines(ov, [pts], True, (0, 255, 0), 1, cv2.LINE_AA, shift=2)
    lab = blend.bgr_to_lab(f.astype(np.float32) / 255)
    tiles = [crop(f, box), crop(ov, box), crop(norm(lab[..., 0], 0, 100), box), crop(norm(lab[..., 1], -5, 40), box), crop(norm(lab[..., 2], -5, 40), box)]
    rows.append(np.hstack([label(tiles[0], a)] + tiles[1:]))
    # numbers
    roi = regions.Roi(0, 0, f.shape[1], f.shape[0])
    inner = regions.polygon_mask(lm[regions.LIPS_INNER], roi) > 0.5
    lips = regions.lips_mask(lm, roi) > 0.5
    li = lab[inner]; ll = lab[lips]
    print(a, "inner L p10/50/90", np.percentile(li[:, 0], [10, 50, 90]).round(1), "a p10/50/90", np.percentile(li[:, 1], [10, 50, 90]).round(1), "b", np.percentile(li[:, 2], [10, 50, 90]).round(1))
    print("   lips L/a/b median", np.median(ll, axis=0).round(1))
cv2.imwrite(str(RENDERS / "inspect.png"), np.vstack(rows))

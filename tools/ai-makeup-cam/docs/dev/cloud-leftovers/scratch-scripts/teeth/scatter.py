from common import *
from makeup_cam import blend

tiles = []
for a in ["1012", "1029", "1060", "1072", "1074", "1082", "1013", "1073"]:
    frames, lms = load(f"{a}_IEO_HAP_HI")
    ops = [opening(l) if np.isfinite(l).all() else 0 for l in lms]
    i = int(np.argmax(ops))
    f, lm = frames[i], lms[i]
    lab = blend.bgr_to_lab(f.astype(np.float32) / 255)
    lab[..., 1:] = cv2.GaussianBlur(lab[..., 1:], (0, 0), 2.5)
    roi = regions.Roi(0, 0, f.shape[1], f.shape[0])
    grown = regions.expand_polygon(lm[regions.LIPS_INNER], 6)
    inner = regions.polygon_mask(grown, roi) > 0.5
    px = lab[inner]
    lips = lab[regions.lips_mask(lm, roi) > 0.5]
    canvas = np.full((300, 300, 3), 255, np.uint8)
    # axes: a in [-10, 40] -> x, b in [-10, 50] -> y (up)
    def to_xy(aa, bb):
        return ((aa + 10) / 50 * 299).astype(int), (299 - (bb + 10) / 60 * 299).astype(int)
    for (L, aa, bb) in px[::2]:
        x, y = to_xy(np.array(aa), np.array(bb))
        c = int(np.clip(L * 2.55, 0, 255))
        cv2.circle(canvas, (int(x), int(y)), 1, (255 - c, 0, c), -1)
    for (L, aa, bb) in lips[::6]:
        x, y = to_xy(np.array(aa), np.array(bb))
        cv2.circle(canvas, (int(x), int(y)), 1, (0, 200, 0), -1)
    x0, y0 = to_xy(np.array(0.0), np.array(0.0))
    cv2.line(canvas, (int(x0), 0), (int(x0), 299), (128, 128, 128), 1)
    cv2.line(canvas, (0, int(y0)), (299, int(y0)), (128, 128, 128), 1)
    cv2.putText(canvas, a, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    box = mouth_box(lm)
    tiles.append(np.vstack([cv2.resize(crop(f, box), (300, 175)), canvas]))
cv2.imwrite(str(RENDERS / "scatter.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))

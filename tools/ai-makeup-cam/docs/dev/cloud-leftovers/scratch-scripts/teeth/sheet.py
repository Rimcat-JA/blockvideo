from common import *

rows = []
for a in ACTORS:
    frames, lms = load(f"{a}_IEO_HAP_HI")
    ops = [opening(l) if np.isfinite(l).all() else 0 for l in lms]
    i = int(np.argmax(ops))
    box = mouth_box(lms[i])
    rows.append(label(crop(frames[i], box), f"{a} f{i} op={ops[i]:.0f}"))
sheet = np.vstack([np.hstack(rows[i:i + 4] + [np.zeros_like(rows[0])] * (4 - len(rows[i:i + 4]))) for i in range(0, len(rows), 4)])
cv2.imwrite(str(RENDERS / "sheet_raw.png"), sheet)

"""Zoomed index maps of all mesh points in a crop (for picking landmark indices)."""
import sys
from pathlib import Path
import numpy as np, cv2

S = Path(sys.argv[1])
name, x0, y0, x1, y1, z = sys.argv[2], *map(int, sys.argv[3:7]), float(sys.argv[7])
tag = sys.argv[8]
only = None if len(sys.argv) < 10 else [int(v) for v in sys.argv[9].split(",")]
img = cv2.imread(str(S / "testimgs" / f"{name}.png"))
lm = np.load(S / "tmp_geometry" / f"{name}.npy")
crop = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
for i, (x, y) in enumerate(lm):
    if only is not None and i not in only:
        continue
    if not (x0 <= x < x1 and y0 <= y < y1):
        continue
    p = (int((x - x0) * z), int((y - y0) * z))
    cv2.circle(crop, p, 2, (0, 255, 255), -1)
    cv2.putText(crop, str(i), (p[0] + 2, p[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(crop, str(i), (p[0] + 2, p[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(S / "renders/geometry" / f"indexmap_{tag}.png"), crop)

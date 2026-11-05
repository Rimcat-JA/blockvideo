import sys
from pathlib import Path
import numpy as np, cv2
S = Path(sys.argv[1]); name = sys.argv[2]; x0,y0,x1,y1 = map(int, sys.argv[3:7]); z = float(sys.argv[7])
img = cv2.imread(str(S / "testimgs" / f"{name}.png"))
lm = np.load(S / "tmp_geometry" / f"{name}.npy")
crop = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
for x, y in lm:
    cv2.circle(crop, (int((x - x0) * z), int((y - y0) * z)), 2, (0, 255, 0), -1)
cv2.imwrite(str(S / "renders/geometry" / f"dots_{name}.png"), crop)

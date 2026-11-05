import sys
import cv2, numpy as np
from pathlib import Path
S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
img = cv2.imread(str(S/"testimgs/frame_720p.png"))
lm = np.load(S/"tmp_lips_blush/lm_frame_720p.npy")
ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(468)
x0,y0,x1,y1 = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else (500,100,820,480)
f = 6
crop = cv2.resize(img[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
for i in ids:
    x, y = (lm[i] - (x0, y0)) * f
    if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
        cv2.circle(crop, (int(x), int(y)), 3, (0,255,0), -1)
        cv2.putText(crop, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)
cv2.imwrite(sys.argv[1], crop)

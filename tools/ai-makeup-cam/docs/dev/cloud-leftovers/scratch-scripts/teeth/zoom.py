import sys
from common import *
import detail
from makeup_cam.config import TeethSettings
name, frame_idx, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
frames, lms = load(name)
detail.SIZE = (560, 300)
rows = []
for vals in [dict(whitening=0, yellow_reduction=0, evenness=0), dict(), dict(whitening=1, yellow_reduction=1, evenness=1)]:
    s = TeethSettings(**vals) if vals else TeethSettings()
    if not vals or vals.get("whitening") != 0:
        tiles = detail.process(frames, lms, frame_idx, s)
        rows.append(tiles[1])
    else:
        tiles = detail.process(frames, lms, frame_idx, TeethSettings())
        rows.append(tiles[0])
cv2.imwrite(str(RENDERS / f"{tag}.png"), np.vstack([label(rows[0], "before"), label(rows[1], "default"), label(rows[2], "max")]))

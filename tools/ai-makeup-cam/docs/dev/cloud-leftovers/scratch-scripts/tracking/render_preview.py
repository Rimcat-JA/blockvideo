import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.config import MakeupSettings
M = Path(sys.argv[1]); T = sys.argv[2]; OUT = sys.argv[3]
s = MakeupSettings(); s.general.show_landmarks = True
tiles = []
for name in ("frame_720p.png", "frame_dark.png", "frame_tilt.png", "frame_far.png"):
    p = MakeupPipeline(s, model_dir=M, effects=[])
    f = cv2.imread(f"{T}/{name}")
    for i in range(3): r = p.process(f, 33 * i)
    p.close()
    lm = r.landmarks
    print(name, r.face_found, {k: round(v, 1) for k, v in r.timings_ms.items()})
    # preview is mirrored: crop around the mirrored face center
    cx = f.shape[1] - 1 - lm[:, 0].mean(); cy = lm[:, 1].mean(); half = int(np.ptp(lm[:, 1]) * 0.75)
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    tiles.append(cv2.resize(r.preview[y0:y0 + 2 * half, x0:x0 + 2 * half], (480, 480), interpolation=cv2.INTER_NEAREST))
cv2.imwrite(OUT, np.hstack(tiles))

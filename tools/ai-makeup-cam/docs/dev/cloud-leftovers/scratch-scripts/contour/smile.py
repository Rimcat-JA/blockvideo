import numpy as np
from clips import *
from makeup_cam import regions
def metrics(lm):
    s = regions.face_scale(lm)
    out = []
    for side in ("right", "left"):
        si = regions.side_indices(side)
        lash = lm[si.eye_lower]; top = lm[si.under_eye_top]; bot = lm[si.under_eye_bottom]
        a, b = lash[0], lash[-1]
        ax = (b - a) / np.linalg.norm(b - a); n = np.array([-ax[1], ax[0]])
        if n[1] < 0: n = -n
        sag = float(np.dot(lash[4] - (a + b) / 2, n)) / s
        gap = float(np.mean(np.linalg.norm(top[2:7] - lash[2:7], axis=1))) / s
        gapb = float(np.mean(np.linalg.norm(bot[2:7] - lash[2:7], axis=1))) / s
        up = lm[si.eye_upper]
        open_ = float(np.linalg.norm(up[4] - lash[4])) / np.linalg.norm(b - a)
        out.append((sag, gap, gapb, open_))
    o = np.array(out).mean(0)
    mouth = np.linalg.norm(lm[291] - lm[61]) / s
    return [*o, mouth]
for a in ["1002", "1029", "1060", "1072", "1076"]:
    for c in ("DFA_NEU_XX", "IEO_HAP_HI"):
        L = np.load(TMP / f"lm_{a}_{c}.npy")
        M = np.array([metrics(lm) for lm in L])
        print(a, c, "sag gap gapb open mouth")
        for i in range(0, len(M), 6):
            print("   ", i, " ".join(f"{v:.3f}" for v in M[i]))

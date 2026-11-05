"""Per-frame curves (opening, presence, refs, mean change) + consecutive strips."""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import *
from makeup_cam import blend
from makeup_cam.config import LipSettings, TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import lips as lips_mod
from makeup_cam.effects import teeth as T


def analyse(name, s, strip_range=None, step=2):
    frames, lms = load(name)
    eff, lip = T.TeethEffect(), lips_mod.LipEffect()
    rec = {k: [] for k in ["op", "pres", "gate", "TL", "TA", "SA", "dL", "db", "ms"]}
    tiles_b, tiles_a = [], []
    for i, (f, lm) in enumerate(zip(frames, lms)):
        img = f.astype(np.float32) / 255
        ctx = FrameContext(height=720, width=1280, landmarks=lm)
        img = lip.apply(img, ctx, LipSettings())
        base = img.copy()
        t0 = time.perf_counter()
        out = eff.apply(img, ctx, s)
        rec["ms"].append((time.perf_counter() - t0) * 1000)
        sc = regions.face_scale(lm)
        op = T._opening(lm) / sc
        st = eff.stats
        pres = eff._presence if eff._presence is not None else 0
        rec["op"].append(op); rec["pres"].append(pres)
        rec["gate"].append(pres * T._smoothstep_scalar(op, *T._OPEN_RAMP) if op > T._OPEN_RAMP[0] else 0)
        rec["TL"].append(st.teeth_l if st else np.nan); rec["TA"].append(st.teeth_a if st else np.nan)
        rec["SA"].append(st.skin_lab[1] if st else np.nan)
        full = regions.Roi(0, 0, 1280, 720)
        inner = regions.polygon_mask(lm[regions.LIPS_INNER], full) > 0.5
        if inner.sum() > 0:
            lb, la = blend.bgr_to_lab(base)[inner], blend.bgr_to_lab(out)[inner]
            rec["dL"].append(float((la[:, 0] - lb[:, 0]).mean())); rec["db"].append(float((la[:, 2] - lb[:, 2]).mean()))
        else:
            rec["dL"].append(0); rec["db"].append(0)
        if strip_range and strip_range[0] <= i < strip_range[1] and (i - strip_range[0]) % step == 0:
            box = mouth_box(lm)
            tiles_b.append(label(crop((base * 255).round().astype(np.uint8), box, (200, 117)), f"f{i}"))
            tiles_a.append(crop((np.clip(out, 0, 1) * 255).round().astype(np.uint8), box, (200, 117)))
    return rec, tiles_b, tiles_a


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--w", type=float, default=0.3)
    p.add_argument("--y", type=float, default=0.4)
    p.add_argument("--e", type=float, default=0.3)
    p.add_argument("--tag", default="temporal")
    p.add_argument("--strip", default="")
    a = p.parse_args()
    s = TeethSettings(whitening=a.w, yellow_reduction=a.y, evenness=a.e)
    fig, axes = plt.subplots(len(a.names), 1, figsize=(11, 2.6 * len(a.names)), squeeze=False)
    strips = []
    sr = tuple(int(v) for v in a.strip.split(":")) if a.strip else None
    for ax, n in zip(axes[:, 0], a.names):
        rec, tb, ta = analyse(n, s, sr)
        if tb:
            strips += [np.hstack(tb[:10]), np.hstack(ta[:10])]
        x = np.arange(len(rec["op"]))
        ax.plot(x, np.array(rec["op"]) * 100, label="open x100", color="0.6")
        ax.plot(x, np.array(rec["gate"]) * 20, label="gate x20", color="k")
        ax.plot(x, rec["TL"], label="teeth L", color="tab:blue")
        ax.plot(x, rec["TA"], label="teeth a", color="tab:red")
        ax.plot(x, rec["SA"], label="skin a", color="tab:orange", ls="--")
        ax.plot(x, np.array(rec["dL"]) * 5, label="dL x5", color="tab:green")
        ax.plot(x, np.array(rec["db"]) * 5, label="db x5", color="tab:purple")
        ax.set_title(f"{n}  ms med {np.median(rec['ms']):.2f} max {max(rec['ms']):.2f}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=7, loc="upper left")
    fig.tight_layout(); fig.savefig(RENDERS / f"{a.tag}.png", dpi=80)
    if strips:
        w = max(r.shape[1] for r in strips)
        cv2.imwrite(str(RENDERS / f"{a.tag}_strip.png"), np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in strips]))

"""Apply lips + teeth to a clip; write before/after mouth strips and print stats."""
import argparse
import importlib
import time

from common import *
from makeup_cam.config import LipSettings, TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import lips as lips_mod
from makeup_cam.effects import teeth as teeth_mod


def run(name, values, step=5, lips=True, verbose=False, tag="", cols=None):
    frames, lms = load(name)
    eff = teeth_mod.TeethEffect()
    lip = lips_mod.LipEffect()
    s = TeethSettings(**values)
    befores, afters, info = [], [], []
    times = []
    for i, (f, lm) in enumerate(zip(frames, lms)):
        img = f.astype(np.float32) / 255
        if not np.isfinite(lm).all():
            eff.reset()
            continue
        ctx = FrameContext(height=720, width=1280, landmarks=lm)
        if lips:
            img = lip.apply(img, ctx, LipSettings())
        base = img.copy()
        t0 = time.perf_counter()
        out = eff.apply(img, ctx, s)
        times.append((time.perf_counter() - t0) * 1000)
        st = eff.stats
        if verbose and st is not None:
            print(i, f"op={teeth_mod._opening(lm)/regions.face_scale(lm):.3f} T={st.teeth_l:.1f} A={st.teeth_a:.1f} skin={tuple(round(v,1) for v in st.skin_lab)} pres={st.presence():.2f} neutral={tuple(round(v,1) for v in st.neutral())}")
        if i % step == 0:
            box = mouth_box(lm)
            befores.append(label(crop((base * 255).round().astype(np.uint8), box), f"{name[:4]} f{i}"))
            afters.append(crop((np.clip(out, 0, 1) * 255).round().astype(np.uint8), box))
    print(name, "ms median/max", np.round(np.median(times), 2), np.round(max(times), 2))
    return befores, afters


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--w", type=float, default=0.3)
    p.add_argument("--y", type=float, default=0.4)
    p.add_argument("--e", type=float, default=0.3)
    p.add_argument("--f", type=float, default=0.5)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--max", type=int, default=8)
    p.add_argument("--tag", default="strip")
    p.add_argument("-v", action="store_true")
    p.add_argument("--nolips", action="store_true")
    a = p.parse_args()
    vals = dict(whitening=a.w, yellow_reduction=a.y, evenness=a.e, feather=a.f)
    rows = []
    for n in a.names:
        if "_" not in n:
            n = f"{n}_IEO_HAP_HI"
        b, af = run(n, vals, a.step, not a.nolips, a.v)
        b, af = b[: a.max], af[: a.max]
        rows.append(np.hstack(b))
        rows.append(np.hstack(af))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(RENDERS / f"{a.tag}.png"), np.vstack(rows))

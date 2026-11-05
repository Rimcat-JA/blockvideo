import sys, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import tracking
rng = np.random.default_rng(0)
iod = 84.0
base = rng.uniform(300, 700, (478, 3))
def run(params, xs, noise=0.7):
    f = tracking.OneEuroFilter(params.min_cutoff, params.beta / iod, params.d_cutoff)
    errs = []
    for i, x in enumerate(xs):
        gt = base + (x, 0, 0)
        out = f(gt + rng.normal(0, noise, gt.shape), i / 30)
        if i > 10: errs.append(np.abs(out - gt)[:, :2].mean())
    return np.mean(errs), np.percentile(errs, 95)
static = np.zeros(120)
slow = 250 * np.sin(2 * np.pi * np.arange(150) / 30 / 3.0)
fast = 250 * np.sin(2 * np.pi * np.arange(150) / 30 / 1.5)
for lo_hi_beta in ((12, 2), (20, 4), (30, 6)):
    tracking._BETA_RANGE = lo_hi_beta
    for s in (0.25, 0.5, 0.75, 1.0):
        p = tracking.smoothing_params(s)
        st = run(p, static); sl = run(p, slow); fa = run(p, fast)
        print(f"beta{lo_hi_beta} s={s:.2f} mc={p.min_cutoff:.2f} b={p.beta:.1f}  rest_err={st[0]:.2f}  slow={sl[0]:.2f}/{sl[1]:.2f}  fast={fa[0]:.2f}/{fa[1]:.2f}")
print("raw rest err", np.mean(np.abs(rng.normal(0, .7, 1000))))

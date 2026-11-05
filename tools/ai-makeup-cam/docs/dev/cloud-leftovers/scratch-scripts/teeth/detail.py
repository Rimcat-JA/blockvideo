"""Big before/after/mask crops at the widest opening of each clip (+ stills)."""
import argparse

from common import *
from makeup_cam.config import LipSettings, TeethSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.effects import lips as lips_mod
from makeup_cam.effects import teeth as teeth_mod

CAPTURE = {}
_orig = teeth_mod._make_up


def _spy(lab, chroma, area, stats, levels, scale):
    L = lab[..., 0]
    filled = cv2.morphologyEx(L, cv2.MORPH_CLOSE, teeth_mod._gap_kernel(scale), borderType=cv2.BORDER_REPLICATE)
    not_red = (1.0 - teeth_mod._smoothstep(chroma[..., 0], stats.red_threshold(), stats.red_threshold() + teeth_mod._RED_WIDTH)) * (1 - teeth_mod._smoothstep(chroma[..., 0] - teeth_mod._HUE_B_WEIGHT * chroma[..., 1], stats.hue_threshold(), stats.hue_threshold() + teeth_mod._HUE_WIDTH))
    like = teeth_mod._smoothstep(L / stats.teeth_l, *teeth_mod._L_RATIO) * not_red * teeth_mod._smoothstep(L / np.maximum(filled, 1), *teeth_mod._GAP_RATIO)
    CAPTURE["mask"] = like * area
    return _orig(lab, chroma, area, stats, levels, scale)


teeth_mod._make_up = _spy
SIZE = (330, 190)


def process(frames, lms, idx, s, lips=True):
    eff = teeth_mod.TeethEffect()
    lip = lips_mod.LipEffect()
    for i in range(idx + 1):  # warm the temporal state
        lm = lms[i]
        if not np.isfinite(lm).all():
            continue
        img = frames[i].astype(np.float32) / 255
        ctx = FrameContext(height=img.shape[0], width=img.shape[1], landmarks=lm)
        if lips:
            img = lip.apply(img, ctx, LipSettings())
        base = img.copy()
        CAPTURE.clear()
        out = eff.apply(img, ctx, s)
    lm = lms[idx]
    box = mouth_box(lm)
    to8 = lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    mask_full = np.zeros(base.shape[:2], np.float32)
    if "mask" in CAPTURE:
        scale = regions.face_scale(lm)
        margin = teeth_mod._FEATHER_REACH * teeth_mod._feather(s) * scale + teeth_mod._gap_kernel_size(scale) / 2.0 + 2.0
        roi = regions.roi_around(lm[regions.LIPS_INNER], base.shape, 0.0, margin)
        mask_full[roi.slices] = cv2.resize(CAPTURE["mask"], (roi.w, roi.h))
    diff = np.clip(np.abs(out - base).max(axis=2) * 8, 0, 1)
    return [crop(to8(base), box, SIZE), crop(to8(out), box, SIZE),
            crop(cv2.cvtColor(to8(mask_full), cv2.COLOR_GRAY2BGR), box, SIZE),
            crop(cv2.applyColorMap(to8(diff), cv2.COLORMAP_INFERNO), box, SIZE)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--w", type=float, default=0.3)
    p.add_argument("--y", type=float, default=0.4)
    p.add_argument("--e", type=float, default=0.3)
    p.add_argument("--f", type=float, default=0.5)
    p.add_argument("--tag", default="detail")
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--nolips", action="store_true")
    a = p.parse_args()
    s = TeethSettings(whitening=a.w, yellow_reduction=a.y, evenness=a.e, feather=a.f)
    rows = []
    for n in a.names:
        if n.startswith("still:"):
            k = n.split(":")[1]
            path = SCRATCH / "testimgs" / f"frame_{k}.png"
            if not path.exists():
                path = SCRATCH / "tmp_lips_blush" / f"frame_{k}.png"
            frames = cv2.imread(str(path))[None]
            lms = np.load(SCRATCH / "tmp_lips_blush" / f"lm_frame_{k}.npy")[None]
            idx = 0
        else:
            if "_" not in n:
                n = f"{n}_IEO_HAP_HI"
            frames, lms = load(n)
            ops = [opening(l) if np.isfinite(l).all() else 0 for l in lms]
            idx = a.frame if a.frame >= 0 else int(np.argmax(ops))
        # stills: repeat the frame to build up the temporal state
        if len(frames) == 1:
            frames = np.repeat(frames, 4, axis=0); lms = np.repeat(lms, 4, axis=0); idx = 3
        tiles = process(frames, lms, idx, s, not a.nolips)
        tiles[0] = label(tiles[0], f"{n[:10]} f{idx}")
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(RENDERS / f"{a.tag}.png"), np.vstack(rows))

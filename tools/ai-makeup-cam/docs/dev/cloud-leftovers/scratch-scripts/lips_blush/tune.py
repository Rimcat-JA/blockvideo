"""Visual tuning: run the real pipeline with only the lips / blush effects and
save zoomed before/after crops to renders/lips_blush/.

usage: python tune.py lips|blush [tag]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam import regions
from makeup_cam.config import MakeupSettings
from makeup_cam.pipeline import MakeupPipeline

S = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = S / "renders" / "lips_blush"
TMP = S / "tmp_lips_blush"
IMAGES = ["frame_720p", "frame_dark", "frame_tilt", "frame_far", "frame_near"]


def load(name: str) -> np.ndarray:
    if name == "frame_near":  # synthetic close-up (webcam-typical face size)
        src = cv2.imread(str(S / "testimgs" / "frame_720p.png"))
        big = cv2.resize(src, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(big[0:720, 560:1840])
    return cv2.imread(str(S / "testimgs" / f"{name}.png"))


def variants(effect: str) -> list[tuple[str, dict]]:
    if effect == "lips":
        return [
            ("default", {}),
            ("gloss_berry", {"color": "#8E2F45", "opacity": 0.8, "finish": "gloss", "gloss": 0.9, "feather": 0.2}),
            ("matte_nude", {"color": "#D9978A", "opacity": 0.85, "finish": "matte", "feather": 0.8, "outline": 0.15}),
            ("red_max", {"color": "#C8102E", "opacity": 1.0, "finish": "satin", "gloss": 1.0, "feather": 1.0, "outline": -0.2}),
        ]
    return [
        ("default", {}),
        ("round_strong", {"shape": "round", "opacity": 0.6, "size": 1.3, "softness": 0.3}),
        ("lifted_strong", {"shape": "lifted", "opacity": 0.7, "size": 1.6, "offset_x": 0.5, "offset_y": -0.5, "softness": 1.0}),
        ("oval_inner_low", {"shape": "oval", "opacity": 0.8, "size": 1.6, "offset_x": -0.5, "offset_y": 0.5, "softness": 0.0}),
    ]


def crop_box(lm: np.ndarray, effect: str, shape) -> regions.Roi:
    if effect == "lips":
        return regions.roi_around(lm[regions.LIPS_OUTER], shape, pad=0.35)
    pts = lm[regions.FACE_OVAL]
    return regions.roi_around(pts, shape, pad=0.05)


def zoom(img: np.ndarray, box: regions.Roi, height: int) -> np.ndarray:
    crop = img[box.slices]
    f = height / crop.shape[0]
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    effect = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    with_masks = "masks" in tag
    from makeup_cam.effects.base import Effect
    if effect == "lips":
        from makeup_cam.effects.lips import LipEffect as Cls
    else:
        from makeup_cam.effects.blush import BlushEffect as Cls

    class MaskProbe(Effect):  # named "skin" so the pipeline runs segmentation
        name = "skin"
        order = 1

        def apply(self, frame, ctx, settings):
            return frame

    effects = [Cls()] + ([MaskProbe()] if with_masks else [])
    height = 300 if effect == "lips" else 420
    timings: dict[str, list[float]] = {}
    for name in IMAGES:
        img = load(name)
        base = MakeupSettings()
        for section in ("skin", "eyeshadow", "eyeliner", "brows", "hair"):
            getattr(base, section).enabled = with_masks and section == "skin"
        pipe = MakeupPipeline(base, model_dir=S / "models", effects=effects)
        ts = 0
        for _ in range(8):
            ts += 33
            res = pipe.process(img, ts)
        lm = res.landmarks
        box = crop_box(lm, effect, img.shape)
        rows = []
        for vname, values in variants(effect):
            s = MakeupSettings.from_dict(base.to_dict())
            for k, v in values.items():
                setattr(getattr(s, effect), k, v)
            pipe.set_settings(s)
            ts += 33
            res = pipe.process(img, ts)
            timings.setdefault(name, []).append(res.timings_ms.get(effect, 0.0))
            before = label(zoom(img, box, height), f"{name} before")
            after = label(zoom(res.output, box, height), f"{vname}")
            diff = cv2.absdiff(res.output, img).astype(np.float32)
            diffz = label(zoom(np.clip(diff * 6, 0, 255).astype(np.uint8), box, height), "diff x6")
            rows.append(np.hstack([before, after, diffz]))
        pipe.close()
        sheet = np.vstack(rows)
        path = OUT / f"{effect}_{name}{('_' + tag) if tag else ''}.png"
        cv2.imwrite(str(path), sheet)
        print("wrote", path, sheet.shape)
    for k, v in timings.items():
        print(k, "effect ms", np.round(v, 2))


if __name__ == "__main__":
    main()

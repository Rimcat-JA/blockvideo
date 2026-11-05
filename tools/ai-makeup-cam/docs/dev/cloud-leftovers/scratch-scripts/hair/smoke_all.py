"""Run the real pipeline with all built-in effects (background segmentation) and check the hair effect never fails."""
import logging, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(Path(__file__).parent))
from makeup_cam.config import MakeupSettings
from makeup_cam.pipeline import MakeupPipeline
from render import load, S

records = []
class Grab(logging.Handler):
    def emit(self, record): records.append(record.getMessage())
logging.getLogger("makeup_cam").addHandler(Grab())
settings = MakeupSettings()
settings.hair.color_strength = 0.5
pipe = MakeupPipeline(settings, model_dir=S / "models")
hair_ms = []
for name in ["frame_720p", "actor_1072", "frame_far"]:
    frame = load(name)
    for i in range(20):
        res = pipe.process(frame, 1000 * (len(hair_ms) + 1) // 30 + i * 33 + len(hair_ms) * 40)
        if "hair" in res.timings_ms:
            hair_ms.append(res.timings_ms["hair"])
pipe.close()
print("hair runs", len(hair_ms), "median ms", float(np.median(hair_ms)) if hair_ms else None)
print("errors", [r for r in records if "hair" in r][:3])

"""Write the built-in preset JSON files from compact overrides (full files, all fields)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
from makeup_cam.config import MakeupSettings, SECTIONS

OUT = Path(sys.argv[1])
OVERRIDES = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
OUT.mkdir(parents=True, exist_ok=True)
for name, sections in OVERRIDES.items():
    data = MakeupSettings().to_dict()
    for section, values in sections.items():
        for key, value in values.items():
            assert key in data[section], (name, section, key)
            data[section][key] = value
    settings = MakeupSettings.from_dict(data)
    assert settings.to_dict() == data, f"{name}: clamped/invalid value"
    (OUT / f"{name}.json").write_text(settings.to_json() + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.json")

"""Load portable JSON configurations; resolve local paths from the project root."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    if "path/to/your" in value:
        raise ValueError(f"Replace the placeholder '{value}' in your configuration before running.")
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def load_config(path):
    config = json.loads(Path(path).read_text())
    for name in ("train_data", "val_data", "data"):
        if name in config:
            for key in ("root", "split", "albedo_root", "relative_depth_root"):
                if key in config[name]:
                    config[name][key] = local_path(config[name][key])
    config["synthesis"]["bank"] = local_path(config["synthesis"]["bank"])
    if config.get("model", {}).get("backbone_weights"):
        config["model"]["backbone_weights"] = local_path(config["model"]["backbone_weights"])
    return config

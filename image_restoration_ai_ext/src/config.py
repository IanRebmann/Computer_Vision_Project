from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class AppConfig:
    device: str
    models: Dict[str, str]
    runtime: Dict[str, Any]
    defaults: Dict[str, Any]

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return AppConfig(
        device=cfg.get("device", "cpu"),
        models=cfg.get("models", {}),
        runtime=cfg.get("runtime", {}),
        defaults=cfg.get("defaults", {}),
    )

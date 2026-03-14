import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_cfg():
    env_path = os.environ.get("VOICECHAT_CONFIG", "").strip()
    if env_path:
        cfg_path = Path(env_path).expanduser()
    else:
        cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        example = ROOT / "config" / "config.example.yaml"
        raise RuntimeError(f"config not found: {cfg_path}. copy {example} to config/config.yaml or set VOICECHAT_CONFIG")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("VOICECHAT_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return ROOT / "config" / "config.yaml"


def load_cfg(path: str | Path | None = None) -> dict:
    cfg_path = resolve_config_path(path)
    if not cfg_path.exists():
        example = ROOT / "config" / "config.example.yaml"
        raise RuntimeError(
            f"config not found: {cfg_path}. copy {example} to config/config.yaml or set VOICECHAT_CONFIG"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise RuntimeError(f"config root must be a mapping: {cfg_path}")

    return cfg


def dump_cfg(cfg: dict) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)

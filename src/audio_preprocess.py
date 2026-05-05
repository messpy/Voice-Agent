from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_whisper_preprocess_cfg(root: Path = ROOT) -> dict:
    cfg_path = root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    audio_pipeline = cfg.get("audio_pipeline", {})
    if not isinstance(audio_pipeline, dict):
        return {}
    section = audio_pipeline.get("whisper_preprocess", {})
    return section if isinstance(section, dict) else {}


def prepare_whisper_audio(src: Path, dst: Path, *, root: Path = ROOT) -> Path:
    cfg = load_whisper_preprocess_cfg(root)
    if not bool(cfg.get("enabled", False)):
        return src

    sample_rate = str(cfg.get("sample_rate", 16000))
    channels = str(cfg.get("channels", 1))
    codec = str(cfg.get("codec", "pcm_s16le")).strip() or "pcm_s16le"
    af = str(cfg.get("af", "")).strip()

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        fallback = dst.with_name(dst.stem + "_preprocessed" + dst.suffix)
        dst = fallback

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        codec,
        "-ar",
        sample_rate,
        "-ac",
        channels,
    ]
    if af:
        cmd += ["-af", af]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        shutil.copyfile(src, dst)
        return dst
    return dst

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from src.llm_api import resolve_llm_config
from src.transcript_correction import build_transcript_correction_context, correct_transcript, normalize_text
from tools.cohere_transcribe import ROOT, ffmpeg_normalize
from tools.timed_record_transcribe import transcribe_whisper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local file transcription with optional AI correction")
    parser.add_argument("input", type=Path, help="input audio file")
    parser.add_argument("--model", type=Path, help="override whisper model path")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/voicechat/transcribe_file_local"))
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ai-correct", action="store_true", help="apply Ollama-based transcript correction")
    return parser.parse_args()


def load_cfg() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = load_cfg()
    whisper_cfg = cfg.get("whisper", {})
    storage_cfg = cfg.get("storage", {})
    sqlite_cfg = storage_cfg.get("sqlite", {})

    whisper_bin = Path(os.environ.get("VOICECHAT_WHISPER_BIN", "").strip() or whisper_cfg["bin"])
    whisper_model = args.model or Path(os.environ.get("VOICECHAT_WHISPER_MODEL", "").strip() or whisper_cfg["model"])
    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_wav = out_dir / f"{args.tag}_normalized.wav"
    raw_prefix = out_dir / f"{args.tag}_raw"
    raw_txt_path = out_dir / f"{args.tag}_raw.txt"
    corrected_path = out_dir / f"{args.tag}_corrected.txt"
    meta_path = out_dir / f"{args.tag}.json"

    ffmpeg_normalize(args.input.expanduser(), normalized_wav, {"ffmpeg": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}})
    raw_text, elapsed_sec = transcribe_whisper(
        whisper_bin=whisper_bin,
        whisper_model=whisper_model,
        wav=normalized_wav,
        out_prefix=raw_prefix,
        lang=args.lang,
        threads=args.threads,
    )
    raw_txt_path.write_text(raw_text + "\n", encoding="utf-8")

    corrected_text = raw_text
    rag_context = ""
    if args.ai_correct:
        llm_cfg = resolve_llm_config(cfg)
        sqlite_path = ROOT / str(sqlite_cfg.get("path", "voicechat.db"))
        rag_context = build_transcript_correction_context(
            root=ROOT,
            cfg=cfg,
            raw_text=raw_text,
            sqlite_path=sqlite_path,
        )
        corrected_text = correct_transcript(
            llm_cfg=llm_cfg,
            raw_text=raw_text,
            rag_context=rag_context,
        )
    corrected_path.write_text(corrected_text + "\n", encoding="utf-8")

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(args.input),
        "normalized_wav": str(normalized_wav),
        "model": str(whisper_model),
        "elapsed_sec": elapsed_sec,
        "raw": normalize_text(raw_text),
        "corrected": normalize_text(corrected_text),
        "ai_correct": bool(args.ai_correct),
        "rag_context": rag_context,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("OK: transcription complete")
    print(f"RAW: {raw_text}")
    print(f"CORRECTED: {corrected_text}")
    print(f"RAW_PATH: {raw_txt_path}")
    print(f"CORRECTED_PATH: {corrected_path}")
    print(f"META_PATH: {meta_path}")


if __name__ == "__main__":
    main()

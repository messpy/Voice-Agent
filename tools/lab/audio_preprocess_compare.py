#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from tools.cohere_transcribe import ROOT
from tools.timed_record_transcribe import transcribe_whisper


PRESETS: list[dict[str, str]] = [
    {"id": "raw_norm", "label": "normalize only", "af": ""},
    {"id": "voice_band", "label": "highpass+lowpass", "af": "highpass=f=120,lowpass=f=3800"},
    {"id": "denoise_light", "label": "afftdn light", "af": "afftdn=nr=10:nf=-28"},
    {"id": "denoise_band", "label": "afftdn + bandpass", "af": "afftdn=nr=12:nf=-30,highpass=f=120,lowpass=f=3800"},
    {"id": "voice_focus", "label": "bandpass + dynaudnorm", "af": "highpass=f=120,lowpass=f=3800,dynaudnorm=f=150:g=9"},
    {"id": "music_suppress", "label": "strong denoise + voice focus", "af": "afftdn=nr=18:nf=-35,highpass=f=140,lowpass=f=3500,dynaudnorm=f=200:g=11"},
    {"id": "anlmdn_band", "label": "anlmdn + bandpass", "af": "anlmdn=s=0.00002:p=0.002:r=0.01,highpass=f=120,lowpass=f=3800"},
    {"id": "compand_band", "label": "bandpass + compand", "af": "highpass=f=120,lowpass=f=3800,compand=attacks=0.02:decays=0.2:points=-80/-80|-24/-12|0/-3"},
    {"id": "speech_norm_band", "label": "bandpass + speechnorm", "af": "highpass=f=120,lowpass=f=3800,speechnorm=e=12.5:r=0.0001:l=1"},
    {"id": "anlmdn_speech", "label": "anlmdn + speechnorm", "af": "anlmdn=s=0.00002:p=0.002:r=0.01,highpass=f=120,lowpass=f=3600,speechnorm=e=10:r=0.0001:l=1"},
    {"id": "compand_denoise", "label": "afftdn + compand", "af": "afftdn=nr=14:nf=-30,highpass=f=120,lowpass=f=3800,compand=attacks=0.02:decays=0.2:points=-80/-80|-24/-10|0/-2"},
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\u3000", " ")).strip()


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def char_ngrams(text: str, *, min_n: int = 2, max_n: int = 3) -> set[str]:
    compact = compact_text(text)
    grams: set[str] = set()
    for size in range(min_n, max_n + 1):
        if len(compact) < size:
            continue
        for idx in range(len(compact) - size + 1):
            grams.add(compact[idx : idx + size])
    return grams


def score_chunk(query: str, chunk: str) -> float:
    qgrams = char_ngrams(query)
    cgrams = char_ngrams(chunk)
    if not qgrams or not cgrams:
        return 0.0
    overlap = len(qgrams & cgrams)
    if overlap == 0:
        return 0.0
    coverage = overlap / max(1, len(qgrams))
    density = overlap / max(1, len(cgrams))
    return round((coverage * 0.8) + (density * 0.2), 4)


def score_ratio(query: str, chunk: str) -> float:
    return round(SequenceMatcher(None, compact_text(query), compact_text(chunk)).ratio(), 4)


def load_cfg() -> dict[str, Any]:
    path = ROOT / "config" / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ffmpeg preprocessing presets for one whisper model")
    parser.add_argument("input", type=Path, help="input audio file")
    parser.add_argument("--reference-text", help="expected transcript for similarity scoring")
    parser.add_argument("--reference-file", type=Path, help="file that contains expected transcript")
    parser.add_argument("--model", type=Path, help="override whisper model path")
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preset-ids", nargs="+", help="subset of preset ids")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/voicechat/preprocess_compare"))
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def preprocess_audio(src: Path, dst: Path, af: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
    ]
    if af:
        cmd += ["-af", af]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout or f"ffmpeg failed rc={proc.returncode}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg()
    whisper_cfg = cfg.get("whisper", {})
    whisper_bin = Path(os.environ.get("VOICECHAT_WHISPER_BIN", "").strip() or whisper_cfg["bin"])
    whisper_model = args.model or Path(os.environ.get("VOICECHAT_WHISPER_MODEL", "").strip() or whisper_cfg["model"])

    reference_text = ""
    if args.reference_text:
        reference_text = normalize_text(args.reference_text)
    elif args.reference_file:
        reference_text = normalize_text(args.reference_file.expanduser().read_text(encoding="utf-8"))

    presets = PRESETS
    if args.preset_ids:
        wanted = set(args.preset_ids)
        presets = [row for row in PRESETS if row["id"] in wanted]
    if not presets:
        raise SystemExit("NG: no presets selected")

    out_dir = args.out_dir.expanduser() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for preset in presets:
        wav = out_dir / f"{preset['id']}.wav"
        txt = out_dir / f"{preset['id']}.txt"
        preprocess_audio(args.input.expanduser(), wav, preset["af"])
        out_prefix = out_dir / f"{preset['id']}_raw"
        text, elapsed_sec = transcribe_whisper(
            whisper_bin=whisper_bin,
            whisper_model=whisper_model,
            wav=wav,
            out_prefix=out_prefix,
            lang=args.lang,
            threads=args.threads,
        )
        txt.write_text(text + "\n", encoding="utf-8")
        item: dict[str, Any] = {
            "id": preset["id"],
            "label": preset["label"],
            "af": preset["af"],
            "wav_path": str(wav),
            "transcript_path": str(txt),
            "elapsed_sec": elapsed_sec,
            "transcript": text,
        }
        if reference_text:
            item["reference_similarity"] = score_chunk(reference_text, text)
            item["reference_ratio"] = score_ratio(reference_text, text)
        results.append(item)

    if reference_text:
        ranked = sorted(results, key=lambda row: (float(row.get("reference_similarity", 0.0)), float(row.get("reference_ratio", 0.0))), reverse=True)
    else:
        ranked = sorted(results, key=lambda row: float(row["elapsed_sec"]))

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(args.input),
        "model": str(whisper_model),
        "reference_text": reference_text,
        "best_preset_id": ranked[0]["id"] if ranked else "",
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "best_preset_id": summary["best_preset_id"]}, ensure_ascii=False))
    for row in ranked:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

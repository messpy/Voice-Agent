#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from tools.cohere_transcribe import ROOT, ffmpeg_normalize
from tools.timed_record_transcribe import transcribe_whisper


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


def load_catalog() -> dict[str, Any]:
    path = ROOT / "config" / "whisper_models.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_existing_path(candidate_paths: list[str]) -> Path | None:
    for raw in candidate_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
    return None


def resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def transcribe_moonshine(
    *,
    python_bin: Path,
    model_name: str,
    wav: Path,
    transcript_path: Path,
    cache_dir: Path | None,
    tmp_dir: Path | None,
) -> tuple[str, float]:
    env = os.environ.copy()
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(cache_dir)
    if tmp_dir:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmp_dir)
    script = """
import json
import time
from pathlib import Path
from moonshine_onnx import transcribe

wav = Path(__import__("os").environ["VOICECHAT_MOONSHINE_WAV"])
model_name = __import__("os").environ["VOICECHAT_MOONSHINE_MODEL"]
start = time.perf_counter()
text = transcribe(str(wav), model=model_name)[0]
elapsed = time.perf_counter() - start
print(json.dumps({"text": text, "elapsed_sec": round(elapsed, 3)}, ensure_ascii=False))
"""
    env["VOICECHAT_MOONSHINE_WAV"] = str(wav)
    env["VOICECHAT_MOONSHINE_MODEL"] = model_name
    proc = subprocess.run(
        [str(python_bin), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Moonshine returned no output")
    payload = json.loads(lines[-1])
    text = normalize_text(payload.get("text", ""))
    transcript_path.write_text(text + "\n", encoding="utf-8")
    return text, float(payload["elapsed_sec"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local speech-to-text models on the same audio file")
    parser.add_argument("input", type=Path, help="input audio file")
    parser.add_argument("--model-ids", nargs="+", help="subset of model ids from config/whisper_models.yaml")
    parser.add_argument("--reference-text", help="expected transcript for similarity scoring")
    parser.add_argument("--reference-file", type=Path, help="file that contains expected transcript")
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/voicechat/whisper_model_compare"))
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_catalog()
    model_rows = catalog.get("models", [])
    if args.model_ids:
        wanted = set(args.model_ids)
        model_rows = [row for row in model_rows if row.get("id") in wanted]
    if not model_rows:
        raise SystemExit("NG: no models selected")

    whisper_bin = Path(os.environ.get("VOICECHAT_WHISPER_BIN", "").strip() or ROOT / "whisper.cpp" / "build" / "bin" / "whisper-cli")
    if any(row.get("family") == "whisper.cpp" for row in model_rows) and not whisper_bin.exists():
        raise SystemExit(f"NG: whisper bin not found: {whisper_bin}")

    out_dir = args.out_dir.expanduser() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_wav = out_dir / "input_normalized.wav"
    ffmpeg_normalize(args.input.expanduser(), normalized_wav, {"ffmpeg": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}})

    reference_text = ""
    if args.reference_text:
        reference_text = normalize_text(args.reference_text)
    elif args.reference_file:
        reference_text = normalize_text(args.reference_file.expanduser().read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for row in model_rows:
        family = row.get("family", "")
        model_path = resolve_existing_path(list(row.get("candidate_paths", [])))
        python_bin = resolve_existing_path(list(row.get("python_bin_candidates", [])))
        item: dict[str, Any] = {
            "id": row.get("id"),
            "family": family,
            "label": row.get("label"),
            "role": row.get("role"),
            "download_url": row.get("download_url", ""),
            "model_path": str(model_path) if model_path else "",
            "python_bin": str(python_bin) if python_bin else "",
            "available": False,
        }
        if family == "whisper.cpp":
            item["available"] = bool(model_path)
        elif family == "moonshine_onnx":
            item["available"] = bool(python_bin)
        else:
            item["status"] = "unsupported_family"
            results.append(item)
            continue

        if not item["available"]:
            item["status"] = "missing"
            results.append(item)
            continue
        transcript_path = out_dir / f"{row['id']}.txt"
        try:
            if family == "whisper.cpp":
                out_prefix = out_dir / f"{row['id']}"
                text, elapsed_sec = transcribe_whisper(
                    whisper_bin=whisper_bin,
                    whisper_model=model_path,
                    wav=normalized_wav,
                    out_prefix=out_prefix,
                    lang=args.lang,
                    threads=args.threads,
                )
                transcript_path.write_text(text + "\n", encoding="utf-8")
            else:
                text, elapsed_sec = transcribe_moonshine(
                    python_bin=python_bin,
                    model_name=str(row.get("model_name", "moonshine/tiny")),
                    wav=normalized_wav,
                    transcript_path=transcript_path,
                    cache_dir=resolve_path(row.get("cache_dir")),
                    tmp_dir=resolve_path(row.get("tmp_dir")),
                )
            item.update(
                {
                    "status": "ok",
                    "elapsed_sec": elapsed_sec,
                    "transcript": text,
                    "transcript_path": str(transcript_path),
                    "chars": len(compact_text(text)),
                }
            )
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
        if reference_text:
            item["reference_text"] = reference_text
            if item.get("status") == "ok":
                item["reference_similarity"] = score_chunk(reference_text, str(item.get("transcript", "")))
        results.append(item)

    if reference_text:
        ok_rows = [row for row in results if row.get("status") == "ok"]
        ok_rows.sort(key=lambda row: (float(row.get("reference_similarity", 0.0)), -float(row.get("elapsed_sec", 999999.0))), reverse=True)
        recommended = ok_rows[0]["id"] if ok_rows else ""
    else:
        ok_rows = [row for row in results if row.get("status") == "ok"]
        ok_rows.sort(key=lambda row: float(row.get("elapsed_sec", 999999.0)))
        recommended = ok_rows[0]["id"] if ok_rows else ""

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(args.input),
        "normalized_wav": str(normalized_wav),
        "whisper_bin": str(whisper_bin),
        "reference_text": reference_text,
        "recommended_model_id": recommended,
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "recommended_model_id": recommended}, ensure_ascii=False))
    for row in results:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

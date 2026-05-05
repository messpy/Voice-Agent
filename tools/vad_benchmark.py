#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import webrtcvad

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_input_audio(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def detect_webrtc_speech(
    wav: Path,
    *,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    speech_ratio: float = 0.12,
) -> tuple[bool, dict[str, Any]]:
    audio, sample_rate = sf.read(str(wav), dtype="int16")
    if sample_rate != 16000:
        raise RuntimeError(f"WebRTC VAD requires 16000Hz audio, got {sample_rate}Hz")
    if getattr(audio, "ndim", 1) != 1:
        raise RuntimeError("WebRTC VAD benchmark expects mono audio")

    vad = webrtcvad.Vad(int(aggressiveness))
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
    pcm = audio.tobytes()
    total_frames = 0
    speech_frames = 0
    for idx in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        total_frames += 1
        if vad.is_speech(pcm[idx : idx + frame_bytes], sample_rate):
            speech_frames += 1
    ratio = (speech_frames / total_frames) if total_frames else 0.0
    return ratio >= speech_ratio, {
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "speech_frames": speech_frames,
        "total_frames": total_frames,
        "speech_ratio": round(ratio, 4),
        "threshold": speech_ratio,
        "aggressiveness": aggressiveness,
    }


@dataclass
class VadResult:
    detector: str
    file: str
    has_speech: bool
    elapsed_sec: float
    status: str
    details: dict[str, Any]
    error: str = ""
    expected_speech: bool | None = None
    correct: bool | None = None


def run_detector(
    detector: str,
    audio_path: Path,
    expected_speech: bool | None,
    silero_cfg: dict[str, Any],
    webrtc_cfg: dict[str, Any],
) -> VadResult:
    started = time.perf_counter()
    try:
        if detector == "silero":
            from tools.wake_vad_record import detect_silero_speech

            has_speech, details = detect_silero_speech(audio_path, silero_cfg)
        elif detector == "webrtc":
            has_speech, details = detect_webrtc_speech(audio_path, **webrtc_cfg)
        else:
            raise RuntimeError(f"unsupported detector: {detector}")
    except Exception as exc:
        return VadResult(
            detector=detector,
            file=str(audio_path),
            has_speech=False,
            elapsed_sec=round(time.perf_counter() - started, 4),
            status="error",
            details={},
            error=str(exc),
            expected_speech=expected_speech,
            correct=False if expected_speech is not None else None,
        )

    correct = None if expected_speech is None else bool(has_speech == expected_speech)
    return VadResult(
        detector=detector,
        file=str(audio_path),
        has_speech=bool(has_speech),
        elapsed_sec=round(time.perf_counter() - started, 4),
        status="ok",
        details=details,
        expected_speech=expected_speech,
        correct=correct,
    )


def resolve_default_files() -> list[Path]:
    suite_dir = ROOT / "test_audio_suite"
    candidates = sorted(suite_dir.glob("seg_*.wav"))
    if not candidates:
        candidates = sorted(suite_dir.glob("*.wav"))
    return candidates


def resolve_existing_benchmark_files() -> list[Path]:
    roots = [
        Path("/home/kennypi/data/audio/music/テイキョウヘイセイダイガク.mp3"),
        Path("/home/kennypi/data/audio/voice_recognition/ItsOnlyNewYork.mp3"),
    ]
    return [path for path in roots if path.exists()]


def infer_expected_speech(src: Path) -> bool | None:
    name = src.name
    if name.startswith("seg_") and name.endswith(".wav"):
        return True
    if name in {"test_5min.wav", "test_5min_normalized.wav"}:
        return True
    if name in {
        "テイキョウヘイセイダイガク.mp3",
        "テイキョウヘイセイダイガク_normalized.wav",
        "ItsOnlyNewYork.mp3",
    }:
        return True
    return None


def print_results(results: list[VadResult]) -> None:
    header = (
        f"{'detector':<8} {'file':<18} {'speech':<6} {'time(s)':>8} "
        f"{'expected':<8} {'correct':<7} {'status':<6}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.detector:<8} "
            f"{Path(item.file).name:<18} "
            f"{('yes' if item.has_speech else 'no'):<6} "
            f"{item.elapsed_sec:>8.4f} "
            f"{('-' if item.expected_speech is None else ('yes' if item.expected_speech else 'no')):<8} "
            f"{('-' if item.correct is None else ('yes' if item.correct else 'no')):<7} "
            f"{item.status:<6}"
        )
        if item.error:
            print(f"  error: {item.error}")


def summarize(results: list[VadResult]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    detectors = sorted({item.detector for item in results})
    for detector in detectors:
        items = [item for item in results if item.detector == detector]
        ok_items = [item for item in items if item.status == "ok"]
        judged = [item for item in ok_items if item.correct is not None]
        summary.append(
            {
                "detector": detector,
                "files": len(items),
                "ok": len(ok_items),
                "errors": len(items) - len(ok_items),
                "avg_elapsed_sec": round(
                    sum(item.elapsed_sec for item in ok_items) / len(ok_items), 4
                )
                if ok_items
                else 0.0,
                "accuracy": round(
                    sum(1 for item in judged if item.correct) / len(judged), 4
                )
                if judged
                else None,
            }
        )
    return summary


def parse_expected_map(raw_items: list[str]) -> dict[str, bool]:
    expected: dict[str, bool] = {}
    for item in raw_items:
        if "=" not in item:
            raise SystemExit(f"NG: --expect は path=yes/no の形式で指定してください: {item}")
        key, raw_value = item.split("=", 1)
        value = raw_value.strip().lower()
        if value not in {"yes", "no", "true", "false", "1", "0"}:
            raise SystemExit(f"NG: expected value must be yes/no: {item}")
        expected[Path(key).name] = value in {"yes", "true", "1"}
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VAD detectors on audio files")
    parser.add_argument("audio", nargs="*", type=Path, help="audio files to test")
    parser.add_argument(
        "--suite",
        default="existing_benchmark",
        choices=["existing_benchmark", "test_audio_suite", "manual"],
        help="reuse the previous benchmark audio set by default",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["webrtc", "silero"],
        choices=["webrtc", "silero"],
        help="detectors to benchmark",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        help="expected speech map: path=yes/no",
    )
    parser.add_argument("--webrtc-aggressiveness", type=int, default=2)
    parser.add_argument("--webrtc-frame-ms", type=int, default=30)
    parser.add_argument("--webrtc-speech-ratio", type=float, default=0.12)
    parser.add_argument("--silero-threshold", type=float, default=0.4)
    parser.add_argument("--silero-min-speech-ms", type=int, default=80)
    parser.add_argument("--silero-min-silence-ms", type=int, default=150)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.audio:
        files = args.audio
    elif args.suite == "existing_benchmark":
        files = resolve_existing_benchmark_files()
    elif args.suite == "test_audio_suite":
        files = resolve_default_files()
    else:
        files = []
    if not files:
        raise SystemExit("NG: benchmark target audio not found")

    expected_map = parse_expected_map(args.expect)
    webrtc_cfg = {
        "aggressiveness": args.webrtc_aggressiveness,
        "frame_ms": args.webrtc_frame_ms,
        "speech_ratio": args.webrtc_speech_ratio,
    }
    silero_cfg = {
        "threshold": args.silero_threshold,
        "min_speech_duration_ms": args.silero_min_speech_ms,
        "min_silence_duration_ms": args.silero_min_silence_ms,
    }

    results: list[VadResult] = []
    with tempfile.TemporaryDirectory(prefix="voicechat-vad-bench-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for src in files:
            normalized = tmp_root / f"{src.stem}_16k.wav"
            normalize_input_audio(src, normalized)
            expected = expected_map.get(src.name, infer_expected_speech(src))
            for detector in args.detectors:
                results.append(
                    run_detector(detector, normalized, expected, silero_cfg, webrtc_cfg)
                )

    print_results(results)
    summary = summarize(results)
    print("\nsummary:")
    for item in summary:
        acc = "-" if item["accuracy"] is None else f"{item['accuracy']:.4f}"
        print(
            f"  {item['detector']}: ok={item['ok']}/{item['files']} "
            f"errors={item['errors']} avg={item['avg_elapsed_sec']:.4f}s acc={acc}"
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": args.suite,
        "detectors": args.detectors,
        "webrtc": webrtc_cfg,
        "silero": silero_cfg,
        "results": [asdict(item) for item in results],
        "summary": summary,
    }
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"vad_benchmark_{tag}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nresult saved: {out_path}")


if __name__ == "__main__":
    main()

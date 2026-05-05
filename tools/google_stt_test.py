from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tools.google_stt import transcribe_google


def record_wav(target: Path, seconds: int, device: str) -> None:
    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(seconds),
        str(target),
    ]
    print("Recording", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google STT rate-limited smoke test")
    parser.add_argument("--seconds", type=int, default=60, help="record duration")
    parser.add_argument("--wav", type=Path, default=Path("/tmp/voicechat/google_stt_test.wav"))
    parser.add_argument("--device", default="plughw:CARD=Device,DEV=0")
    parser.add_argument("--google-config", action="store_true", help="dump google config")
    parser.add_argument("--language-code", default="ja-JP")
    parser.add_argument("--model", default="latest_long")
    parser.add_argument("--speech-contexts", nargs="+", default=["スタックチャン", "ずんだもん"])
    parser.add_argument("--manual", action="store_true", help="skip recording; assume input file exists")
    return parser.parse_args()


def build_google_cfg(args: argparse.Namespace) -> dict[str, object]:
    contexts = [{"phrases": args.speech_contexts}]
    cfg: dict[str, object] = {
        "language_code": args.language_code,
        "model": args.model,
        "use_enhanced": True,
        "enable_automatic_punctuation": True,
        "speech_contexts": contexts,
    }
    if args.google_config:
        print("Google config:")
        for key, value in sorted(cfg.items()):
            print(f"  {key}: {value}")
    return cfg


def main() -> None:
    args = parse_args()
    target = args.wav
    if not args.manual:
        target.parent.mkdir(parents=True, exist_ok=True)
        record_wav(target, args.seconds, args.device)
        time.sleep(0.5)

    if not target.exists() or target.stat().st_size == 0:
        raise SystemExit(f"recording missing: {target}")

    cfg = build_google_cfg(args)
    print("Sending", target, "for transcription")
    text, elapsed = transcribe_google(target, cfg)
    print("elapsed_sec:", elapsed)
    print("result:")
    print(text or "(empty)")


if __name__ == "__main__":
    main()

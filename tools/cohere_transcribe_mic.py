from __future__ import annotations

import argparse
import time
from pathlib import Path

from tools.cohere_transcribe import (
    merge_cli_config,
    record_wav,
    resolve_api_key,
    run_transcription,
    temporary_wav_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record from microphone and transcribe with Cohere")
    parser.add_argument("--seconds", type=int, default=10, help="record duration in seconds")
    parser.add_argument("--device", help="ALSA input device, e.g. plughw:CARD=Device,DEV=0")
    parser.add_argument("--api-key", help="override COHERE_API_KEY")
    parser.add_argument("--api-url", help="override Cohere transcription endpoint")
    parser.add_argument("--language", help="language hint, e.g. ja")
    parser.add_argument("--model", help="transcription model name")
    parser.add_argument("--prompt", help="optional prompt")
    parser.add_argument("--timeout-sec", type=int, help="request timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="directory for txt/json outputs")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"), help="output file tag")
    parser.add_argument("--keep-raw", action="store_true", help="keep raw arecord wav in output dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_cli_config(
        language=args.language,
        model=args.model,
        prompt=args.prompt,
        api_url=args.api_url,
        timeout_sec=args.timeout_sec,
        output_dir=args.output_dir,
    )
    api_key = resolve_api_key(args.api_key)

    raw_wav = temporary_wav_path()
    print(f"INFO: recording {args.seconds} seconds")
    record_wav(raw_wav, args.seconds, cfg, device=args.device)

    result = run_transcription(
        src_audio=raw_wav,
        cfg=cfg,
        api_key=api_key,
        tag=args.tag,
    )

    if args.keep_raw:
        kept_raw = Path(result["normalized_audio"]).parent / f"{args.tag}_recorded.wav"
        kept_raw.write_bytes(raw_wav.read_bytes())
        print(f"RAW_AUDIO: {kept_raw}")

    raw_wav.unlink(missing_ok=True)

    print("OK: transcription complete")
    print(f"TEXT: {result['text']}")
    print(f"TXT: {result['txt_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"NORMALIZED_AUDIO: {result['normalized_audio']}")
    print(f"ELAPSED_SEC: {result['elapsed_sec']}")


if __name__ == "__main__":
    main()

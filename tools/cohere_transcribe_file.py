from __future__ import annotations

import argparse
import time
from pathlib import Path

from tools.cohere_transcribe import merge_cli_config, resolve_api_key, run_transcription


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Cohere")
    parser.add_argument("input", type=Path, help="input audio file")
    parser.add_argument("--api-key", help="override COHERE_API_KEY")
    parser.add_argument("--api-url", help="override Cohere transcription endpoint")
    parser.add_argument("--language", help="language hint, e.g. ja")
    parser.add_argument("--model", help="transcription model name")
    parser.add_argument("--prompt", help="optional prompt")
    parser.add_argument("--timeout-sec", type=int, help="request timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="directory for txt/json outputs")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"), help="output file tag")
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
    result = run_transcription(
        src_audio=args.input.expanduser(),
        cfg=cfg,
        api_key=api_key,
        tag=args.tag,
    )
    print("OK: transcription complete")
    print(f"TEXT: {result['text']}")
    print(f"TXT: {result['txt_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"NORMALIZED_AUDIO: {result['normalized_audio']}")
    print(f"ELAPSED_SEC: {result['elapsed_sec']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.remote.faster_whisper_transcribe import transcribe_to_files


DEFAULT_MODELS = [
    "small",
    "medium",
    "distil-large-v3",
    "large-v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="faster-whisper multi-model benchmark")
    parser.add_argument("--wav", required=True, help="input wav path")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="models to benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--llm-provider", default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-api-key", default="")
    parser.add_argument("--gemini-api-base", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--openai-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--anthropic-api-base", default="https://api.anthropic.com/v1")
    parser.add_argument("--anthropic-api-key", default="")
    parser.add_argument("--anthropic-version", default="2023-06-01")
    parser.add_argument("--skip-correct", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_cfg = {
        "provider": args.llm_provider,
        "model": args.llm_model,
        "timeout_sec": args.llm_timeout,
    }
    if args.llm_provider == "ollama":
        llm_cfg.update({"host": args.ollama_host, "api_key": args.ollama_api_key, "web_search": {}})
    elif args.llm_provider == "gemini":
        llm_cfg.update({"api_base": args.gemini_api_base, "api_key": args.gemini_api_key})
    elif args.llm_provider == "openai":
        llm_cfg.update({"api_base": args.openai_api_base, "api_key": args.openai_api_key})
    elif args.llm_provider == "anthropic":
        llm_cfg.update(
            {
                "api_base": args.anthropic_api_base,
                "api_key": args.anthropic_api_key,
                "anthropic_version": args.anthropic_version,
            }
        )

    rows: list[dict] = []
    for model_name in args.models:
        safe_name = model_name.replace("/", "_")
        out_prefix = out_dir / safe_name
        result = transcribe_to_files(
            wav_path=wav_path,
            out_prefix=out_prefix,
            model_name=model_name,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            llm_cfg=llm_cfg,
            skip_correct=args.skip_correct,
        )
        row = {"model": model_name, **result}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

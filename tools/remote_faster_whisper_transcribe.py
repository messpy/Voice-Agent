#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel
from llm_api import llm_chat


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def correct_text(llm_cfg: dict, raw_text: str) -> str:
    system_prompt = (
        "あなたは音声認識の補正器。"
        "誤変換、助詞の崩れ、不要な空白を自然な日本語に直す。"
        "話し言葉として不自然な箇所は、元の意味を保ったまま自然な口語に整える。"
        "固有名詞や人名は確信がないならそのまま残す。"
        "意味を勝手に足さない。"
        "聞き取れない部分は無理に補わない。"
        "出力は補正後の本文だけにする。"
    )
    corrected = llm_chat(llm_cfg, system_prompt, f"音声認識結果:\n{raw_text}\n\n補正後テキストだけを返して。")
    return corrected or raw_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="faster-whisper remote transcribe")
    parser.add_argument("--wav", required=True, help="input wav path")
    parser.add_argument("--out-prefix", required=True, help="output file prefix")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name")
    parser.add_argument("--device", default="cpu", help="faster-whisper device")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type")
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
    parser.add_argument("--skip-correct", action="store_true", help="AI補正をスキップする")
    return parser.parse_args()


def transcribe_to_files(
    *,
    wav_path: Path,
    out_prefix: Path,
    model_name: str,
    device: str,
    compute_type: str,
    beam_size: int,
    llm_cfg: dict,
    skip_correct: bool,
) -> dict:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(str(out_prefix) + "_raw.txt")
    corrected_path = Path(str(out_prefix) + "_corrected.txt")
    meta_path = Path(str(out_prefix) + ".json")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    t0 = time.time()
    segments, info = model.transcribe(
        str(wav_path),
        language="ja",
        beam_size=beam_size,
        vad_filter=False,
        condition_on_previous_text=True,
    )
    raw_text = normalize_text(" ".join(segment.text.strip() for segment in segments))
    elapsed_sec = round(time.time() - t0, 3)
    raw_path.write_text(raw_text + "\n", encoding="utf-8")

    corrected_text = raw_text if skip_correct else correct_text(llm_cfg, raw_text)
    corrected_path.write_text(corrected_text + "\n", encoding="utf-8")

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "language": info.language,
        "language_probability": info.language_probability,
        "elapsed_sec": elapsed_sec,
        "wav": str(wav_path),
        "raw_path": str(raw_path),
        "corrected_path": str(corrected_path),
        "raw": raw_text,
        "corrected": corrected_text,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "raw_path": str(raw_path),
        "corrected_path": str(corrected_path),
        "meta_path": str(meta_path),
        "elapsed_sec": elapsed_sec,
    }


def main() -> None:
    args = parse_args()
    llm_cfg = {
        "provider": args.llm_provider,
        "model": args.llm_model,
        "timeout_sec": args.llm_timeout,
    }
    if args.llm_provider == "ollama":
        llm_cfg.update(
            {
                "host": args.ollama_host,
                "api_key": args.ollama_api_key,
                "web_search": {},
            }
        )
    elif args.llm_provider == "gemini":
        llm_cfg.update(
            {
                "api_base": args.gemini_api_base,
                "api_key": args.gemini_api_key,
            }
        )
    elif args.llm_provider == "openai":
        llm_cfg.update(
            {
                "api_base": args.openai_api_base,
                "api_key": args.openai_api_key,
            }
        )
    elif args.llm_provider == "anthropic":
        llm_cfg.update(
            {
                "api_base": args.anthropic_api_base,
                "api_key": args.anthropic_api_key,
                "anthropic_version": args.anthropic_version,
            }
        )
    result = transcribe_to_files(
        wav_path=Path(args.wav),
        out_prefix=Path(args.out_prefix),
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        llm_cfg=llm_cfg,
        skip_correct=args.skip_correct,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from src.llm_api import llm_chat


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


def build_llm_cfg(args: argparse.Namespace) -> dict:
    llm_cfg: dict[str, object] = {
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
    return llm_cfg


def whisperx_command(
    wav_path: Path,
    output_dir: Path,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    task: str,
    output_format: str,
    batch_size: int | None,
    diarize: bool,
    diarize_model: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    align_model: str | None,
    vad_onset: float | None,
    vad_offset: float | None,
) -> list[str]:
    cmd = [
        "whisperx",
        str(wav_path),
        "--model",
        model,
        "--device",
        device,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--task",
        task,
        "--output_dir",
        str(output_dir),
        "--output_format",
        output_format,
    ]
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if diarize:
        cmd.append("--diarize")
        if diarize_model:
            cmd.extend(["--diarize_model", diarize_model])
    if min_speakers is not None:
        cmd.extend(["--min_speakers", str(min_speakers)])
    if max_speakers is not None:
        cmd.extend(["--max_speakers", str(max_speakers)])
    if align_model:
        cmd.extend(["--align_model", align_model])
    if vad_onset is not None:
        cmd.extend(["--vad_onset", str(vad_onset)])
    if vad_offset is not None:
        cmd.extend(["--vad_offset", str(vad_offset)])
    return cmd


def find_transcript_file(wav: Path, output_dir: Path, diarize: bool) -> Path:
    candidates = sorted(output_dir.glob(f"{wav.stem}*.txt"))
    if not candidates:
        raise RuntimeError(f"whisperx did not produce a transcript for {wav.name}")
    exact = output_dir / f"{wav.stem}.txt"
    if exact.exists():
        return exact
    if diarize:
        for candidate in candidates:
            if "diarize" in candidate.stem.lower():
                return candidate
    return candidates[0]


def transcribe_to_files(
    *,
    wav_path: Path,
    out_prefix: Path,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    task: str,
    output_format: str,
    batch_size: int | None,
    diarize: bool,
    diarize_model: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    align_model: str | None,
    vad_onset: float | None,
    vad_offset: float | None,
    llm_cfg: dict,
    skip_correct: bool,
) -> dict:
    out_prefix_str = str(out_prefix)
    raw_path = Path(out_prefix_str + "_raw.txt")
    corrected_path = Path(out_prefix_str + "_corrected.txt")
    meta_path = Path(out_prefix_str + ".json")

    intermediate_dir = out_prefix.parent / f"{out_prefix.name}_whisperx"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    cmd = whisperx_command(
        wav_path,
        intermediate_dir,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        task=task,
        output_format=output_format,
        batch_size=batch_size,
        diarize=diarize,
        diarize_model=diarize_model,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        align_model=align_model,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
    )
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_sec = round(time.time() - t0, 3)
    if proc.returncode != 0:
        stderr = proc.stderr or ""
        stdout = proc.stdout or ""
        raise RuntimeError(f"whisperx failed (rc={proc.returncode})\nstdout:{stdout}\nstderr:{stderr}")

    transcript_file = find_transcript_file(wav_path, intermediate_dir, diarize)
    raw_text = normalize_text(transcript_file.read_text(encoding="utf-8", errors="replace"))
    raw_path.write_text(raw_text + "\n", encoding="utf-8")
    corrected_text = raw_text if skip_correct else correct_text(llm_cfg, raw_text)
    corrected_path.write_text(corrected_text + "\n", encoding="utf-8")

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "task": task,
        "output_format": output_format,
        "diarize": diarize,
        "diarize_model": diarize_model,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "align_model": align_model,
        "vad_onset": vad_onset,
        "vad_offset": vad_offset,
        "batch_size": batch_size,
        "wav": str(wav_path),
        "raw_path": str(raw_path),
        "corrected_path": str(corrected_path),
        "whisperx_output_dir": str(intermediate_dir),
        "elapsed_sec": elapsed_sec,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperX remote transcribe")
    parser.add_argument("--wav", required=True, help="input wav path")
    parser.add_argument("--out-prefix", required=True, help="output file prefix")
    parser.add_argument("--model", default="large", help="WhisperX model")
    parser.add_argument("--device", default="cpu", help="WhisperX device")
    parser.add_argument("--compute-type", default="int8", help="WhisperX compute type")
    parser.add_argument("--language", default="Japanese", help="language to force (Japanese by default)")
    parser.add_argument("--task", default="transcribe", help="WhisperX task (transcribe/translate)")
    parser.add_argument("--output-format", default="txt", help="whisperx --output_format value")
    parser.add_argument("--batch-size", type=int, help="WhisperX batch size")
    parser.add_argument("--diarize", action="store_true", help="enable WhisperX diarization")
    parser.add_argument("--diarize-model", default="pyannote", help="WhisperX diarize model")
    parser.add_argument("--min-speakers", type=int, help="minimum speakers for diarization")
    parser.add_argument("--max-speakers", type=int, help="maximum speakers for diarization")
    parser.add_argument("--align-model", help="WhisperX align_model override")
    parser.add_argument("--vad-onset", type=float, help="WhisperX vad_onset threshold")
    parser.add_argument("--vad-offset", type=float, help="WhisperX vad_offset threshold")
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
    parser.add_argument("--skip-correct", action="store_true", help="skip LLM correction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_cfg = build_llm_cfg(args)
    result = transcribe_to_files(
        wav_path=Path(args.wav),
        out_prefix=Path(args.out_prefix),
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        task=args.task,
        output_format=args.output_format,
        batch_size=args.batch_size,
        diarize=args.diarize,
        diarize_model=args.diarize_model,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        align_model=args.align_model,
        vad_onset=args.vad_onset,
        vad_offset=args.vad_offset,
        llm_cfg=llm_cfg,
        skip_correct=args.skip_correct,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

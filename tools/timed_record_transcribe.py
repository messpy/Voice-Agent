#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml
from src.llm_api import llm_chat, resolve_llm_config


ROOT = Path(__file__).resolve().parents[1]


def die(msg: str, code: int = 1) -> None:
    print(msg)
    raise SystemExit(code)


def load_cfg() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        die(f"NG: config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def whisper_env_for_bin(whisper_bin: Path) -> dict[str, str]:
    env = dict(os.environ)
    lib_dirs = [
        whisper_bin.parent.parent / "src",
        whisper_bin.parent.parent / "ggml" / "src",
    ]
    existing = [str(path) for path in lib_dirs if path.exists()]
    if existing:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(existing + ([current] if current else []))
    return env


def speak_voicevox(text: str, host: str, speaker: int, volume_scale: float, audio_out: str, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    base = host.rstrip("/")
    query_resp = requests.post(
        base + "/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    query_resp.raise_for_status()
    query = query_resp.json()
    query["volumeScale"] = volume_scale

    synth_resp = requests.post(
        base + "/synthesis",
        params={"speaker": speaker},
        json=query,
        timeout=120,
    )
    synth_resp.raise_for_status()
    out_wav.write_bytes(synth_resp.content)

    cmd = ["aplay"]
    if audio_out:
        cmd += ["-D", audio_out]
    cmd.append(str(out_wav))
    subprocess.run(cmd, check=True)


def transcribe_whisper(whisper_bin: Path, whisper_model: Path, wav: Path, out_prefix: Path, lang: str, threads: int) -> tuple[str, float]:
    for suffix in (".txt", ".json", ".srt", ".vtt"):
        try:
            Path(str(out_prefix) + suffix).unlink()
        except FileNotFoundError:
            pass

    cmd = [
        str(whisper_bin),
        "-m",
        str(whisper_model),
        "-f",
        str(wav),
        "-l",
        lang,
        "-t",
        str(threads),
        "--beam-size",
        "6",
        "--best-of",
        "6",
        "--temperature",
        "0.0",
        "-nt",
        "-otxt",
        "-of",
        str(out_prefix),
    ]
    env = whisper_env_for_bin(whisper_bin)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    dt = round(time.time() - t0, 3)
    if proc.returncode != 0:
        die(proc.stdout or f"NG: whisper failed rc={proc.returncode}")
    txt_path = Path(str(out_prefix) + ".txt")
    raw = txt_path.read_text(encoding="utf-8", errors="replace") if txt_path.exists() else ""
    raw = raw.split("whisper_print_timings:", 1)[0]
    raw = re.sub(r"^output_txt:.*$", "", raw, flags=re.M)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return ("\n".join(lines).strip(), dt)


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


def process_transcript(
    *,
    whisper_bin: Path,
    whisper_model: Path,
    wav_path: Path,
    raw_prefix: Path,
    raw_txt_path: Path,
    corrected_path: Path,
    meta_path: Path,
    lang: str,
    threads: int,
    llm_cfg: dict,
) -> None:
    print("INFO: 生文字起こし")
    raw_text, elapsed_sec = transcribe_whisper(whisper_bin, whisper_model, wav_path, raw_prefix, lang, threads)
    raw_txt_path.write_text(raw_text + "\n", encoding="utf-8")

    print("INFO: AI補正")
    corrected_text = correct_text(llm_cfg, raw_text)
    corrected_path.write_text(corrected_text + "\n", encoding="utf-8")

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(whisper_model),
        "wav": str(wav_path),
        "raw_path": str(raw_txt_path),
        "corrected_path": str(corrected_path),
        "elapsed_sec": elapsed_sec,
        "raw": normalize_text(raw_text),
        "corrected": normalize_text(corrected_text),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: WAV={wav_path}")
    print(f"OK: RAW={raw_txt_path}")
    print(f"OK: CORRECTED={corrected_path}")
    print(f"OK: META={meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="開始/終了音声つき録音と文字起こし")
    parser.add_argument("--seconds", type=int, default=300, help="録音秒数")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"), help="出力タグ")
    parser.add_argument("--background", action="store_true", help="文字起こしとAI補正をバックグラウンドで実行する")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg()

    audio_cfg = cfg["audio"]
    whisper_cfg = cfg["whisper"]
    llm_cfg = resolve_llm_config(cfg)
    tts_cfg = cfg["tts"]

    audio_in = audio_cfg["input"]
    audio_out = audio_cfg.get("output", "default")
    whisper_bin = Path(os.environ.get("VOICECHAT_WHISPER_BIN", "").strip() or whisper_cfg["bin"])
    whisper_model = Path(os.environ.get("VOICECHAT_WHISPER_MODEL", "").strip() or whisper_cfg["model"])
    lang = whisper_cfg.get("lang", "ja")
    threads = int(whisper_cfg.get("threads", 4))
    vv_host = tts_cfg["engine_host"]
    vv_speaker = int(tts_cfg.get("speaker", 3))
    vv_volume = float(tts_cfg.get("voicevox_volume_scale", 0.1))
    out_dir = Path("/tmp/voicechat")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"timed_record_{args.tag}.wav"
    raw_prefix = out_dir / f"timed_record_{args.tag}_raw"
    raw_txt_path = out_dir / f"timed_record_{args.tag}_raw.txt"
    corrected_path = out_dir / f"timed_record_{args.tag}_corrected.txt"
    meta_path = out_dir / f"timed_record_{args.tag}.json"

    if os.environ.get("VOICECHAT_BG_TRANSCRIBE_ONLY") == "1":
        process_transcript(
            whisper_bin=whisper_bin,
            whisper_model=whisper_model,
            wav_path=wav_path,
            raw_prefix=raw_prefix,
            raw_txt_path=raw_txt_path,
            corrected_path=corrected_path,
            meta_path=meta_path,
            lang=lang,
            threads=threads,
            llm_cfg=llm_cfg,
        )
        return

    print("INFO: スタート音声")
    speak_voicevox("スタート", vv_host, vv_speaker, vv_volume, audio_out, out_dir / f"timed_record_{args.tag}_start.wav")

    print(f"INFO: 録音開始 {args.seconds}s")
    rec_cmd = [
        "arecord",
        "-D",
        audio_in,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(args.seconds),
        str(wav_path),
    ]
    subprocess.run(rec_cmd, check=True)

    print("INFO: 終了音声")
    speak_voicevox("5分終了", vv_host, vv_speaker, vv_volume, audio_out, out_dir / f"timed_record_{args.tag}_end.wav")

    if args.background:
        worker_cmd = [
            sys.executable,
            str(Path(__file__)),
            "--tag",
            args.tag,
            "--seconds",
            str(args.seconds),
        ]
        env = dict(os.environ)
        env["VOICECHAT_BG_TRANSCRIBE_ONLY"] = "1"
        log_path = out_dir / f"timed_record_{args.tag}_bg.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.Popen(worker_cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        print(f"OK: WAV={wav_path}")
        print(f"OK: BG_LOG={log_path}")
        print(f"OK: RAW={raw_txt_path}")
        print(f"OK: CORRECTED={corrected_path}")
        print(f"OK: META={meta_path}")
        return

    process_transcript(
        whisper_bin=whisper_bin,
        whisper_model=whisper_model,
        wav_path=wav_path,
        raw_prefix=raw_prefix,
        raw_txt_path=raw_txt_path,
        corrected_path=corrected_path,
        meta_path=meta_path,
        lang=lang,
        threads=threads,
        llm_cfg=llm_cfg,
    )


if __name__ == "__main__":
    main()

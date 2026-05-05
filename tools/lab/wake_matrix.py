#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wake_vad_record import (
    arecord_chunk_pcm,
    contains_any_wake,
    load_cfg,
    load_wake_aliases,
    normalize_text,
    normalize_transcript_text,
    play_wav_file,
    speak_with_voicevox,
    synth_tone_wav,
    transcribe_local,
    wav_write_pcm16_mono,
)


def list_capture_pcm_candidates() -> list[str]:
    proc = subprocess.run(
        ["arecord", "-L"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return []
    lines = (proc.stdout or b"").decode("utf-8", errors="replace").splitlines()
    candidates: list[str] = []
    for line in lines:
        pcm = line.strip()
        if not pcm:
            continue
        if pcm.startswith(("dsnoop:CARD=", "plughw:CARD=", "sysdefault:CARD=")):
            candidates.append(pcm)
    ordered: list[str] = []
    for pcm in candidates:
        if pcm not in ordered:
            ordered.append(pcm)
    return ordered


def existing_model_candidates(cfg: dict) -> list[Path]:
    model_dir = ROOT / "whisper.cpp" / "models"
    model_paths = [
        Path(str(cfg.get("wake", {}).get("whisper_model", "")).strip()),
        Path(str(cfg.get("whisper", {}).get("realtime_model", "")).strip()),
        Path(str(cfg.get("whisper", {}).get("model", "")).strip()),
    ]
    models: list[Path] = []
    for item in model_paths:
        if not item:
            continue
        model_path = item if item.is_absolute() else ROOT / item
        if model_path.exists() and model_path not in models:
            models.append(model_path)
    if model_dir.exists():
        for model_path in sorted(model_dir.glob("ggml-*.bin")):
            if model_path.exists() and model_path not in models:
                models.append(model_path)
    return models


def remote_model_candidates(cfg: dict) -> list[str]:
    remote_cfg = cfg.get("remote", {})
    model_name = normalize_text(str(remote_cfg.get("whisper_model", "")))
    return [model_name] if model_name else []


def load_phrase_candidates(args: argparse.Namespace) -> list[str]:
    phrases: list[str] = []
    for item in args.phrases or []:
        body = normalize_text(str(item))
        if body:
            phrases.append(body)
    raw_phrase_csv = getattr(args, "phrase_csv", "")
    phrase_csv = normalize_text("" if raw_phrase_csv is None else str(raw_phrase_csv))
    if phrase_csv:
        for item in phrase_csv.split(","):
            body = normalize_text(item)
            if body:
                phrases.append(body)
    raw_phrases_file = getattr(args, "phrases_file", "")
    phrases_file = normalize_text("" if raw_phrases_file is None else str(raw_phrases_file))
    if phrases_file:
        path = Path(phrases_file)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"NG: phrases file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            body = normalize_text(line)
            if body and not body.startswith("#"):
                phrases.append(body)
    ordered: list[str] = []
    for item in phrases or ["スタックチャン", "アレクサ", "ずんだもん"]:
        if item not in ordered:
            ordered.append(item)
    return ordered


def announce_test(
    *,
    text: str,
    cfg: dict,
    out_wav: Path,
) -> None:
    tts_cfg = cfg.get("tts", {})
    if not tts_cfg.get("enabled", True):
        return
    speak_with_voicevox(
        text=text,
        audio_out=str(cfg["audio"].get("output", "default")),
        out_wav=out_wav,
        engine_host=str(tts_cfg.get("engine_host", "http://127.0.0.1:50021")),
        speaker=int(tts_cfg.get("speaker", 3)),
        speed_scale=float(tts_cfg.get("voicevox_speed_scale", 1.0)),
        pitch_scale=float(tts_cfg.get("voicevox_pitch_scale", 0.0)),
        intonation_scale=float(tts_cfg.get("voicevox_intonation_scale", 1.0)),
        volume_scale=float(tts_cfg.get("voicevox_volume_scale", 1.0)),
    )


def announce_result(
    *,
    phrase: str,
    recognized: str,
    matched: bool,
    elapsed_sec: float,
    cfg: dict,
    out_wav: Path,
) -> None:
    status = "一致したのだ" if matched else "一致しなかったのだ"
    body = recognized or "音声なし"
    text = (
        f"結果です。"
        f"認識は {body}。"
        f"{status}。"
        f"時間は {elapsed_sec:.1f} 秒。"
        f"期待した言葉は {phrase} なのだ。"
    )
    announce_test(text=text, cfg=cfg, out_wav=out_wav)


def play_prompt_effect(*, cfg: dict, out_wav: Path) -> None:
    effect_cfg = (
        cfg.get("effects", {}).get("commands", {}).get("volume_down", {})
        if isinstance(cfg.get("effects", {}).get("commands", {}), dict)
        else {}
    )
    if not isinstance(effect_cfg, dict) or not effect_cfg.get("enabled", False):
        return
    file_value = normalize_text(str(effect_cfg.get("file", "")))
    if file_value:
        wav_path = Path(file_value)
        if not wav_path.is_absolute():
            wav_path = ROOT / wav_path
        if wav_path.exists():
            play_wav_file(wav_path, str(cfg["audio"].get("output", "default")))
            return
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))
    synth_tone_wav(
        out_wav,
        sample_rate=sample_rate,
        duration_ms=int(effect_cfg.get("duration_ms", 90)),
        frequency_hz=float(effect_cfg.get("frequency_hz", 880)),
        volume=float(effect_cfg.get("volume", 0.25)),
        gap_ms=int(effect_cfg.get("gap_ms", 0)),
        sequence=effect_cfg.get("sequence"),
    )
    play_wav_file(out_wav, str(cfg["audio"].get("output", "default")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Microphone x model wake-word matrix test")
    parser.add_argument("--phrase", action="append", dest="phrases", help="Test phrase to speak; repeatable")
    parser.add_argument("--phrase-csv", help="Comma-separated test phrases")
    parser.add_argument("--phrases-file", help="UTF-8 text file with one test phrase per line")
    parser.add_argument("--input", action="append", dest="inputs", help="ALSA capture PCM; repeatable")
    parser.add_argument("--model", action="append", dest="models", help="Whisper model path; repeatable")
    parser.add_argument("--record-sec", type=float, default=2.0, help="Recording seconds per attempt")
    parser.add_argument("--sleep-sec", type=float, default=0.6, help="Delay after spoken instruction")
    parser.add_argument("--announce", action="store_true", default=True, help="Speak test instructions")
    parser.add_argument("--no-announce", action="store_false", dest="announce", help="Do not speak test instructions")
    parser.add_argument("--list-only", action="store_true", help="Only print discovered inputs/models")
    args = parser.parse_args()

    cfg = load_cfg()
    workdir = Path(str(cfg.get("paths", {}).get("workdir", "/tmp/voicechat"))) / "lab" / "wake_matrix"
    workdir.mkdir(parents=True, exist_ok=True)

    sr = int(cfg["audio"].get("sample_rate", 16000))
    whisper_bin = Path(str(cfg["whisper"]["bin"]))
    lang = str(cfg["whisper"].get("lang", "ja"))
    beam = int(cfg["whisper"].get("beam", 6))
    best = int(cfg["whisper"].get("best", 6))
    temp = float(cfg["whisper"].get("temp", 0.0))
    threads = int(cfg["whisper"].get("threads", 4))

    inputs = args.inputs or list_capture_pcm_candidates()
    models = [Path(item) for item in args.models] if args.models else existing_model_candidates(cfg)
    remote_models = remote_model_candidates(cfg)
    phrases = load_phrase_candidates(args)

    wake_cfg = cfg.get("wake", {})
    wake_words = [normalize_text(str(item)) for item in wake_cfg.get("words", []) if normalize_text(str(item))]
    if not wake_words:
        wake_words = [normalize_text(str(wake_cfg.get("word", "スタックチャン")))]
    wake_aliases = load_wake_aliases(wake_cfg, wake_words)

    print("[Wake Matrix]")
    print(f"workdir: {workdir}")
    print(f"phrases: {phrases}")
    print(f"inputs: {inputs}")
    print(f"models: {[str(item) for item in models]}")
    if remote_models:
        print(f"remote_models: {remote_models}")
    if args.list_only:
        return

    if not whisper_bin.exists():
        raise SystemExit(f"NG: whisper bin not found: {whisper_bin}")
    if not inputs:
        raise SystemExit("NG: no capture input candidates found")
    if not models:
        raise SystemExit("NG: no whisper model candidates found")

    results_path = workdir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    test_index = 0

    for phrase in phrases:
        for input_pcm in inputs:
            for model_path in models:
                if not model_path.exists():
                    continue
                test_index += 1
                model_label = model_path.name
                input_label = input_pcm
                display_prompt = (
                    f"テスト {test_index} です。"
                    f"次は {phrase} と言ってください。"
                    f"入力は {input_label}、モデルは {model_label} です。"
                )
                spoken_prompt = (
                    f"テスト {test_index} です。"
                    f"モデルは {model_label} です。"
                    f"次は {phrase} と言ってくださいなのだ。"
                )
                print("")
                print("=" * 72)
                print(f"TEST {test_index}: phrase={phrase} input={input_label} model={model_label}")
                print(display_prompt)
                if args.announce:
                    announce_test(text=spoken_prompt, cfg=cfg, out_wav=workdir / f"prompt_{test_index:03d}.wav")
                play_prompt_effect(cfg=cfg, out_wav=workdir / f"prompt_effect_{test_index:03d}.wav")
                time.sleep(max(0.0, args.sleep_sec))

                wav_path = workdir / f"capture_{test_index:03d}.wav"
                started = time.time()
                try:
                    pcm = arecord_chunk_pcm(input_pcm, args.record_sec, sr)
                except Exception as exc:
                    result = {
                        "test_index": test_index,
                        "phrase": phrase,
                        "input": input_pcm,
                        "model": str(model_path),
                        "error": normalize_text(str(exc)),
                    }
                    print(json.dumps(result, ensure_ascii=False))
                    results_path.write_text("", encoding="utf-8") if not results_path.exists() else None
                    with results_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                wav_write_pcm16_mono(wav_path, [pcm], sr)
                recognized, elapsed_sec, _ = transcribe_local(
                    whisper_bin=whisper_bin,
                    whisper_model=model_path,
                    wav=wav_path,
                    lang=lang,
                    threads=threads,
                    beam=beam,
                    best=best,
                    temp=temp,
                )
                recognized = normalize_transcript_text(recognized)
                matched, matched_wake_word = contains_any_wake(recognized, wake_words, wake_aliases)
                result = {
                    "test_index": test_index,
                    "phrase": phrase,
                    "input": input_pcm,
                    "model": str(model_path),
                    "recognized": recognized,
                    "matched": matched,
                    "matched_wake_word": matched_wake_word,
                    "record_sec": args.record_sec,
                    "elapsed_sec": elapsed_sec,
                    "wall_sec": round(time.time() - started, 3),
                    "wav_path": str(wav_path),
                }
                print(json.dumps(result, ensure_ascii=False))
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                if args.announce:
                    announce_result(
                        phrase=phrase,
                        recognized=recognized,
                        matched=matched,
                        elapsed_sec=elapsed_sec,
                        cfg=cfg,
                        out_wav=workdir / f"result_{test_index:03d}.wav",
                    )

    print("")
    print(f"OK: results={results_path}")


if __name__ == "__main__":
    main()

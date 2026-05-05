from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import requests

from tools.cohere_transcribe import ROOT, load_cfg
from tools.timed_record_transcribe import transcribe_whisper


def die(message: str, code: int = 1) -> None:
    print(message)
    raise SystemExit(code)


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\u3000", " ").split()).strip()


def normalize_phrase_test_entry(item: object, idx: int) -> dict[str, str]:
    if isinstance(item, dict):
        text = normalize_text(str(item.get("text", "")))
        if not text:
            return {}
        entry_id = normalize_text(str(item.get("id", ""))) or f"phrase_{idx:02d}"
        note = normalize_text(str(item.get("note", "")))
        return {"id": entry_id, "text": text, "note": note}
    text = normalize_text(str(item))
    if not text:
        return {}
    return {"id": f"phrase_{idx:02d}", "text": text, "note": ""}


def load_phrase_test_items(phrase_test_cfg: dict[str, Any]) -> list[dict[str, str]]:
    phrases = phrase_test_cfg.get("phrases")
    if isinstance(phrases, list) and phrases:
        items = [normalize_phrase_test_entry(item, idx) for idx, item in enumerate(phrases, start=1)]
        return [item for item in items if item]

    phrases_file = str(phrase_test_cfg.get("phrases_file", "")).strip()
    if not phrases_file:
        return []

    path = Path(phrases_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise RuntimeError(f"phrase_test phrases_file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("phrases", [])
        if not isinstance(data, list):
            raise RuntimeError(f"phrase_test json must be a list or {{\"phrases\": [...]}}: {path}")
        items = [normalize_phrase_test_entry(item, idx) for idx, item in enumerate(data, start=1)]
        return [item for item in items if item]

    items: list[dict[str, str]] = []
    line_no = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = normalize_text(line)
        if not text or text.startswith("#"):
            continue
        line_no += 1
        items.append({"id": f"phrase_{line_no:02d}", "text": text, "note": ""})
    return items


def load_style_cycle(style_cfg: object) -> list[dict[str, str | int]]:
    if not isinstance(style_cfg, list):
        return []
    items: list[dict[str, str | int]] = []
    for idx, item in enumerate(style_cfg, start=1):
        if not isinstance(item, dict):
            continue
        try:
            speaker = int(item.get("speaker"))
        except Exception:
            continue
        name = normalize_text(str(item.get("name", ""))) or f"style_{idx:02d}"
        items.append({"speaker": speaker, "name": name})
    return items


def char_ngrams(text: str, *, min_n: int = 2, max_n: int = 3) -> set[str]:
    compact = normalize_text(text).replace(" ", "")
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
    return (coverage * 0.8) + (density * 0.2)


def judge_similarity(expected: str, recognized: str) -> str:
    similarity = score_chunk(expected, recognized)
    if normalize_text(expected) == normalize_text(recognized):
        return "MATCH"
    if similarity >= 0.75:
        return "NEAR"
    return "MISMATCH"


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False) + "\n")


def append_event_sqlite(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            create table if not exists events (
                id integer primary key autoincrement,
                ts integer not null,
                date text not null,
                event_type text not null,
                mode text,
                backend text,
                model text,
                expected text,
                recognized text,
                elapsed_sec real,
                payload_json text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists recognition_aliases (
                id integer primary key autoincrement,
                alias_type text not null,
                target text not null,
                alias text not null,
                source text not null default 'auto',
                hits integer not null default 1,
                last_seen_ts integer not null,
                enabled integer not null default 1,
                unique(alias_type, target, alias)
            )
            """
        )
        conn.execute("create index if not exists idx_recognition_aliases_type_target on recognition_aliases(alias_type, target)")
        conn.execute("create index if not exists idx_recognition_aliases_alias on recognition_aliases(alias)")
        conn.execute(
            """
            insert into events (ts, date, event_type, mode, backend, model, expected, recognized, elapsed_sec, payload_json)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(payload["ts"]),
                str(payload["date"]),
                "phrase_test",
                "phrase_test",
                str(payload.get("backend", "")),
                str(payload.get("model", "")),
                str(payload.get("expected", "")),
                str(payload.get("recognized", "")),
                float(payload.get("elapsed_sec", 0.0)),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()


def load_recognition_aliases(sqlite_path: Path, alias_type: str) -> dict[str, set[str]]:
    if not sqlite_path.exists():
        return {}
    rows: dict[str, set[str]] = {}
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select target, alias
            from recognition_aliases
            where alias_type = ?
              and enabled = 1
            order by hits desc, last_seen_ts desc
            """,
            (alias_type,),
        )
        for target, alias in cursor.fetchall():
            target_text = normalize_text(str(target))
            alias_text = normalize_text(str(alias))
            if not target_text or not alias_text:
                continue
            rows.setdefault(target_text, set()).add(alias_text)
    return rows


def save_recognition_alias(
    sqlite_path: Path,
    *,
    alias_type: str,
    target: str,
    alias: str,
    source: str = "auto",
) -> None:
    target_text = normalize_text(target)
    alias_text = normalize_text(alias)
    if not sqlite_path.exists() or not target_text or not alias_text:
        return
    if target_text == alias_text:
        return
    now_ts = int(time.time())
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            insert into recognition_aliases(alias_type, target, alias, source, hits, last_seen_ts, enabled)
            values (?, ?, ?, ?, 1, ?, 1)
            on conflict(alias_type, target, alias)
            do update set
                hits = recognition_aliases.hits + 1,
                last_seen_ts = excluded.last_seen_ts,
                enabled = 1
            """,
            (alias_type, target_text, alias_text, source, now_ts),
        )
        conn.commit()


def play_wav_file(wav_path: Path, audio_out: str) -> None:
    cmd = ["aplay"]
    if audio_out:
        cmd += ["-D", audio_out]
    cmd.append(str(wav_path))
    subprocess.run(cmd, check=True)


def wav_write_pcm16_mono(dst_wav: Path, pcm_list: list[bytes], sample_rate: int) -> None:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in pcm_list:
            wf.writeframes(chunk)


def synth_tone_wav(
    dst_wav: Path,
    *,
    sample_rate: int,
    duration_ms: int,
    frequency_hz: float,
    volume: float,
    gap_ms: int = 0,
    sequence: list[dict[str, float | int]] | None = None,
) -> None:
    import math

    steps = sequence or [
        {
            "frequency_hz": frequency_hz,
            "duration_ms": duration_ms,
            "volume": volume,
            "gap_ms": gap_ms,
        }
    ]
    pcm_chunks: list[bytes] = []
    for step in steps:
        step_duration_ms = int(step.get("duration_ms", duration_ms))
        step_frequency_hz = float(step.get("frequency_hz", frequency_hz))
        step_volume = float(step.get("volume", volume))
        step_gap_ms = int(step.get("gap_ms", gap_ms))
        frame_count = max(1, int(sample_rate * (step_duration_ms / 1000.0)))
        amplitude = max(0.0, min(1.0, step_volume)) * 32767.0
        tone_bytes = bytearray()
        for idx in range(frame_count):
            sample = int(amplitude * math.sin(2.0 * math.pi * step_frequency_hz * (idx / sample_rate)))
            tone_bytes.extend(sample.to_bytes(2, byteorder="little", signed=True))
        pcm_chunks.append(bytes(tone_bytes))
        if step_gap_ms > 0:
            gap_frames = max(1, int(sample_rate * (step_gap_ms / 1000.0)))
            pcm_chunks.append(b"\x00\x00" * gap_frames)
    wav_write_pcm16_mono(dst_wav, pcm_chunks, sample_rate)


def play_effect(name: str, effect_cfg: dict[str, Any], audio_out: str, out_dir: Path, sample_rate: int) -> None:
    if not effect_cfg.get("enabled", False):
        return
    file_value = normalize_text(str(effect_cfg.get("file", "")))
    if file_value:
        effect_wav = Path(file_value)
        if not effect_wav.is_absolute():
            effect_wav = ROOT / effect_wav
        if effect_wav.exists():
            play_wav_file(effect_wav, audio_out)
            return
    out_wav = out_dir / f"effect_{name}.wav"
    synth_tone_wav(
        out_wav,
        sample_rate=sample_rate,
        duration_ms=int(effect_cfg.get("duration_ms", 90)),
        frequency_hz=float(effect_cfg.get("frequency_hz", 880)),
        volume=float(effect_cfg.get("volume", 0.25)),
        gap_ms=int(effect_cfg.get("gap_ms", 0)),
        sequence=effect_cfg.get("sequence"),
    )
    play_wav_file(out_wav, audio_out)


def speak_with_voicevox(
    text: str,
    audio_out: str,
    out_wav: Path,
    *,
    engine_host: str,
    speaker: int,
    speed_scale: float,
    pitch_scale: float,
    intonation_scale: float,
    volume_scale: float,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    host = engine_host.rstrip("/")
    query_resp = requests.post(
        host + "/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    query_resp.raise_for_status()
    query = query_resp.json()
    query["speedScale"] = speed_scale
    query["pitchScale"] = pitch_scale
    query["intonationScale"] = intonation_scale
    query["volumeScale"] = volume_scale
    synth_resp = requests.post(
        host + "/synthesis",
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


def safe_speak(text: str, tts_ctx: dict[str, Any], filename: str) -> None:
    if not text:
        return
    try:
        speak_with_voicevox(
            text,
            str(tts_ctx["audio_out"]),
            Path(tts_ctx["out_dir"]) / filename,
            engine_host=str(tts_ctx["engine_host"]),
            speaker=int(tts_ctx["speaker"]),
            speed_scale=float(tts_ctx["speed_scale"]),
            pitch_scale=float(tts_ctx["pitch_scale"]),
            intonation_scale=float(tts_ctx["intonation_scale"]),
            volume_scale=float(tts_ctx["volume_scale"]),
        )
    except Exception as exc:
        print(f"WARN: voice output failed: {exc}")


def build_result_speech(expected: str, recognized: str, verdict: str, backend: str) -> str:
    expected_body = normalize_text(expected)
    recognized_body = normalize_text(recognized) or "聞き取れなかったのだ"
    if verdict == "MATCH":
        return f"{backend}で聞き取ったのだ。結果は、{recognized_body}。ちゃんと一致したのだ。"
    if verdict == "NEAR":
        return f"{backend}で聞き取ったのだ。結果は、{recognized_body}。かなり近いのだ。お手本は、{expected_body}なのだ。"
    return f"{backend}で聞き取ったのだ。結果は、{recognized_body}。お手本は、{expected_body}だったのだ。"


def resolve_phrase_alias_target(recognized: str, alias_map: dict[str, set[str]]) -> tuple[str, bool]:
    body = normalize_text(recognized)
    if not body:
        return "", False
    if body in alias_map:
        return body, False
    for target, aliases in alias_map.items():
        if body in aliases:
            return target, True
    return body, False


def pick_phrase(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any], str]:
    mode_cfg = (((cfg.get("assistant") or {}).get("modes") or {}).get("phrase_test") or {})
    items = load_phrase_test_items(mode_cfg)
    if args.text:
        item = {"id": "manual", "text": normalize_text(args.text), "note": "manual"}
    elif args.phrase_id:
        matches = [item for item in items if item.get("id") == args.phrase_id]
        if not matches:
            die(f"NG: phrase_id not found: {args.phrase_id}")
        item = matches[0]
    elif args.index:
        if args.index < 1 or args.index > len(items):
            die(f"NG: index out of range: {args.index}")
        item = items[args.index - 1]
    else:
        if not items:
            die("NG: phrase_test items are empty")
        item = random.choice(items)
    prompt_template = str(mode_cfg.get("prompt_template", "「{phrase}」と言ってください。"))
    return item, mode_cfg, prompt_template.format(phrase=item["text"])


def transcribe_with_local_whisper(raw_wav: Path, cfg: dict[str, Any], out_dir: Path, tag: str) -> tuple[str, float]:
    whisper_cfg = cfg.get("whisper", {})
    whisper_bin = Path(str(whisper_cfg.get("bin", "")).strip())
    whisper_model = Path(str(whisper_cfg.get("model", "")).strip())
    if not whisper_bin.exists():
        raise RuntimeError(f"whisper bin missing: {whisper_bin}")
    if not whisper_model.exists():
        raise RuntimeError(f"whisper model missing: {whisper_model}")
    out_prefix = out_dir / f"{tag}_whisper_fallback"
    return transcribe_whisper(
        whisper_bin=whisper_bin,
        whisper_model=whisper_model,
        wav=raw_wav,
        out_prefix=out_prefix,
        lang=str(whisper_cfg.get("lang", "ja")),
        threads=int(whisper_cfg.get("threads", 4)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zundamon phrase-repeat test with Cohere")
    parser.add_argument("--api-key", help="override COHERE_API_KEY")
    parser.add_argument("--api-url", help="override Cohere transcription endpoint")
    parser.add_argument("--language", default="ja", help="language hint")
    parser.add_argument("--model", help="transcription model name")
    parser.add_argument("--prompt", help="optional prompt")
    parser.add_argument("--timeout-sec", type=int, help="request timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="directory for txt/json outputs")
    parser.add_argument("--seconds", type=int, default=8, help="record duration")
    parser.add_argument("--device", help="ALSA input device")
    parser.add_argument("--audio-out", help="ALSA output device")
    parser.add_argument("--index", type=int, help="1-based phrase index")
    parser.add_argument("--phrase-id", help="phrase id to use")
    parser.add_argument("--text", help="override expected phrase")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"), help="output file tag")
    parser.add_argument("--no-fallback", action="store_true", help="disable local Whisper fallback")
    parser.add_argument("--dry-run", action="store_true", help="print prompt only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg()
    item, mode_cfg, spoken_prompt = pick_phrase(args, cfg)

    print(f"PHRASE_ID: {item['id']}")
    print(f"EXPECTED: {item['text']}")
    print(f"PROMPT: {spoken_prompt}")
    if args.dry_run:
        return

    from tools.cohere_transcribe import merge_cli_config, output_dir_from_cfg, record_wav, resolve_api_key, run_transcription, temporary_wav_path

    cohere_cfg = merge_cli_config(
        language=args.language,
        model=args.model,
        prompt=args.prompt,
        api_url=args.api_url,
        timeout_sec=args.timeout_sec,
        output_dir=args.output_dir,
    )
    out_dir = output_dir_from_cfg(cohere_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = out_dir / "phrase_test_results.db"
    append_event_sqlite(
        sqlite_path,
        {
            "ts": int(time.time()),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "init",
            "backend": "init",
            "model": "",
            "expected": "",
            "recognized": "",
            "elapsed_sec": 0.0,
        },
    )
    alias_map = load_recognition_aliases(sqlite_path, "phrase")

    tts_cfg = cfg.get("tts", {})
    if not bool(tts_cfg.get("enabled", True)):
        die("NG: tts.enabled is false")
    style_cycle = load_style_cycle(mode_cfg.get("style_cycle", []))
    style = style_cycle[0] if style_cycle else {"speaker": int(tts_cfg.get("speaker", 3)), "name": "default"}
    audio_cfg = cfg.get("audio", {})
    effects_cfg = cfg.get("effects", {})
    tts_ctx = {
        "audio_out": args.audio_out if args.audio_out is not None else str(audio_cfg.get("output", "default")),
        "out_dir": out_dir,
        "engine_host": str(tts_cfg.get("engine_host", "http://127.0.0.1:50021")),
        "speaker": int(style["speaker"]),
        "speed_scale": float(tts_cfg.get("voicevox_speed_scale", 1.0)),
        "pitch_scale": float(tts_cfg.get("voicevox_pitch_scale", 0.03)),
        "intonation_scale": float(tts_cfg.get("voicevox_intonation_scale", 1.15)),
        "volume_scale": float(tts_cfg.get("voicevox_volume_scale", 1.0)),
    }

    safe_speak(spoken_prompt, tts_ctx, f"{args.tag}_prompt.wav")
    play_effect("ready", effects_cfg.get("ready", {}), str(tts_ctx["audio_out"]), out_dir, int(audio_cfg.get("sample_rate", 16000)))

    raw_wav = temporary_wav_path(prefix="cohere_phrase_")
    backend_used = "cohere"
    model_used = str(cohere_cfg.get("model", ""))
    result_text = ""
    elapsed_sec = 0.0
    error_message = ""

    try:
        print(f"INFO: recording {args.seconds} seconds")
        record_wav(raw_wav, args.seconds, cohere_cfg, device=args.device)
        play_effect("recorded", effects_cfg.get("recorded", {}), str(tts_ctx["audio_out"]), out_dir, int(audio_cfg.get("sample_rate", 16000)))
        raw_copy = out_dir / f"{args.tag}_recorded.wav"
        raw_copy.write_bytes(raw_wav.read_bytes())
        safe_speak("聞き取っているのだ。少し待ってほしいのだ。", tts_ctx, f"{args.tag}_processing.wav")

        try:
            api_key = resolve_api_key(args.api_key)
            result = run_transcription(
                src_audio=raw_wav,
                cfg=cohere_cfg,
                api_key=api_key,
                tag=args.tag,
            )
            result_text = str(result["text"])
            elapsed_sec = float(result["elapsed_sec"])
            model_used = str(cohere_cfg.get("model", ""))
        except BaseException as exc:
            error_message = str(exc)
            if args.no_fallback:
                raise
            print(f"WARN: cohere failed, falling back to local whisper: {exc}")
            safe_speak("Cohere がだめだったので、ローカルで聞き直すのだ。", tts_ctx, f"{args.tag}_fallback.wav")
            result_text, elapsed_sec = transcribe_with_local_whisper(raw_wav, cfg, out_dir, args.tag)
            backend_used = "local_whisper"
            whisper_cfg = cfg.get("whisper", {})
            model_used = str(whisper_cfg.get("model", ""))
            fallback_txt = out_dir / f"{args.tag}_fallback.txt"
            fallback_json = out_dir / f"{args.tag}_fallback.json"
            fallback_txt.write_text(result_text + "\n", encoding="utf-8")
            fallback_json.write_text(
                json.dumps(
                    {
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "backend": backend_used,
                        "model": model_used,
                        "elapsed_sec": elapsed_sec,
                        "text": result_text,
                        "cohere_error": error_message,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        safe_speak(f"テストに失敗したのだ。{exc}", tts_ctx, f"{args.tag}_error.wav")
        raise
    finally:
        raw_wav.unlink(missing_ok=True)

    normalized_recognized, alias_applied = resolve_phrase_alias_target(result_text, alias_map)
    verdict = judge_similarity(item["text"], normalized_recognized)
    similarity = round(score_chunk(item["text"], normalized_recognized), 4)
    raw_similarity = round(score_chunk(item["text"], result_text), 4)
    if normalize_text(result_text) and normalize_text(item["text"]) != normalize_text(result_text):
        if alias_applied or verdict in {"MATCH", "NEAR"}:
            save_recognition_alias(
                sqlite_path,
                alias_type="phrase",
                target=item["text"],
                alias=result_text,
                source="cohere_phrase_test",
            )
    payload = {
        "ts": int(time.time()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "phrase_test",
        "backend": backend_used,
        "model": model_used,
        "phrase_id": item["id"],
        "phrase_note": item.get("note", ""),
        "prompt_speaker": int(style["speaker"]),
        "prompt_style": str(style["name"]),
        "expected": item["text"],
        "recognized": normalized_recognized,
        "recognized_raw": result_text,
        "raw_user": result_text,
        "corrected_user": normalized_recognized,
        "elapsed_sec": elapsed_sec,
        "similarity": similarity,
        "raw_similarity": raw_similarity,
        "judgement": verdict,
        "cohere_error": error_message,
        "alias_applied": alias_applied,
    }

    result_speech = build_result_speech(item["text"], normalized_recognized, verdict, "Cohere" if backend_used == "cohere" else "ローカル")
    safe_speak(result_speech, tts_ctx, f"{args.tag}_result.wav")

    print("===== PHRASE TEST =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("=======================")

    append_jsonl(out_dir / "phrase_test_results.jsonl", payload)
    append_event_sqlite(sqlite_path, payload)


if __name__ == "__main__":
    main()

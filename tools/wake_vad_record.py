import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import socket
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree
from pathlib import Path
from typing import Any

import requests
import soundfile as sf
import webrtcvad
import yaml
from src.audio_preprocess import prepare_whisper_audio
from src.llm_api import llm_chat, llm_chat_messages, llm_healthcheck, resolve_llm_config
from src.speech_recognition_runner import speech_recognition_once
from src.vosk_runner import vosk_once


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAG_GLOBS = [
    "README.md",
    "config/**/*.yaml",
    "logs/**/*.txt",
]


def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_cfg():
    cfg_override = os.environ.get("VOICECHAT_CONFIG", "").strip()
    p = Path(cfg_override) if cfg_override else (ROOT / "config" / "config.yaml")
    if not p.exists():
        die(f"NG: config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def is_music_playing(mpv_socket_path: str = "/tmp/voicechat-mpv.sock") -> bool:
    sock = Path(mpv_socket_path)
    if not sock.exists():
        return False
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(0.2)
        client.connect(str(sock))
        client.close()
        return True
    except OSError:
        return False


def get_music_volume_percent() -> int | None:
    commands = [
        ["amixer", "sget", "Master"],
        ["amixer", "-c", "0", "sget", "Master"],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True
            )
        except OSError:
            continue
        if proc.returncode != 0:
            continue
        match = re.search(r"(\d+)%", proc.stdout)
        if match:
            return int(match.group(1))
    return None


def set_music_volume_percent(volume: int) -> bool:
    volume = max(0, min(100, volume))
    commands = [
        ["amixer", "sset", "Master", f"{volume}%"],
        ["amixer", "-c", "0", "sset", "Master", f"{volume}%"],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError as exc:
            print(f"WARN: music volume restore failed: {exc}")
            continue
        if proc.returncode == 0:
            print(f"INFO: music volume set to {volume}% with: {' '.join(cmd)}")
            return True
    return False


def duck_music_volume(step: int = 10) -> int | None:
    current = get_music_volume_percent()
    if current is None:
        return None
    target = max(0, current - step)
    if set_music_volume_percent(target):
        print(f"INFO: music volume ducked: {current}% -> {target}%")
        return current
    return None


def restore_music_volume(volume: int | None) -> None:
    if volume is None:
        return
    if set_music_volume_percent(volume):
        print(f"INFO: music volume restored to {volume}%")


def resolve_recognition_mode_bundle(cfg: dict) -> tuple[str, dict[str, dict], dict]:
    runtime_cfg = cfg.get("runtime", {})
    wake_cfg = cfg.get("wake", {})
    whisper_cfg = cfg.get("whisper", {})
    profiles_cfg = cfg.get("recognition_profiles", {})
    if not isinstance(profiles_cfg, dict):
        profiles_cfg = {}

    default_name = (
        normalize_text(str(runtime_cfg.get("recognition_mode", "balanced")))
        or "balanced"
    )
    default_backend = (
        str(runtime_cfg.get("transcription_backend", "local")).strip() or "local"
    )
    default_ai_correction = bool(runtime_cfg.get("ai_correction", True))
    default_realtime_model = (
        str(whisper_cfg.get("realtime_model", "")).strip()
        or str(whisper_cfg.get("model", "")).strip()
    )
    default_final_model = str(whisper_cfg.get("model", "")).strip()
    default_wake_model = (
        str(wake_cfg.get("whisper_model", "")).strip() or default_final_model
    )
    default_wake_threads = int(
        wake_cfg.get("threads", int(whisper_cfg.get("threads", 4)))
    )

    profiles: dict[str, dict] = {}
    for name, profile in profiles_cfg.items():
        mode_name = normalize_text(str(name))
        if not mode_name or not isinstance(profile, dict):
            continue
        merged = {
            "label": normalize_text(str(profile.get("label", mode_name))) or mode_name,
            "backend": str(profile.get("backend", default_backend)).strip()
            or default_backend,
            "ai_correction": bool(profile.get("ai_correction", default_ai_correction)),
            "realtime_model": str(
                profile.get("realtime_model", default_realtime_model)
            ).strip()
            or default_realtime_model,
            "final_model": str(profile.get("final_model", default_final_model)).strip()
            or default_final_model,
            "wake_model": str(profile.get("wake_model", default_wake_model)).strip()
            or default_wake_model,
            "wake_threads": int(profile.get("wake_threads", default_wake_threads)),
            "description": normalize_text(str(profile.get("description", ""))),
        }
        profiles[mode_name] = merged

    if default_name not in profiles:
        default_name = next(iter(profiles.keys()), "balanced")
    selected = dict(profiles.get(default_name, {}))
    if not selected:
        selected = {
            "label": default_name,
            "backend": default_backend,
            "ai_correction": default_ai_correction,
            "realtime_model": default_realtime_model,
            "final_model": default_final_model,
            "wake_model": default_wake_model,
            "wake_threads": default_wake_threads,
            "description": "",
        }
        profiles[default_name] = dict(selected)
    return default_name, profiles, selected


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


def load_phrase_test_items(phrase_test_cfg: dict) -> list[dict[str, str]]:
    phrases = phrase_test_cfg.get("phrases")
    if isinstance(phrases, list) and phrases:
        items = [
            normalize_phrase_test_entry(item, idx)
            for idx, item in enumerate(phrases, start=1)
        ]
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
            raise RuntimeError(
                f'phrase_test json must be a list or {{"phrases": [...]}}: {path}'
            )
        items = [
            normalize_phrase_test_entry(item, idx)
            for idx, item in enumerate(data, start=1)
        ]
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


def normalize_wake_token(text: str) -> str:
    return normalize_text(text).replace(" ", "").replace("　", "").lower()


def normalize_kana_token(text: str) -> str:
    body = normalize_wake_token(text)
    if not body:
        return ""
    chars: list[str] = []
    for ch in body:
        code = ord(ch)
        # Katakana -> Hiragana
        if 0x30A1 <= code <= 0x30F6:
            ch = chr(code - 0x60)
        chars.append(ch)
    body = "".join(chars)
    # Reduce long-vowel marks because STT often varies there.
    return body.replace("ー", "")


def is_regex_alias_pattern(text: str) -> bool:
    body = normalize_text(text)
    if not body:
        return False
    return body.startswith("re:") or any(ch in body for ch in ".*+?[](){}|^$")


def wake_alias_matches_text(alias: str, body: str, body_kana: str) -> bool:
    token = normalize_wake_token(alias)
    if not token:
        return False
    if is_regex_alias_pattern(alias):
        pattern = normalize_text(alias)
        if pattern.startswith("re:"):
            pattern = pattern[3:].strip()
        try:
            if re.search(pattern, body):
                return True
            if body_kana and re.search(pattern, body_kana):
                return True
        except re.error:
            return False
        return False
    if token in body:
        return True
    if body_kana and normalize_kana_token(token) in body_kana:
        return True
    return False


def normalize_transcript_text(text: str) -> str:
    body = normalize_text(text)
    if body in {"[音声なし]", "(音声なし)", "（音声なし）", "[no audio]", "(no audio)"}:
        return ""
    return body


def load_wake_aliases(wake_cfg: dict, wake_words: list[str]) -> dict[str, set[str]]:
    alias_cfg = wake_cfg.get("aliases", {})
    aliases: dict[str, set[str]] = {}
    for wake_word in wake_words:
        canonical = normalize_text(wake_word)
        if not canonical:
            continue
        candidates = {
            normalize_wake_token(canonical),
            normalize_wake_token(canonical.lower()),
        }
        raw_aliases = alias_cfg.get(canonical, [])
        if isinstance(raw_aliases, list):
            for item in raw_aliases:
                alias = normalize_text(str(item))
                if not alias:
                    continue
                if is_regex_alias_pattern(alias):
                    candidates.add(alias)
                    continue
                token = normalize_wake_token(alias)
                if token:
                    candidates.add(token)
        aliases[canonical] = candidates
    return aliases


def merge_wake_aliases(
    base_aliases: dict[str, set[str]], learned_aliases: dict[str, set[str]]
) -> dict[str, set[str]]:
    merged = {key: set(value) for key, value in base_aliases.items()}
    for target, aliases in learned_aliases.items():
        merged.setdefault(target, set()).update(
            normalize_wake_token(item)
            for item in aliases
            if normalize_wake_token(item) and not is_regex_alias_pattern(item)
        )
    return merged


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


def run(
    cmd: list[str],
    *,
    stdin: bytes | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=stdin,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )


def wav_write_pcm16_mono(dst_wav: Path, pcm_list: list[bytes], sample_rate: int):
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
            sample = int(
                amplitude
                * math.sin(2.0 * math.pi * step_frequency_hz * (idx / sample_rate))
            )
            tone_bytes.extend(sample.to_bytes(2, byteorder="little", signed=True))
        pcm_chunks.append(bytes(tone_bytes))
        if step_gap_ms > 0:
            gap_frames = max(1, int(sample_rate * (step_gap_ms / 1000.0)))
            pcm_chunks.append(b"\x00\x00" * gap_frames)
    wav_write_pcm16_mono(dst_wav, pcm_chunks, sample_rate)


def load_wav_pcm16_mono(src_wav: Path) -> tuple[bytes, int]:
    with wave.open(str(src_wav), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if channels != 1 or sample_width != 2:
        raise RuntimeError(
            f"unsupported wav format for wake pipeline: channels={channels} sample_width={sample_width}"
        )
    return frames, sample_rate


def render_command_template(parts: list[object], values: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for part in parts:
        body = str(part)
        for key, value in values.items():
            body = body.replace("{" + key + "}", str(value))
        rendered.append(body)
    return rendered


def maybe_apply_rnnoise(
    src_wav: Path, workdir: Path, rnnoise_cfg: dict, *, tag: str
) -> Path:
    if not rnnoise_cfg.get("enabled", False):
        return src_wav
    command = [
        str(part) for part in rnnoise_cfg.get("command", []) if str(part).strip()
    ]
    if not command:
        raise RuntimeError("audio_pipeline.rnnoise.enabled=true but command is empty")
    out_wav = workdir / f"{tag}_rnnoise.wav"
    cmd = render_command_template(
        command, {"input": str(src_wav), "output": str(out_wav)}
    )
    result = run(cmd, capture=True)
    if result.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        err = (
            (result.stderr or b"").decode("utf-8", errors="replace")
            if result.stderr
            else ""
        )
        raise RuntimeError(f"rnnoise command failed: {' '.join(cmd)}\n{err}")
    return out_wav


def detect_silero_speech(wav: Path, silero_cfg: dict) -> tuple[bool, dict[str, Any]]:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception as exc:
        raise RuntimeError(
            "silero-vad is not installed. Install extras: uv sync --extra wake"
        ) from exc

    model = load_silero_vad()
    audio_array, sample_rate = sf.read(str(wav), dtype="float32", always_2d=False)
    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = audio_array[:, 0]
    if sample_rate != 16000:
        raise RuntimeError(f"Silero VAD requires 16000Hz audio, got {sample_rate}Hz")
    audio = torch.from_numpy(audio_array)
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=float(silero_cfg.get("threshold", 0.5)),
        min_speech_duration_ms=int(silero_cfg.get("min_speech_duration_ms", 120)),
        min_silence_duration_ms=int(silero_cfg.get("min_silence_duration_ms", 150)),
    )
    return bool(timestamps), {"segments": len(timestamps)}


def should_reset_command_as_no_speech(
    wav: Path,
    *,
    command_router_cfg: dict,
    silero_cfg: dict,
) -> tuple[bool, dict[str, Any]]:
    gate_cfg = command_router_cfg.get("silero_gate", {})
    if not gate_cfg.get("enabled", False):
        return False, {"enabled": False}
    has_speech, meta = detect_silero_speech(wav, silero_cfg)
    return (not has_speech), {"enabled": True, "has_speech": has_speech, **meta}


def detect_porcupine_wake(
    wav: Path, porcupine_cfg: dict, wake_words: list[str]
) -> tuple[bool, str, dict[str, Any]]:
    try:
        import pvporcupine
    except Exception as exc:
        raise RuntimeError(
            "pvporcupine is not installed. Install extras: uv sync --extra wake"
        ) from exc

    access_key_env = (
        str(porcupine_cfg.get("access_key_env", "PORCUPINE_ACCESS_KEY")).strip()
        or "PORCUPINE_ACCESS_KEY"
    )
    access_key = os.environ.get(access_key_env, "").strip()
    if not access_key:
        raise RuntimeError(f"Porcupine access key env is empty: {access_key_env}")

    keyword_paths_cfg = porcupine_cfg.get("keyword_paths", {})
    keyword_paths: list[str] = []
    keyword_labels: list[str] = []
    for wake_word in wake_words:
        keyword_path = str(keyword_paths_cfg.get(wake_word, "")).strip()
        if not keyword_path:
            continue
        keyword_path_obj = Path(keyword_path)
        if not keyword_path_obj.is_absolute():
            keyword_path_obj = ROOT / keyword_path_obj
        if not keyword_path_obj.exists():
            continue
        keyword_paths.append(str(keyword_path_obj))
        keyword_labels.append(wake_word)
    if not keyword_paths:
        raise RuntimeError(
            "wake.backend=porcupine but no keyword_paths were configured"
        )

    sensitivities_cfg = porcupine_cfg.get("sensitivities", {})
    sensitivities: list[float] = []
    for wake_word in keyword_labels:
        sensitivities.append(
            float(
                sensitivities_cfg.get(wake_word, porcupine_cfg.get("sensitivity", 0.6))
            )
        )

    create_kwargs: dict[str, Any] = {
        "access_key": access_key,
        "keyword_paths": keyword_paths,
        "sensitivities": sensitivities,
    }
    model_path = str(porcupine_cfg.get("model_path", "")).strip()
    if model_path:
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = ROOT / model_path_obj
        create_kwargs["model_path"] = str(model_path_obj)

    porcupine = pvporcupine.create(**create_kwargs)
    try:
        pcm_bytes, sample_rate = load_wav_pcm16_mono(wav)
        if sample_rate != porcupine.sample_rate:
            raise RuntimeError(
                f"Porcupine requires {porcupine.sample_rate}Hz audio, got {sample_rate}Hz"
            )
        frame_len = porcupine.frame_length
        samples = memoryview(pcm_bytes).cast("h")
        processed_frames = 0
        for idx in range(0, len(samples) - frame_len + 1, frame_len):
            result = porcupine.process(samples[idx : idx + frame_len])
            processed_frames += 1
            if result >= 0:
                matched = (
                    keyword_labels[result]
                    if result < len(keyword_labels)
                    else wake_words[0]
                )
                return True, matched, {"processed_frames": processed_frames}
        return False, "", {"processed_frames": processed_frames}
    finally:
        porcupine.delete()


def resolve_wake_backend(
    requested_backend: str,
    porcupine_cfg: dict,
    wake_words: list[str],
) -> tuple[str, str]:
    backend = normalize_text(requested_backend).lower() or "whisper"
    if backend in {"vosk", "speech_recognition"}:
        return backend, ""
    if backend != "porcupine":
        return backend, ""

    access_key_env = (
        str(porcupine_cfg.get("access_key_env", "PORCUPINE_ACCESS_KEY")).strip()
        or "PORCUPINE_ACCESS_KEY"
    )
    if not os.environ.get(access_key_env, "").strip():
        return "speech_recognition", f"Porcupine access key is missing: {access_key_env}"

    keyword_paths_cfg = porcupine_cfg.get("keyword_paths", {})
    available_count = 0
    missing_paths: list[str] = []
    for wake_word in wake_words:
        keyword_path = str(keyword_paths_cfg.get(wake_word, "")).strip()
        if not keyword_path:
            missing_paths.append(wake_word)
            continue
        keyword_path_obj = Path(keyword_path)
        if not keyword_path_obj.is_absolute():
            keyword_path_obj = ROOT / keyword_path_obj
        if not keyword_path_obj.exists():
            missing_paths.append(wake_word)
            continue
        available_count += 1
    if available_count == 0:
        return (
            "speech_recognition",
            f"Porcupine keyword files are missing for: {', '.join(missing_paths)}",
        )
    if missing_paths:
        return (
            "porcupine",
            f"Porcupine keyword files are missing for: {', '.join(missing_paths)}",
        )
    return "porcupine", ""


def available_porcupine_wake_words(
    porcupine_cfg: dict, wake_words: list[str]
) -> list[str]:
    keyword_paths_cfg = porcupine_cfg.get("keyword_paths", {})
    available: list[str] = []
    for wake_word in wake_words:
        keyword_path = str(keyword_paths_cfg.get(wake_word, "")).strip()
        if not keyword_path:
            continue
        keyword_path_obj = Path(keyword_path)
        if not keyword_path_obj.is_absolute():
            keyword_path_obj = ROOT / keyword_path_obj
        if keyword_path_obj.exists():
            available.append(wake_word)
    return available


def arecord_chunk_pcm(audio_in: str, sec: float, sample_rate: int) -> bytes:
    cmd = [
        "arecord",
        "-q",
        "-D",
        audio_in,
        "-d",
        str(max(1, round(sec))),
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-t",
        "raw",
    ]
    p = run(cmd, capture=True)
    if p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        die(f"NG: arecord failed exit={p.returncode}\n{err}")
    return p.stdout or b""


def whisper_transcribe_txt(
    whisper_bin: Path,
    whisper_model: Path,
    wav: Path,
    lang: str,
    threads: int,
    beam: int,
    best: int,
    temp: float,
) -> str:
    outbase = wav.parent / "out"
    txt = Path(str(outbase) + ".txt")
    whisper_env = whisper_env_for_bin(whisper_bin)
    prepared_wav = prepare_whisper_audio(wav, Path(str(outbase) + "_input.wav"))

    for suf in (".txt", ".json", ".srt", ".vtt"):
        try:
            Path(str(outbase) + suf).unlink()
        except FileNotFoundError:
            pass

    cmd = [
        str(whisper_bin),
        "-m",
        str(whisper_model),
        "-f",
        str(prepared_wav),
        "-l",
        lang,
        "-t",
        str(threads),
        "-bs",
        str(beam),
        "-bo",
        str(best),
        "-tp",
        str(temp),
        "-nt",
        "-otxt",
        "-of",
        str(outbase),
    ]
    rc = run(cmd, env=whisper_env).returncode
    if rc != 0:
        return ""

    if txt.exists() and txt.stat().st_size > 0:
        return normalize_text(txt.read_text(encoding="utf-8", errors="replace"))
    return ""


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def json_dumps_line(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def append_jsonl(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json_dumps_line(data) + "\n")


def write_runtime_state(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def play_wav_file(wav_path: Path, audio_out: str):
    play_cmd = ["aplay"]
    if audio_out:
        play_cmd += ["-D", audio_out]
    play_cmd.append(str(wav_path))
    run(play_cmd)


def init_event_db(path: Path):
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
        conn.execute("create index if not exists idx_events_ts on events(ts)")
        conn.execute("create index if not exists idx_events_type on events(event_type)")
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
        conn.execute(
            "create index if not exists idx_recognition_aliases_type_target on recognition_aliases(alias_type, target)"
        )
        conn.execute(
            "create index if not exists idx_recognition_aliases_alias on recognition_aliases(alias)"
        )


def append_db_event(path: Path, event_type: str, data: dict):
    payload = dict(data)
    ts = int(payload.get("ts") or time.time())
    date = str(payload.get("date") or time.strftime("%Y-%m-%d %H:%M:%S"))
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            insert into events (
                ts, date, event_type, mode, backend, model,
                expected, recognized, elapsed_sec, payload_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                date,
                event_type,
                payload.get("mode"),
                payload.get("backend"),
                payload.get("model"),
                payload.get("expected"),
                payload.get("recognized"),
                payload.get("elapsed_sec"),
                json.dumps(payload, ensure_ascii=False),
            ),
        )


def append_event_logs(
    *,
    payload: dict,
    event_type: str,
    jsonl_enabled: bool,
    jsonl_path: Path,
    sqlite_enabled: bool,
    sqlite_path: Path,
):
    if jsonl_enabled:
        append_jsonl(jsonl_path, payload)
    if sqlite_enabled:
        append_db_event(sqlite_path, event_type, payload)


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
):
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


def start_background_command(cmd: list[str], *, cwd: str = ""):
    subprocess.Popen(
        cmd,
        cwd=cwd or None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_for_condition(
    check_fn, timeout_sec: float, interval_sec: float = 1.0
) -> tuple[bool, str]:
    last_err = ""
    deadline = time.time() + max(0.5, timeout_sec)
    while time.time() < deadline:
        try:
            check_fn()
            return True, ""
        except Exception as exc:
            last_err = str(exc)
            time.sleep(interval_sec)
    return False, last_err


def check_http_health(url: str):
    resp = requests.get(url.rstrip("/"), timeout=2)
    resp.raise_for_status()


def ensure_service(
    *,
    name: str,
    service_cfg: dict,
    check_fn,
):
    if not service_cfg.get("enabled", False):
        return
    try:
        check_fn()
        return
    except Exception as exc:
        if not service_cfg.get("auto_start", False):
            raise RuntimeError(f"{name} not reachable: {exc}")
        start_cmd = [
            str(part)
            for part in service_cfg.get("start_command", [])
            if str(part).strip()
        ]
        if not start_cmd:
            raise RuntimeError(
                f"{name} not reachable and start_command is not configured: {exc}"
            )
        start_background_command(start_cmd, cwd=str(service_cfg.get("cwd", "")).strip())
        ok, last_err = wait_for_condition(
            check_fn, float(service_cfg.get("startup_wait_sec", 20))
        )
        if not ok:
            raise RuntimeError(f"{name} failed to start: {last_err or exc}")


def ensure_runtime_services(cfg: dict):
    services_cfg = cfg.get("services", {})
    runtime_cfg = cfg.get("runtime", {})
    tts_cfg = cfg.get("tts", {})
    llm_cfg = resolve_llm_config(cfg)
    tts_enabled = bool(tts_cfg.get("enabled", True))
    ai_correction_enabled = bool(runtime_cfg.get("ai_correction", True))
    transcription_backend = (
        str(runtime_cfg.get("transcription_backend", "local")).strip() or "local"
    )

    if tts_enabled:
        voicevox_cfg = dict(services_cfg.get("voicevox", {}))
        if voicevox_cfg:
            health_url = str(
                voicevox_cfg.get(
                    "health_url",
                    tts_cfg.get("engine_host", "http://127.0.0.1:50021").rstrip("/")
                    + "/version",
                )
            )
            ensure_service(
                name="voicevox",
                service_cfg=voicevox_cfg,
                check_fn=lambda: check_http_health(health_url),
            )

    if llm_cfg.get("provider") == "ollama" and (
        ai_correction_enabled or transcription_backend == "local"
    ):
        ollama_service_cfg = dict(services_cfg.get("ollama", {}))
        if ollama_service_cfg:
            ensure_service(
                name="ollama",
                service_cfg=ollama_service_cfg,
                check_fn=lambda: llm_healthcheck(llm_cfg),
            )


def build_memory_search_query(
    query: str, history: list[dict[str, str]], lookback_turns: int
) -> str:
    lines = [normalize_text(query)]
    recent_users = [turn["content"] for turn in history if turn.get("role") == "user"][
        -lookback_turns:
    ]
    for item in recent_users:
        body = normalize_text(item)
        if body and body not in lines:
            lines.append(body)
    return " ".join(lines).strip()


def retrieve_event_memories(
    sqlite_path: Path,
    query: str,
    *,
    top_k: int,
    min_score: float,
    scan_limit: int,
) -> list[dict]:
    if not sqlite_path.exists():
        return []
    body = normalize_text(query)
    if not body:
        return []
    rows: list[dict] = []
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select id, date, event_type, mode, expected, recognized, payload_json
            from events
            order by ts desc
            limit ?
            """,
            (scan_limit,),
        )
        for row in cursor.fetchall():
            payload_json = row[6] or "{}"
            try:
                payload = json.loads(payload_json)
            except Exception:
                payload = {}
            memory_text = " ".join(
                normalize_text(str(value))
                for value in [
                    row[4] or "",
                    row[5] or "",
                    payload.get("raw_user", ""),
                    payload.get("corrected_user", ""),
                    payload.get("assistant", ""),
                    payload.get("command_reply", ""),
                    payload.get("phrase_note", ""),
                ]
                if str(value).strip()
            )
            score = score_chunk(body, memory_text)
            if score < min_score:
                continue
            rows.append(
                {
                    "id": row[0],
                    "date": row[1],
                    "event_type": row[2],
                    "mode": row[3],
                    "expected": row[4] or "",
                    "recognized": row[5] or "",
                    "memory_text": memory_text,
                    "score": round(score, 4),
                }
            )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:top_k]


def format_memory_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["過去の会話・操作ログ。日付が必要な質問では優先して使うこと。"]
    for item in results:
        summary = item["recognized"] or item["memory_text"]
        lines.append(
            f"[{item['date']}] ({item['event_type']}/{item['mode']}) {summary}"
        )
    return "\n".join(lines)


def fetch_recent_raw_transcript(sqlite_path: Path) -> str:
    if not sqlite_path.exists():
        return ""
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select date, payload_json
            from events
            where payload_json like '%"raw_user"%'
            order by ts desc
            limit 20
            """
        )
        for date, payload_json in cursor.fetchall():
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            raw_user = normalize_text(str(payload.get("raw_user", "")))
            if raw_user:
                return f"{date} の生文字起こしは、{raw_user}"
    return ""


def fetch_today_raw_transcripts(sqlite_path: Path, today: str, limit: int = 5) -> str:
    if not sqlite_path.exists():
        return ""
    rows: list[str] = []
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select date, payload_json
            from events
            where date like ?
              and payload_json like '%"raw_user"%'
            order by ts desc
            limit ?
            """,
            (f"{today}%", limit * 4),
        )
        for date, payload_json in cursor.fetchall():
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            raw_user = normalize_text(str(payload.get("raw_user", "")))
            if not raw_user:
                continue
            rows.append(f"{date[-8:]} は {raw_user}")
            if len(rows) >= limit:
                break
    if not rows:
        return ""
    return "今日の生文字起こしは、" + "。".join(rows)


def build_ai_alias_map(ai_control_cfg: dict) -> dict[str, dict]:
    items = {}
    for item in ai_control_cfg.get("aliases", []):
        alias = normalize_text(str(item.get("alias", "")))
        if alias:
            items[alias] = item
    return items


def apply_ai_alias(llm_cfg: dict, alias_cfg: dict) -> dict:
    updated = dict(llm_cfg)
    updated["provider"] = str(alias_cfg["provider"])
    updated["model"] = str(alias_cfg["model"])
    return updated


def validate_llm_runtime(llm_cfg: dict) -> tuple[bool, str]:
    try:
        llm_healthcheck(llm_cfg)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def execute_internal_command(
    *,
    command_hit: dict,
    llm_cfg: dict,
    ai_alias_map: dict[str, dict],
    recognition_mode_name: str,
    recognition_profiles: dict[str, dict],
) -> tuple[bool, dict, str, str]:
    action_type = normalize_text(str(command_hit.get("action_type", "")))
    if not action_type:
        return False, llm_cfg, "", recognition_mode_name
    alias_name = normalize_text(str(command_hit.get("ai_alias", "")))
    alias_cfg = ai_alias_map.get(alias_name)
    if action_type == "internal_set_model_alias":
        if not alias_cfg:
            return (
                True,
                llm_cfg,
                "そのAI設定は見つからないのだ。",
                recognition_mode_name,
            )
        new_llm_cfg = apply_ai_alias(llm_cfg, alias_cfg)
        ok, err = validate_llm_runtime(new_llm_cfg)
        if not ok:
            return (
                True,
                llm_cfg,
                f"切り替えに失敗したのだ。{err}",
                recognition_mode_name,
            )
        provider = new_llm_cfg["provider"]
        model = new_llm_cfg["model"]
        return (
            True,
            new_llm_cfg,
            f"AIを {provider} の {model} に切り替えたのだ。",
            recognition_mode_name,
        )
    if action_type == "internal_pull_model_alias":
        if not alias_cfg:
            return (
                True,
                llm_cfg,
                "そのモデル設定は見つからないのだ。",
                recognition_mode_name,
            )
        pull_cmd = alias_cfg.get("pull_command", [])
        if not isinstance(pull_cmd, list) or not pull_cmd:
            return True, llm_cfg, "pull コマンドが未設定なのだ。", recognition_mode_name
        proc = run([str(part) for part in pull_cmd], capture=True)
        if proc.returncode != 0:
            stderr = normalize_text(
                (proc.stderr or b"").decode("utf-8", errors="replace")
            )
            stdout = normalize_text(
                (proc.stdout or b"").decode("utf-8", errors="replace")
            )
            detail = stderr or stdout or f"rc={proc.returncode}"
            return (
                True,
                llm_cfg,
                f"pull に失敗したのだ。{detail}",
                recognition_mode_name,
            )
        return (
            True,
            llm_cfg,
            f"{alias_cfg.get('model', alias_name)} を pull したのだ。",
            recognition_mode_name,
        )
    if action_type == "internal_set_recognition_mode":
        target_mode = normalize_text(str(command_hit.get("recognition_mode", "")))
        if not target_mode or target_mode not in recognition_profiles:
            return (
                True,
                llm_cfg,
                "その認識モードは見つからないのだ。",
                recognition_mode_name,
            )
        if target_mode == recognition_mode_name:
            label = (
                normalize_text(
                    str(recognition_profiles[target_mode].get("label", target_mode))
                )
                or target_mode
            )
            return (
                True,
                llm_cfg,
                f"認識モードは {label} のままなのだ。",
                recognition_mode_name,
            )
        label = (
            normalize_text(
                str(recognition_profiles[target_mode].get("label", target_mode))
            )
            or target_mode
        )
        return True, llm_cfg, f"認識モードを {label} に切り替えたのだ。", target_mode
    if action_type == "internal_show_recognition_mode":
        profile = recognition_profiles.get(recognition_mode_name, {})
        label = (
            normalize_text(str(profile.get("label", recognition_mode_name)))
            or recognition_mode_name
        )
        description = normalize_text(str(profile.get("description", "")))
        reply = f"今の認識モードは {label} なのだ。"
        if description:
            reply += f" {description}"
        return True, llm_cfg, reply, recognition_mode_name
    return False, llm_cfg, "", recognition_mode_name


def build_action_payload(item: dict) -> dict:
    args = item.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return {
        "action_type": normalize_text(str(item.get("action_type", ""))),
        "action_name": normalize_text(str(item.get("action_name", ""))),
        "args": args,
    }


def execute_action_runner(
    item: dict,
    command_router_cfg: dict,
    command_execution_enabled: bool,
) -> dict:
    payload = build_action_payload(item)
    action_type = payload["action_type"]
    if not action_type:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "missing action_type",
        }
    runner_cfg = command_router_cfg.get("action_runners", {}).get(action_type, {})
    if not isinstance(runner_cfg, dict):
        runner_cfg = {}
    runner_command = runner_cfg.get("command", [])
    if not isinstance(runner_command, list) or not runner_command:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"missing action runner for {action_type}",
        }
    if not command_execution_enabled:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "skipped": True,
            "payload": payload,
        }
    env = os.environ.copy()
    extra_env = runner_cfg.get("env", {})
    if isinstance(extra_env, dict):
        for key, value in extra_env.items():
            env[str(key)] = str(value)
    proc = subprocess.run(
        [str(part) for part in runner_command],
        input=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(runner_cfg.get("cwd", "")) or None,
        env=env,
        check=False,
    )
    stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "payload": payload,
    }
    if stdout_text.strip():
        try:
            parsed = json.loads(stdout_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            result["action_result"] = parsed
            if "success" in parsed:
                result["ok"] = bool(parsed.get("success"))
            if "message" in parsed:
                result["message"] = normalize_text(str(parsed.get("message", "")))
    return result


def split_text_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    body = normalize_text(text)
    if not body:
        return []
    if len(body) <= chunk_size:
        return [body]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(body), step):
        chunk = body[start : start + chunk_size]
        if len(chunk) < max(80, chunk_size // 4):
            continue
        chunks.append(chunk)
    return chunks or [body[:chunk_size]]


def char_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> set[str]:
    body = normalize_text(text).lower()
    if not body:
        return set()
    grams: set[str] = set()
    compact = body.replace(" ", "")
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


def load_rag_corpus(
    root: Path,
    patterns: list[str],
    chunk_size: int,
    chunk_overlap: int,
    max_files: int,
    max_file_chars: int,
) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if len(seen) > max_files:
                return chunks
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            text = text[:max_file_chars]
            for idx, chunk in enumerate(
                split_text_chunks(text, chunk_size, chunk_overlap), start=1
            ):
                chunks.append(
                    {
                        "path": str(path.relative_to(root)),
                        "chunk_id": f"{path.relative_to(root)}#{idx}",
                        "content": chunk,
                    }
                )
    return chunks


def weather_code_to_japanese(code: int) -> str:
    mapping = {
        0: "快晴",
        1: "おおむね晴れ",
        2: "薄曇り",
        3: "くもり",
        45: "霧",
        48: "霧氷",
        51: "弱い霧雨",
        53: "霧雨",
        55: "強い霧雨",
        61: "弱い雨",
        63: "雨",
        65: "強い雨",
        71: "弱い雪",
        73: "雪",
        75: "強い雪",
        80: "弱いにわか雨",
        81: "にわか雨",
        82: "激しいにわか雨",
        95: "雷雨",
    }
    return mapping.get(code, "不明")


def fetch_weather_topic(weather_cfg: dict) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": weather_cfg["latitude"],
        "longitude": weather_cfg["longitude"],
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "timezone": weather_cfg.get("timezone", "auto"),
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    current = r.json()["current"]
    return (
        f"{weather_cfg.get('label', '現在地')}の今の天気は"
        f"{weather_code_to_japanese(int(current['weather_code']))}、"
        f"気温{round(float(current['temperature_2m']))}度、"
        f"風速{round(float(current['wind_speed_10m']))}キロです。"
    )


def fetch_news_topic(news_cfg: dict) -> str:
    r = requests.get(news_cfg["rss_url"], timeout=15)
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)
    titles = []
    for item in root.findall(".//item"):
        title = item.findtext("title")
        if title:
            titles.append(normalize_text(title))
        if len(titles) >= int(news_cfg.get("count", 3)):
            break
    if not titles:
        return "ニュース見出しは取得できませんでした。"
    return "主なニュースは、" + " / ".join(titles) + " です。"


def build_duo_topic(duo_cfg: dict) -> str:
    parts = []
    topic_cfg = duo_cfg.get("topic_source", {})
    if topic_cfg.get("weather_enabled", False):
        try:
            parts.append(fetch_weather_topic(topic_cfg["weather"]))
        except Exception as exc:
            parts.append(f"天気情報の取得に失敗しました: {exc}")
    if topic_cfg.get("news_enabled", False):
        try:
            parts.append(fetch_news_topic(topic_cfg["news"]))
        except Exception as exc:
            parts.append(f"ニュース取得に失敗しました: {exc}")
    if not parts:
        return duo_cfg.get("topic", "最近の話題")
    return " ".join(parts)


def build_briefing(topic_source_cfg: dict) -> str:
    parts = []
    if topic_source_cfg.get("weather_enabled", False):
        try:
            parts.append(fetch_weather_topic(topic_source_cfg["weather"]))
        except Exception as exc:
            parts.append(f"天気情報の取得に失敗しました。{exc}")
    if topic_source_cfg.get("news_enabled", False):
        try:
            parts.append(fetch_news_topic(topic_source_cfg["news"]))
        except Exception as exc:
            parts.append(f"ニュース取得に失敗しました。{exc}")
    return " ".join(parts)


def retrieve_rag_context(
    query: str,
    corpus: list[dict[str, str]],
    top_k: int,
    min_score: float,
) -> list[dict[str, str]]:
    ranked = []
    for item in corpus:
        score = score_chunk(query, item["content"])
        if score >= min_score:
            ranked.append((score, item))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "score": round(score, 4),
            "path": item["path"],
            "chunk_id": item["chunk_id"],
            "content": item["content"],
        }
        for score, item in ranked[:top_k]
    ]


def format_rag_context(results: list[dict[str, str]]) -> str:
    if not results:
        return ""
    lines = ["参考メモ。関連がある時だけ使い、無関係なら無視すること。"]
    for item in results:
        lines.append(f"[{item['path']}] {item['content']}")
    return "\n".join(lines)


def summarize_history(
    llm_cfg: dict, current_summary: str, recent_turns: list[dict[str, str]]
) -> str:
    if not recent_turns:
        return current_summary
    transcript = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in recent_turns
    )
    system_prompt = (
        "あなたは会話メモ作成器。"
        "ユーザーの目的、好み、未解決事項、次にやることだけを日本語で短く整理する。"
        "冗長な言い回しは禁止。"
        "事実が不明なら書かない。"
    )
    user_prompt = (
        "既存メモ:\n"
        f"{current_summary or 'なし'}\n\n"
        "最近の会話:\n"
        f"{transcript}\n\n"
        "更新後メモを5行以内で返して。"
    )
    try:
        summary = llm_chat(llm_cfg, system_prompt, user_prompt)
    except Exception:
        return current_summary
    return summary or current_summary


def correct_transcript(llm_cfg: dict, raw_text: str) -> str:
    system_prompt = (
        "あなたは音声認識の補正器。"
        "誤変換、助詞の崩れ、不要な空白を自然な日本語に直す。"
        "話し言葉として不自然な箇所は、元の意味を保ったまま自然な口語に整える。"
        "短い相づちやくだけた言い回しは、勝手に硬くしすぎない。"
        "固有名詞や人名は確信がないならそのまま残す。"
        "意味を勝手に足さない。"
        "聞き取れない部分は無理に補わない。"
        "出力は補正後の本文だけにする。"
    )
    user_prompt = f"音声認識結果:\n{raw_text}\n\n補正後テキストだけを返して。"
    corrected = llm_chat(llm_cfg, system_prompt, user_prompt)
    return corrected or raw_text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def coerce_dynamic_capture(raw_value: str | None, spec: dict | str | None) -> object:
    if isinstance(spec, str):
        spec = {"target": spec}
    spec = spec or {}
    if raw_value in (None, ""):
        return spec.get("default")
    cast = str(spec.get("type", "str")).strip()
    if cast == "int":
        try:
            value = int(raw_value)
        except ValueError:
            value = spec.get("default")
    elif cast == "times_ten":
        try:
            value = int(raw_value) * 10
        except ValueError:
            value = spec.get("default")
    elif cast == "direct_percent":
        try:
            value = int(raw_value)
        except ValueError:
            value = spec.get("default")
    else:
        value = raw_value
    if isinstance(value, int):
        if "min" in spec:
            value = max(int(spec["min"]), value)
        if "max" in spec:
            value = min(int(spec["max"]), value)
    return value


def match_dynamic_command(text: str, command_router_cfg: dict) -> dict | None:
    body = compact_text(text).lower()
    if not body:
        return None
    patterns = command_router_cfg.get("dynamic_patterns", [])
    if not isinstance(patterns, list):
        return None
    for item in patterns:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern", "")).strip()
        if not pattern:
            continue
        matched = re.search(pattern, body)
        if not matched:
            continue
        command = dict(item)
        args = (
            dict(command.get("args", {}))
            if isinstance(command.get("args", {}), dict)
            else {}
        )
        captures = item.get("captures", {})
        if isinstance(captures, dict) and captures:
            for group_name, spec in captures.items():
                target = (
                    spec.get("target", group_name)
                    if isinstance(spec, dict)
                    else str(spec)
                )
                value = coerce_dynamic_capture(
                    matched.groupdict().get(group_name), spec
                )
                if target and value is not None:
                    args[str(target)] = value
        else:
            group_name = str(item.get("number_group", "value")).strip() or "value"
            raw_value = matched.groupdict().get(group_name)
            scale = str(item.get("number_scale", "direct_percent")).strip()
            spec = {
                "target": "value",
                "type": scale,
                "min": item.get("min"),
                "max": item.get("max"),
            }
            value = coerce_dynamic_capture(raw_value, spec)
            if value is not None:
                args["value"] = value
        command["args"] = args
        return command
    return None


def render_command_reply(template: str, command_hit: dict) -> str:
    body = normalize_text(template)
    if not body:
        return ""
    args = command_hit.get("args", {})
    if not isinstance(args, dict):
        return body
    try:
        return normalize_text(body.format(**args))
    except Exception:
        return body


def build_help_reply(command_router_cfg: dict) -> str:
    help_cfg = command_router_cfg.get("help", {})
    intro = normalize_text(str(help_cfg.get("intro", "使える操作は次のとおりなのだ。")))
    outro = normalize_text(
        str(help_cfg.get("outro", "ベリベリのあとに続けて言ってほしいのだ。"))
    )
    separator = normalize_text(str(help_cfg.get("separator", "、")))
    labels: list[str] = []

    for item in command_router_cfg.get("commands", []):
        if item.get("help_enabled", True) is False:
            continue
        label = normalize_text(str(item.get("help_label", "")))
        if not label:
            phrases = item.get("phrases", [])
            if phrases:
                label = normalize_text(str(phrases[0]))
        if label and label not in labels:
            labels.append(label)

    for item in command_router_cfg.get("dynamic_patterns", []):
        if item.get("help_enabled", True) is False:
            continue
        label = normalize_text(str(item.get("help_label", "")))
        if label and label not in labels:
            labels.append(label)

    max_items = int(help_cfg.get("max_items", 0) or 0)
    if max_items > 0:
        labels = labels[:max_items]
    body = separator.join(labels)
    parts = [part for part in [intro, body, outro] if part]
    return normalize_text(" ".join(parts))


def command_text_matches_any(text: str, phrases: list[str]) -> bool:
    body = compact_text(text).lower()
    if not body:
        return False
    return any(
        compact_text(str(item)).lower() == body
        for item in phrases
        if normalize_text(str(item))
    )


def is_command_reset_phrase(text: str, command_router_cfg: dict) -> bool:
    confirm_cfg = command_router_cfg.get("candidate_confirmation", {})
    phrases = confirm_cfg.get(
        "reset_phrases", ["何でもない", "なんでもない", "キャンセル", "なし", "やめる"]
    )
    if not isinstance(phrases, list):
        return False
    return command_text_matches_any(text, [str(item) for item in phrases])


def is_confirmation_yes(text: str, command_router_cfg: dict) -> bool:
    confirm_cfg = command_router_cfg.get("candidate_confirmation", {})
    phrases = confirm_cfg.get("yes_phrases", ["はい", "うん", "お願い", "それ"])
    if not isinstance(phrases, list):
        return False
    return command_text_matches_any(text, [str(item) for item in phrases])


def is_confirmation_no(text: str, command_router_cfg: dict) -> bool:
    confirm_cfg = command_router_cfg.get("candidate_confirmation", {})
    phrases = confirm_cfg.get("no_phrases", ["いいえ", "違う", "ちがう", "違います"])
    if not isinstance(phrases, list):
        return False
    return command_text_matches_any(text, [str(item) for item in phrases])


def choose_command_confirmation_candidate(
    command_candidates: list[dict[str, str | float]],
    command_router_cfg: dict,
) -> tuple[dict | None, str]:
    confirm_cfg = command_router_cfg.get("candidate_confirmation", {})
    if not confirm_cfg.get("enabled", True):
        return None, ""
    if not command_candidates:
        return None, ""

    min_score = float(confirm_cfg.get("min_score", 0.45))
    top_k = max(1, int(confirm_cfg.get("top_k", 2)))
    unique_candidates: list[dict[str, str | float]] = []
    seen_phrases: set[str] = set()
    for item in command_candidates:
        phrase = normalize_text(str(item.get("phrase", "")))
        if not phrase or phrase in seen_phrases:
            continue
        seen_phrases.add(phrase)
        unique_candidates.append(item)
        if len(unique_candidates) >= top_k:
            break
    if not unique_candidates:
        return None, ""

    best = unique_candidates[0]
    best_phrase = normalize_text(str(best.get("phrase", "")))
    best_score = float(best.get("score", 0.0))
    if not best_phrase or best_score < min_score:
        return None, ""

    candidate_item = match_command(best_phrase, command_router_cfg)
    if not candidate_item:
        return None, ""

    alternatives = [
        normalize_text(str(item.get("phrase", "")))
        for item in unique_candidates[1:]
        if normalize_text(str(item.get("phrase", "")))
    ]
    if alternatives:
        template = normalize_text(
            str(
                confirm_cfg.get(
                    "reply_with_alternatives",
                    "もしかして {primary} なのだ？ 候補には {alternatives} もあるのだ。{primary} でよければ はい、違うなら言い直してほしいのだ。",
                )
            )
        )
        return candidate_item, template.format(
            primary=best_phrase, alternatives="、".join(alternatives)
        )

    template = normalize_text(
        str(
            confirm_cfg.get(
                "reply_single",
                "もしかして {primary} なのだ？ はい か いいえ、または言い直してほしいのだ。",
            )
        )
    )
    return candidate_item, template.format(primary=best_phrase)


def judge_phrase_result(llm_cfg: dict, expected_text: str, recognized_text: str) -> str:
    system_prompt = (
        "あなたは音声認識の評価器。"
        "期待文と認識文を比べて、意味として一致しているかを判定する。"
        "出力は次のどれか1語だけにする: MATCH / NEAR / MISMATCH"
    )
    user_prompt = (
        f"期待文: {expected_text}\n"
        f"認識文: {recognized_text}\n\n"
        "意味が一致なら MATCH、ほぼ同じなら NEAR、大きく違うなら MISMATCH。"
    )
    try:
        verdict = llm_chat(llm_cfg, system_prompt, user_prompt)
    except Exception:
        return "UNKNOWN"
    verdict = normalize_text(verdict).upper()
    if "MATCH" in verdict and "MISMATCH" not in verdict:
        return "MATCH"
    if "NEAR" in verdict:
        return "NEAR"
    if "MISMATCH" in verdict:
        return "MISMATCH"
    return "UNKNOWN"


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


def contains_wake(
    text: str,
    wake_word: str,
    wake_aliases: dict[str, set[str]] | None = None,
    wake_cfg: dict | None = None,
) -> bool:
    if not text:
        return False
    body = normalize_wake_token(text)
    body_kana = normalize_kana_token(text)
    canonical = normalize_text(wake_word)
    canonical_kana = normalize_kana_token(canonical)
    match_threshold = max(0.72, float((wake_cfg or {}).get("match_threshold", 0.74)))
    short_similarity_threshold = max(
        0.84, float((wake_cfg or {}).get("short_similarity_threshold", 0.88))
    )
    candidates = (
        wake_aliases.get(canonical)
        if wake_aliases and canonical in wake_aliases
        else {normalize_wake_token(canonical), normalize_wake_token(canonical.lower())}
    )
    for candidate in candidates:
        if wake_alias_matches_text(candidate, body, body_kana):
            return True
    if body_kana == canonical_kana:
        return True
    if (
        canonical_kana
        and abs(len(body_kana) - len(canonical_kana)) <= 1
        and score_chunk(body_kana, canonical_kana) >= match_threshold
    ):
        return True
    # For short recognized text, also try edit distance (helps with STT misrecognitions)
    if len(body) <= 10 and len(body) >= 2:
        for candidate in candidates:
            cand_kana = normalize_kana_token(candidate)
            if not cand_kana or abs(len(body_kana) - len(cand_kana)) > 1:
                continue
            if _short_text_similarity(body_kana, cand_kana) >= short_similarity_threshold:
                return True
    return False


def _short_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    # Levenshtein-based similarity for short texts
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    # Simple edit distance
    if len1 > len2:
        text1, text2 = text2, text1
        len1, len2 = len2, len1
    current_row = range(len1 + 1)
    for i in range(1, len2 + 1):
        previous_row, current_row = current_row, [i] + [0] * len1
        for j in range(1, len1 + 1):
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1]
            if text1[j - 1] != text2[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    distance = current_row[len1]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def is_wake_blocked(text: str, wake_cfg: dict) -> bool:
    body = normalize_wake_token(text)
    body_kana = normalize_kana_token(text)
    if not body:
        return False
    blocklist = wake_cfg.get("blocklist", [])
    if not isinstance(blocklist, list):
        return False
    for item in blocklist:
        token = normalize_wake_token(str(item))
        if token and token in body:
            return True
        kana = normalize_kana_token(str(item))
        if kana and score_chunk(body_kana, kana) >= 0.82:
            return True
    return False


def contains_any_wake(
    text: str,
    wake_words: list[str],
    wake_cfg: dict | None = None,
    wake_aliases: dict[str, set[str]] | None = None,
) -> tuple[bool, str]:
    if wake_cfg and is_wake_blocked(text, wake_cfg):
        return False, ""
    for wake_word in wake_words:
        if contains_wake(text, wake_word, wake_aliases, wake_cfg):
            return True, wake_word
    return False, ""


def extract_inline_command_from_wake(
    text: str,
    wake_words: list[str],
    wake_aliases: dict[str, set[str]] | None = None,
) -> tuple[str, str]:
    body = normalize_wake_token(text)
    if not body:
        return "", ""

    candidates: list[tuple[str, str]] = []
    for wake_word in wake_words:
        canonical = normalize_text(wake_word)
        if not canonical:
            continue
        aliases = (
            wake_aliases.get(canonical)
            if wake_aliases and canonical in wake_aliases
            else {
                normalize_wake_token(canonical),
                normalize_wake_token(canonical.lower()),
            }
        )
        for alias in aliases:
            if is_regex_alias_pattern(alias):
                continue
            token = normalize_wake_token(alias)
            if token:
                candidates.append((token, canonical))

    # Prefer the longest alias so "ベリベリベリ" does not get shortened to "ベリベリ".
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    for token, canonical in candidates:
        if token and body.startswith(token):
            remainder = body
            while remainder.startswith(token):
                remainder = remainder[len(token) :]
            remainder = normalize_text(re.sub(r"^[、。,．・!！?？ー\-]+", "", remainder))
            if not remainder or len(compact_text(remainder)) <= 1:
                return "", ""
            return remainder, canonical
        idx = body.find(token)
        if idx <= 0:
            continue
        remainder = body[:idx] + body[idx + len(token) :]
        remainder = normalize_text(re.sub(r"^[、。,．・!！?？ー\-]+", "", remainder))
        if remainder and len(compact_text(remainder)) > 1:
            return remainder, canonical
    return "", ""


def is_speech_frame(
    vad: webrtcvad.Vad, pcm16: bytes, sample_rate: int, frame_ms: int = 30
) -> tuple[int, int]:
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
    if frame_bytes <= 0:
        return (0, 0)
    total = 0
    speech = 0
    for i in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
        total += 1
        if vad.is_speech(pcm16[i : i + frame_bytes], sample_rate):
            speech += 1
    return (speech, total)


def transcribe_local(
    *,
    whisper_bin: Path,
    whisper_model: Path,
    wav: Path,
    lang: str,
    threads: int,
    beam: int,
    best: int,
    temp: float,
) -> tuple[str, float, str]:
    started_at = time.time()
    text = whisper_transcribe_txt(
        whisper_bin, whisper_model, wav, lang, threads, beam, best, temp
    )
    elapsed_sec = round(time.time() - started_at, 3)
    return text, elapsed_sec, str(whisper_model)


def transcribe_vosk(
    *,
    vosk_model: Path,
    wav: Path,
    boost_volume: float = 10.0,
) -> tuple[str, float, str]:
    started_at = time.time()
    rc, _, text = vosk_once(vosk_model, wav, log_path=None, boost_volume=boost_volume)
    elapsed_sec = round(time.time() - started_at, 3)
    if rc != 0:
        return f"[vosk error] {text}", elapsed_sec, str(vosk_model)
    return text, elapsed_sec, str(vosk_model)


def transcribe_speech_recognition(
    *,
    wav: Path,
    speech_recognition_cfg: dict | None = None,
    boost_volume: float = 1.0,
) -> tuple[str, float, str]:
    cfg = speech_recognition_cfg or {}
    engine = str(cfg.get("engine", "google")).strip() or "google"
    language = str(cfg.get("language", "ja-JP")).strip() or "ja-JP"
    show_all = bool(cfg.get("show_all", False))
    timeout_sec = float(cfg.get("timeout_sec", 2.5))
    rc, elapsed_sec, text = speech_recognition_once(
        wav=wav,
        engine=engine,
        language=language,
        show_all=show_all,
        boost_volume=boost_volume,
        request_timeout_sec=timeout_sec,
    )
    if rc != 0:
        return text or "NG: speech_recognition error", elapsed_sec, f"speech_recognition:{engine}"
    return text, elapsed_sec, f"speech_recognition:{engine}"


def transcribe_remote(
    *,
    wav: Path,
    workdir: Path,
    remote_cfg: dict,
    llm_cfg: dict,
) -> tuple[str, str, float, str]:
    ssh_key = str(remote_cfg["ssh_key"])
    remote_user = str(remote_cfg["user"])
    remote_host = str(remote_cfg["host"])
    remote_port = str(remote_cfg.get("port", 22))
    remote_workdir = str(remote_cfg["remote_workdir"]).rstrip("/")
    remote_python = str(remote_cfg["remote_python"])
    remote_script = str(remote_cfg["remote_script"])
    remote_model = str(remote_cfg.get("whisper_model", "large-v3"))
    remote_device = str(remote_cfg.get("device", "cpu"))
    remote_compute_type = str(remote_cfg.get("compute_type", "int8"))
    remote_skip_correction = bool(remote_cfg.get("skip_correction", False))

    tag = f"{wav.stem}_{int(time.time())}"
    remote_incoming = f"{remote_workdir}/incoming/{wav.name}"
    remote_out_prefix = f"{remote_workdir}/results/{tag}"
    local_remote_dir = workdir / "remote_results"
    local_remote_dir.mkdir(parents=True, exist_ok=True)
    ssh_target = f"{remote_user}@{remote_host}"

    mkdir_cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-p",
        remote_port,
        ssh_target,
        f"mkdir -p {remote_workdir}/incoming {remote_workdir}/results",
    ]
    if run(mkdir_cmd).returncode != 0:
        raise RuntimeError("remote mkdir failed")

    scp_push_cmd = [
        "scp",
        "-i",
        ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-P",
        remote_port,
        str(wav),
        f"{ssh_target}:{remote_incoming}",
    ]
    if run(scp_push_cmd).returncode != 0:
        raise RuntimeError("remote wav upload failed")

    remote_cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-p",
        remote_port,
        ssh_target,
        remote_python,
        remote_script,
        "--wav",
        remote_incoming,
        "--out-prefix",
        remote_out_prefix,
        "--model",
        remote_model,
        "--device",
        remote_device,
        "--compute-type",
        remote_compute_type,
        "--llm-provider",
        str(llm_cfg["provider"]),
        "--llm-model",
        str(llm_cfg["model"]),
        "--llm-timeout",
        str(int(llm_cfg.get("timeout_sec", 120))),
    ]
    if llm_cfg["provider"] == "ollama":
        remote_cmd.extend(
            [
                "--ollama-host",
                str(llm_cfg["host"]),
                "--ollama-api-key",
                str(llm_cfg.get("api_key", "")),
            ]
        )
    elif llm_cfg["provider"] == "gemini":
        remote_cmd.extend(
            [
                "--gemini-api-base",
                str(llm_cfg["api_base"]),
                "--gemini-api-key",
                str(llm_cfg["api_key"]),
            ]
        )
    elif llm_cfg["provider"] == "openai":
        remote_cmd.extend(
            [
                "--openai-api-base",
                str(llm_cfg["api_base"]),
                "--openai-api-key",
                str(llm_cfg["api_key"]),
            ]
        )
    elif llm_cfg["provider"] == "anthropic":
        remote_cmd.extend(
            [
                "--anthropic-api-base",
                str(llm_cfg["api_base"]),
                "--anthropic-api-key",
                str(llm_cfg["api_key"]),
                "--anthropic-version",
                str(llm_cfg["anthropic_version"]),
            ]
        )
    if remote_skip_correction:
        remote_cmd.append("--skip-correct")
    remote_run = run(remote_cmd, capture=True)
    if remote_run.returncode != 0:
        err = (remote_run.stderr or b"").decode("utf-8", errors="replace")
        out = (remote_run.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"remote transcribe failed: {out}\n{err}")

    remote_result = json.loads(
        (remote_run.stdout or b"{}").decode("utf-8", errors="replace").strip() or "{}"
    )
    raw_remote = remote_result.get("raw_path") or (remote_out_prefix + "_raw.txt")
    corrected_remote = remote_result.get("corrected_path") or (
        remote_out_prefix + "_corrected.txt"
    )
    meta_remote = remote_result.get("meta_path") or (remote_out_prefix + ".json")

    for remote_file in (raw_remote, corrected_remote, meta_remote):
        scp_pull_cmd = [
            "scp",
            "-i",
            ssh_key,
            "-o",
            "IdentitiesOnly=yes",
            "-P",
            remote_port,
            f"{ssh_target}:{remote_file}",
            str(local_remote_dir / Path(remote_file).name),
        ]
        if run(scp_pull_cmd).returncode != 0:
            raise RuntimeError(f"remote result download failed: {remote_file}")

    raw_text = normalize_text(
        (local_remote_dir / Path(raw_remote).name).read_text(
            encoding="utf-8", errors="replace"
        )
    )
    corrected_text = normalize_text(
        (local_remote_dir / Path(corrected_remote).name).read_text(
            encoding="utf-8", errors="replace"
        )
    )
    meta = json.loads(
        (local_remote_dir / Path(meta_remote).name).read_text(
            encoding="utf-8", errors="replace"
        )
    )
    elapsed_sec = float(meta.get("elapsed_sec", 0.0))
    return raw_text, corrected_text, elapsed_sec, remote_model


def _google_transcribe(wav: Path, cfg: dict[str, object] | None) -> tuple[str, float]:
    try:
        from tools.google_stt import transcribe_google
    except RuntimeError as exc:
        raise RuntimeError(f"Google Speech-to-Text helper unavailable: {exc}") from exc
    return transcribe_google(wav, cfg)


def transcribe_audio(
    *,
    backend: str,
    whisper_bin: Path,
    whisper_model: Path,
    vosk_model: Path | None,
    wav: Path,
    lang: str,
    threads: int,
    beam: int,
    best: int,
    temp: float,
    workdir: Path,
    remote_cfg: dict,
    llm_cfg: dict,
    google_cfg: dict[str, object] | None = None,
    speech_recognition_cfg: dict[str, object] | None = None,
    rnnoise_cfg: dict | None = None,
    preprocess_tag: str = "",
) -> tuple[str, str, float, str]:
    prepared_wav = wav
    if rnnoise_cfg:
        prepared_wav = maybe_apply_rnnoise(
            wav, workdir, rnnoise_cfg, tag=preprocess_tag or wav.stem
        )
    if backend == "ssh_remote":
        raw_text, corrected_text, elapsed_sec, model_name = transcribe_remote(
            wav=prepared_wav,
            workdir=workdir,
            remote_cfg=remote_cfg,
            llm_cfg=llm_cfg,
        )
        return raw_text, corrected_text, elapsed_sec, model_name

    if backend == "google":
        raw_text, elapsed_sec = _google_transcribe(prepared_wav, google_cfg or {})
        return raw_text, raw_text, elapsed_sec, "google"

    if backend == "vosk":
        if not vosk_model:
            return f"[vosk error] model not configured", "", 0.0, "vosk"
        raw_text, elapsed_sec, model_name = transcribe_vosk(
            vosk_model=vosk_model,
            wav=prepared_wav,
        )
        return raw_text, raw_text, elapsed_sec, model_name

    if backend == "speech_recognition":
        raw_text, elapsed_sec, model_name = transcribe_speech_recognition(
            wav=prepared_wav,
            speech_recognition_cfg=speech_recognition_cfg,
        )
        return raw_text, raw_text, elapsed_sec, model_name

    raw_text, elapsed_sec, model_name = transcribe_local(
        whisper_bin=whisper_bin,
        whisper_model=whisper_model,
        wav=prepared_wav,
        lang=lang,
        threads=threads,
        beam=beam,
        best=best,
        temp=temp,
    )
    return raw_text, raw_text, elapsed_sec, model_name


def match_command(text: str, command_router_cfg: dict) -> dict | None:
    if not command_router_cfg.get("enabled", False):
        return None
    dynamic_hit = match_dynamic_command(text, command_router_cfg)
    if dynamic_hit:
        return dynamic_hit
    body = compact_text(text).lower()
    if not body:
        return None
    for item in command_router_cfg.get("commands", []):
        phrases = item.get("phrases", [])
        for phrase in phrases:
            if compact_text(str(phrase)).lower() in body:
                return item
    return None


def build_command_phrase_index(
    command_router_cfg: dict,
    learned_aliases: dict[str, set[str]] | None = None,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    learned_aliases = learned_aliases or {}
    for item in command_router_cfg.get("commands", []):
        command_id = normalize_text(str(item.get("id", "")))
        reply = normalize_text(str(item.get("reply", "")))
        phrases = [
            normalize_text(str(phrase))
            for phrase in item.get("phrases", [])
            if normalize_text(str(phrase))
        ]
        if not phrases:
            continue
        canonical_phrase = phrases[0]
        for phrase in phrases:
            items.append(
                {
                    "command_id": command_id,
                    "phrase": canonical_phrase,
                    "match_text": phrase,
                    "reply": reply,
                }
            )
        for alias in learned_aliases.get(command_id, set()):
            alias_text = normalize_text(str(alias))
            if not alias_text:
                continue
            items.append(
                {
                    "command_id": command_id,
                    "phrase": canonical_phrase,
                    "match_text": alias_text,
                    "reply": reply,
                }
            )
    return items


def retrieve_command_phrase_candidates(
    text: str,
    phrase_index: list[dict[str, str]],
    *,
    top_k: int,
    min_score: float,
) -> list[dict[str, str | float]]:
    body = normalize_text(text)
    if not body:
        return []
    rows: list[dict[str, str | float]] = []
    for item in phrase_index:
        score = score_chunk(body, str(item.get("match_text", item["phrase"])))
        if score < min_score:
            continue
        rows.append(
            {
                "command_id": item["command_id"],
                "phrase": item["phrase"],
                "match_text": item.get("match_text", item["phrase"]),
                "reply": item["reply"],
                "score": round(score, 4),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows[:top_k]


def normalize_command_text(
    text: str,
    phrase_index: list[dict[str, str]],
    command_router_cfg: dict,
    llm_cfg: dict,
) -> tuple[str, list[dict[str, str | float]]]:
    normalization_cfg = command_router_cfg.get("normalization", {})
    if not normalization_cfg.get("enabled", False):
        return text, []

    candidates = retrieve_command_phrase_candidates(
        text,
        phrase_index,
        top_k=max(1, int(normalization_cfg.get("top_k", 5))),
        min_score=float(normalization_cfg.get("min_score", 0.32)),
    )
    if not candidates:
        return text, []

    best = candidates[0]
    best_phrase = str(best["phrase"])
    best_score = float(best["score"])
    if best_score >= float(normalization_cfg.get("auto_apply_score", 0.72)):
        return best_phrase, candidates

    if not normalization_cfg.get("llm_enabled", True):
        return text, candidates

    lines = ["候補一覧。次のどれか1つだけをそのまま返す。該当なしなら NONE を返す。"]
    for idx, item in enumerate(candidates, start=1):
        lines.append(f"{idx}. {item['phrase']} (score={item['score']})")
    system_prompt = (
        "あなたは音声コマンド正規化器。"
        "認識結果を既知コマンド候補へ正規化する。"
        "候補にない意味を作らない。"
        "出力は候補の文言そのものか NONE のみ。"
    )
    user_prompt = f"認識結果: {text}\n\n" + "\n".join(lines)
    try:
        result = normalize_text(llm_chat(llm_cfg, system_prompt, user_prompt))
    except Exception:
        return text, candidates
    if result == "NONE":
        return text, candidates
    for item in candidates:
        if normalize_text(str(item["phrase"])) == result:
            return result, candidates
    return text, candidates


def execute_command_action(
    item: dict, command_router_cfg: dict, command_execution_enabled: bool
) -> dict:
    action_type = normalize_text(str(item.get("action_type", "")))
    if action_type:
        return execute_action_runner(
            item, command_router_cfg, command_execution_enabled
        )
    command = item.get("command", [])
    if not isinstance(command, list) or not command:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "missing command",
        }
    if not command_execution_enabled:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "skipped": True,
        }
    proc = run([str(part) for part in command], capture=True)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or b"").decode("utf-8", errors="replace"),
        "stderr": (proc.stderr or b"").decode("utf-8", errors="replace"),
    }


def render_timed_record_end_text(template: str, seconds: int) -> str:
    minutes = max(1, round(seconds / 60))
    return template.format(seconds=seconds, minutes=minutes)


def build_chat_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_text: str,
    memory_summary: str = "",
    rag_context: str = "",
    event_memory_context: str = "",
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if memory_summary:
        messages.append({"role": "system", "content": f"会話メモ:\n{memory_summary}"})
    if event_memory_context:
        messages.append({"role": "system", "content": event_memory_context})
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def speak_with_voicevox(
    text: str,
    audio_out: str,
    out_wav: Path,
    engine_host: str,
    speaker: int,
    speed_scale: float,
    pitch_scale: float,
    intonation_scale: float,
    volume_scale: float,
):
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

    play_cmd = ["aplay"]
    if audio_out:
        play_cmd += ["-D", audio_out]
    play_cmd += [str(out_wav)]
    run(play_cmd)


def run_duo_mode(
    *,
    mode_cfg: dict,
    speak,
    llm_cfg: dict,
):
    duo_cfg = mode_cfg.get("duo", {})
    participants = duo_cfg.get("participants", [])
    if len(participants) < 2:
        die("NG: ai_duo mode requires two participants")

    topic = build_duo_topic(duo_cfg)
    opening = duo_cfg.get("opening", f"今日の話題は、{topic}")
    max_turns = int(duo_cfg.get("max_turns", 0))
    pause_sec = float(duo_cfg.get("pause_sec", 0.4))
    llm_options = duo_cfg.get("ollama_options", {})
    history_lines: list[str] = []
    last_line = opening

    def generate_reply(
        actor: dict, current_last_line: str, current_history_lines: list[str]
    ) -> str:
        actor_name = actor["name"]
        style = actor.get("style_prompt", "")
        partner_lines = "\n".join(current_history_lines[-6:])
        user_prompt = (
            f"会話テーマ: {topic}\n"
            f"直前の発言: {current_last_line}\n"
            f"最近の会話:\n{partner_lines}\n\n"
            f"{actor_name}として1文で自然に続けてください。"
        )
        if style:
            user_prompt += f"\n話し方の条件: {style}"
        reply = llm_chat(llm_cfg, actor["system_prompt"], user_prompt, llm_options)
        return reply or "少し考え込んでしまったのだ。"

    print("INFO: AI DUO MODE")
    print(f"INFO: topic={topic}")
    speak(opening, "duo_opening.wav", speaker_override=participants[0].get("speaker"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        turn = 0
        reply = generate_reply(participants[0], last_line, history_lines)

        while max_turns <= 0 or turn < max_turns:
            actor = participants[turn % len(participants)]
            actor_name = actor["name"]
            line = f"{actor_name}: {reply}"
            print(line, flush=True)
            history_lines.append(line)
            last_line = reply

            speak_future = executor.submit(
                speak,
                reply,
                f"duo_turn_{turn % len(participants)}.wav",
                actor.get("speaker"),
            )

            turn += 1
            if max_turns > 0 and turn >= max_turns:
                speak_future.result()
                break

            next_actor = participants[turn % len(participants)]
            next_reply = generate_reply(next_actor, last_line, history_lines)
            speak_future.result()
            time.sleep(pause_sec)
            reply = next_reply


def sanity(cfg):
    whisper_cfg = cfg["whisper"]
    tts_cfg = cfg["tts"]
    runtime_cfg = cfg.get("runtime", {})
    wake_cfg = cfg.get("wake", {})
    recognition_mode_name, _, recognition_profile = resolve_recognition_mode_bundle(cfg)
    llm_cfg = resolve_llm_config(cfg)
    whisper_bin = Path(
        os.environ.get("VOICECHAT_WHISPER_BIN", "").strip() or whisper_cfg["bin"]
    )
    realtime_whisper_model = Path(
        str(recognition_profile.get("realtime_model", "")).strip()
        or str(whisper_cfg.get("realtime_model", "")).strip()
        or str(whisper_cfg["model"])
    )
    whisper_model = Path(
        str(recognition_profile.get("final_model", "")).strip()
        or os.environ.get("VOICECHAT_WHISPER_MODEL", "").strip()
        or str(whisper_cfg["model"])
    )
    wake_whisper_model = Path(
        str(recognition_profile.get("wake_model", "")).strip()
        or str(wake_cfg.get("whisper_model", "")).strip()
        or str(whisper_model)
    )
    tts_enabled = os.environ.get("VOICECHAT_TTS_ENABLED", "").strip()
    tts_enabled_value = (
        tts_cfg.get("enabled", True)
        if not tts_enabled
        else tts_enabled.lower() in {"1", "true", "yes", "on"}
    )
    transcription_backend = str(
        recognition_profile.get(
            "backend", runtime_cfg.get("transcription_backend", "local")
        )
    )
    ai_correction_enabled = bool(
        recognition_profile.get("ai_correction", runtime_cfg.get("ai_correction", True))
    )

    if shutil.which("arecord") is None:
        die("NG: arecord not found")
    if shutil.which("aplay") is None:
        die("NG: aplay not found")
    if transcription_backend == "local" and not whisper_bin.exists():
        die(f"NG: whisper bin missing: {whisper_bin}")
    if transcription_backend == "local" and not realtime_whisper_model.exists():
        die(f"NG: realtime whisper model missing: {realtime_whisper_model}")
    if transcription_backend == "local" and not whisper_model.exists():
        die(f"NG: whisper model missing: {whisper_model}")
    if transcription_backend == "local" and not wake_whisper_model.exists():
        die(f"NG: wake whisper model missing: {wake_whisper_model}")
    if transcription_backend == "speech_recognition" or str(
        wake_cfg.get("backend", "whisper")
    ).strip().lower() == "speech_recognition":
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            die(
                "NG: speech_recognition backend selected but SpeechRecognition is not installed"
            )
    if transcription_backend == "vosk":
        vosk_cfg = cfg.get("vosk", {})
        vosk_model_path = vosk_cfg.get("model") or vosk_cfg.get("realtime_model")
        if not vosk_model_path:
            die("NG: vosk model not configured in vosk section")
        vosk_model = Path(vosk_model_path)
        if not vosk_model.exists():
            die(f"NG: vosk model missing: {vosk_model}")
    if transcription_backend == "ssh_remote":
        remote_cfg = cfg.get("remote", {})
        ssh_key = Path(str(remote_cfg.get("ssh_key", "")))
        if not ssh_key.exists():
            die(f"NG: ssh key missing: {ssh_key}")
    requested_wake_backend = str(wake_cfg.get("backend", "whisper")).strip().lower()
    google_enabled = (
        transcription_backend == "google" or requested_wake_backend == "google"
    )
    if google_enabled and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        die("NG: GOOGLE_APPLICATION_CREDENTIALS must be set for Google Speech-to-Text")
    if tts_enabled_value:
        try:
            r = requests.get(tts_cfg["engine_host"].rstrip("/") + "/version", timeout=2)
            r.raise_for_status()
        except Exception as exc:
            die(f"NG: voicevox engine not reachable: {exc}")
    if ai_correction_enabled or transcription_backend == "local":
        try:
            llm_healthcheck(llm_cfg)
        except Exception as exc:
            die(f"NG: llm provider not reachable: {exc}")


def main():
    cfg = load_cfg()
    ensure_runtime_services(cfg)
    sanity(cfg)

    workdir = Path(cfg["paths"]["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    runtime_state_path = workdir / "runtime_state.json"
    runtime_cfg = cfg.get("runtime", {})
    run_mode = str(runtime_cfg.get("run_mode", "assistant")).strip() or "assistant"
    recognition_mode_name, recognition_profiles, recognition_profile = (
        resolve_recognition_mode_bundle(cfg)
    )
    ai_correction_enabled = bool(
        recognition_profile.get("ai_correction", runtime_cfg.get("ai_correction", True))
    )
    transcription_backend = (
        str(
            recognition_profile.get(
                "backend", runtime_cfg.get("transcription_backend", "local")
            )
        ).strip()
        or "local"
    )
    command_execution_enabled = bool(runtime_cfg.get("command_execution", True))
    announce_startup = bool(runtime_cfg.get("announce_startup", True))
    storage_cfg = cfg.get("storage", {})
    jsonl_cfg = storage_cfg.get("jsonl", {})
    sqlite_cfg = storage_cfg.get("sqlite", {})
    event_jsonl_enabled = bool(jsonl_cfg.get("enabled", True))
    event_jsonl_path = workdir / str(jsonl_cfg.get("path", "events.jsonl"))
    event_sqlite_enabled = bool(sqlite_cfg.get("enabled", True))
    event_sqlite_path = workdir / str(sqlite_cfg.get("path", "voicechat.db"))
    if event_sqlite_enabled:
        init_event_db(event_sqlite_path)

    audio_in = cfg["audio"]["input"]
    audio_out = cfg["audio"].get("output", "default")
    sr = int(cfg["audio"]["sample_rate"])

    wake_cfg = cfg["wake"]
    wake_word = wake_cfg["word"]
    wake_words = [
        normalize_text(str(item))
        for item in wake_cfg.get("words", [])
        if normalize_text(str(item))
    ]
    if not wake_words:
        wake_words = [wake_word]
    wake_aliases = load_wake_aliases(wake_cfg, wake_words)
    if event_sqlite_enabled:
        wake_aliases = merge_wake_aliases(
            wake_aliases, load_recognition_aliases(event_sqlite_path, "wake")
        )
    window_sec = float(wake_cfg["window_sec"])
    requested_wake_backend = (
        str(wake_cfg.get("backend", "whisper")).strip() or "whisper"
    )
    wake_pre_vad_backend = (
        str(wake_cfg.get("pre_vad_backend", "webrtc")).strip() or "webrtc"
    )
    wake_pre_vad_hard_gate = bool(wake_cfg.get("pre_vad_hard_gate", False))
    startup_arm_delay_sec = max(0.0, float(wake_cfg.get("startup_arm_delay_sec", 0.3)))
    wake_inline_command_enabled = bool(wake_cfg.get("inline_command", True))
    porcupine_cfg = wake_cfg.get("porcupine", {})
    wake_backend, wake_backend_fallback_reason = resolve_wake_backend(
        requested_wake_backend, porcupine_cfg, wake_words
    )
    wake_words_porcupine = available_porcupine_wake_words(porcupine_cfg, wake_words)
    wake_words_whisper_fallback = [
        item for item in wake_words if item not in wake_words_porcupine
    ]

    vad_cfg = cfg.get("vad", {})
    vad_enabled = bool(vad_cfg.get("enabled", True))
    vad_aggr = int(vad_cfg.get("aggressiveness", 1))
    silence_ms = int(vad_cfg.get("silence_ms", 900))
    max_record_sec = int(vad_cfg.get("max_record_sec", 20))
    chunk_sec = float(vad_cfg.get("chunk_sec", 1.0))
    speech_ratio = float(vad_cfg.get("speech_ratio", 0.2))
    silero_cfg = vad_cfg.get("silero", {})
    audio_pipeline_cfg = cfg.get("audio_pipeline", {})
    rnnoise_cfg = audio_pipeline_cfg.get("rnnoise", {})

    whisper_cfg = cfg["whisper"]
    wbin = Path(
        os.environ.get("VOICECHAT_WHISPER_BIN", "").strip() or whisper_cfg["bin"]
    )
    realtime_wmodel = Path(
        str(whisper_cfg.get("realtime_model", "")).strip() or str(whisper_cfg["model"])
    )
    wmodel = Path(
        os.environ.get("VOICECHAT_WHISPER_MODEL", "").strip() or whisper_cfg["model"]
    )
    lang = whisper_cfg.get("lang", "ja")
    beam = int(whisper_cfg.get("beam", 5))
    best = int(whisper_cfg.get("best", 5))
    temp = float(whisper_cfg.get("temp", 0.0))
    threads = int(whisper_cfg.get("threads", 4))
    vosk_cfg = cfg.get("vosk", {})
    vosk_model = None
    if vosk_cfg:
        vosk_model_path = vosk_cfg.get("realtime_model") or vosk_cfg.get("model")
        if vosk_model_path:
            vosk_model = Path(vosk_model_path)
    wake_wmodel = Path(
        str(recognition_profile.get("wake_model", "")).strip()
        or str(wake_cfg.get("whisper_model", "")).strip()
        or str(wmodel)
    )
    wake_threads = int(
        recognition_profile.get(
            "wake_threads", wake_cfg.get("threads", max(1, min(threads, 2)))
        )
    )

    llm_cfg = resolve_llm_config(cfg)
    remote_cfg = cfg.get("remote", {})
    google_cfg = cfg.get("google_stt", {})
    speech_recognition_cfg = cfg.get("speech_recognition", {})
    ai_control_cfg = cfg.get("ai_control", {})
    ai_alias_map = build_ai_alias_map(ai_control_cfg)
    command_router_cfg = cfg.get("command_router", {})
    learned_command_aliases = (
        load_recognition_aliases(event_sqlite_path, "command")
        if event_sqlite_enabled
        else {}
    )
    command_phrase_index = build_command_phrase_index(
        command_router_cfg, learned_command_aliases
    )
    command_ready_prompt = normalize_text(
        str(command_router_cfg.get("ready_prompt", "はい、どうぞ。"))
    )
    unknown_command_reply = normalize_text(
        str(
            command_router_cfg.get(
                "unknown_command_reply",
                command_router_cfg.get("fallback_reply", "そのコマンドはないのだ。"),
            )
        )
    )
    short_input_reset_chars = max(
        0, int(command_router_cfg.get("short_input_reset_chars", 3))
    )
    short_input_reset_reply = normalize_text(
        str(
            command_router_cfg.get(
                "short_input_reset_reply",
                "短すぎて分からなかったのだ。もう一度ベリベリからお願いするのだ。",
            )
        )
    )
    no_speech_timeout_sec = max(
        0.0, float(command_router_cfg.get("no_speech_timeout_sec", 5))
    )
    no_speech_reset_reply = normalize_text(
        str(command_router_cfg.get("no_speech_reset_reply", ""))
    )
    timed_record_cfg = cfg.get("timed_record", {})

    assistant_cfg = cfg.get("assistant", {})
    active_mode = assistant_cfg.get("active_mode", "default")
    if run_mode in {"phrase_test", "debug_stt", "ai_duo"}:
        active_mode = run_mode
    elif run_mode == "always_on":
        active_mode = str(
            runtime_cfg.get(
                "assistant_mode", assistant_cfg.get("active_mode", "default")
            )
        )
    modes = assistant_cfg.get("modes", {})
    mode_cfg = modes.get(active_mode, {})
    system_prompt = mode_cfg.get(
        "system_prompt",
        assistant_cfg.get(
            "system_prompt",
            "あなたは日本語の音声アシスタント。会話量は状況に応じて調整する。自然な日本語で話す。",
        ),
    )
    startup_greeting = mode_cfg.get(
        "startup_greeting",
        assistant_cfg.get("startup_greeting", "起動しました。準備できています。"),
    )
    if mode_cfg.get("ollama_model"):
        llm_cfg = dict(llm_cfg)
        llm_cfg["model"] = mode_cfg["ollama_model"]
    startup_briefing_cfg = mode_cfg.get("startup_briefing", {})
    history_turns = int(
        mode_cfg.get("history_turns", assistant_cfg.get("history_turns", 6))
    )
    history_turns = max(0, history_turns)
    conversation_history: deque[dict[str, str]] = deque(
        maxlen=history_turns * 2 if history_turns else None
    )
    summary_cfg = mode_cfg.get(
        "summary_memory", assistant_cfg.get("summary_memory", {})
    )
    summary_enabled = bool(summary_cfg.get("enabled", True))
    summary_update_every = max(1, int(summary_cfg.get("update_every_turns", 3)))
    summary_text = normalize_text(summary_cfg.get("initial_summary", ""))
    summary_since_refresh = 0
    memory_search_cfg = mode_cfg.get(
        "memory_search", assistant_cfg.get("memory_search", {})
    )
    memory_search_enabled = bool(memory_search_cfg.get("enabled", True))
    memory_search_top_k = max(1, int(memory_search_cfg.get("top_k", 5)))
    memory_search_min_score = float(memory_search_cfg.get("min_score", 0.18))
    memory_search_scan_limit = max(50, int(memory_search_cfg.get("scan_limit", 2000)))
    memory_search_lookback_turns = max(
        0, int(memory_search_cfg.get("lookback_turns", 3))
    )
    rag_cfg = mode_cfg.get("rag", assistant_cfg.get("rag", {}))
    rag_enabled = bool(rag_cfg.get("enabled", True))
    rag_patterns = rag_cfg.get("paths", DEFAULT_RAG_GLOBS)
    rag_top_k = max(1, int(rag_cfg.get("top_k", 3)))
    rag_min_score = float(rag_cfg.get("min_score", 0.12))
    rag_chunk_size = max(200, int(rag_cfg.get("chunk_size", 500)))
    rag_chunk_overlap = max(0, int(rag_cfg.get("chunk_overlap", 80)))
    rag_max_files = max(1, int(rag_cfg.get("max_files", 24)))
    rag_max_file_chars = max(500, int(rag_cfg.get("max_file_chars", 4000)))
    log_cfg = mode_cfg.get(
        "conversation_log", assistant_cfg.get("conversation_log", {})
    )
    log_enabled = bool(log_cfg.get("enabled", True))
    log_path = workdir / log_cfg.get("path", "conversation.jsonl")
    debug_cfg = mode_cfg.get("debug_stt", {})
    debug_min_chars = max(1, int(debug_cfg.get("min_chars", 4)))
    debug_echo_cooldown_sec = float(debug_cfg.get("echo_cooldown_sec", 4.0))
    debug_echo_similarity = float(debug_cfg.get("echo_similarity", 0.72))
    phrase_test_cfg = mode_cfg.get("phrase_test", {})
    phrase_test_items = load_phrase_test_items(phrase_test_cfg)
    phrase_style_cycle = load_style_cycle(phrase_test_cfg.get("style_cycle", []))
    phrase_prompt_template = phrase_test_cfg.get(
        "prompt_template", "「{phrase}」と言ってください。"
    )
    phrase_repeat_prompt = phrase_test_cfg.get(
        "repeat_prompt", "もう一度同じ文をお願いします。"
    )
    phrase_index = 0
    last_spoken_text = ""
    last_spoken_at = 0.0
    rag_corpus = (
        load_rag_corpus(
            ROOT,
            rag_patterns,
            rag_chunk_size,
            rag_chunk_overlap,
            rag_max_files,
            rag_max_file_chars,
        )
        if rag_enabled
        else []
    )

    tts_cfg = cfg["tts"]
    tts_enabled_env = os.environ.get("VOICECHAT_TTS_ENABLED", "").strip()
    tts_enabled = (
        bool(tts_cfg.get("enabled", True))
        if not tts_enabled_env
        else tts_enabled_env.lower() in {"1", "true", "yes", "on"}
    )
    vv_host = tts_cfg.get("engine_host", "http://127.0.0.1:50021")
    vv_speaker = int(tts_cfg.get("speaker", 3))
    vv_speed = float(tts_cfg.get("voicevox_speed_scale", 1.0))
    vv_pitch = float(tts_cfg.get("voicevox_pitch_scale", 0.0))
    vv_intonation = float(tts_cfg.get("voicevox_intonation_scale", 1.0))
    vv_volume = float(tts_cfg.get("voicevox_volume_scale", 1.0))
    effects_cfg = cfg.get("effects", {})
    ready_effect_cfg = effects_cfg.get("ready", {})
    recorded_effect_cfg = effects_cfg.get("recorded", {})
    volume_change_effect_cfg = (
        effects_cfg.get("commands", {})
        if isinstance(effects_cfg.get("commands", {}), dict)
        else {}
    ).get("volume_change", {})
    command_effects_cfg = (
        effects_cfg.get("commands", {})
        if isinstance(effects_cfg.get("commands", {}), dict)
        else {}
    )

    def apply_recognition_mode(mode_name: str):
        nonlocal recognition_mode_name, recognition_profile
        nonlocal \
            ai_correction_enabled, \
            transcription_backend, \
            realtime_wmodel, \
            wmodel, \
            wake_wmodel, \
            wake_threads
        nonlocal runtime_state_base

        target_mode = normalize_text(mode_name) or recognition_mode_name
        recognition_profile = dict(
            recognition_profiles.get(target_mode, recognition_profile)
        )
        recognition_mode_name = target_mode
        ai_correction_enabled = bool(
            recognition_profile.get(
                "ai_correction", runtime_cfg.get("ai_correction", True)
            )
        )
        transcription_backend = (
            str(
                recognition_profile.get(
                    "backend", runtime_cfg.get("transcription_backend", "local")
                )
            ).strip()
            or "local"
        )
        realtime_wmodel = Path(
            str(recognition_profile.get("realtime_model", "")).strip()
            or str(whisper_cfg.get("realtime_model", "")).strip()
            or str(whisper_cfg["model"])
        )
        wmodel = Path(
            str(recognition_profile.get("final_model", "")).strip()
            or str(whisper_cfg["model"])
        )
        wake_wmodel = Path(
            str(recognition_profile.get("wake_model", "")).strip()
            or str(wake_cfg.get("whisper_model", "")).strip()
            or str(wmodel)
        )
        wake_threads = int(
            recognition_profile.get(
                "wake_threads", wake_cfg.get("threads", max(1, min(threads, 2)))
            )
        )
        runtime_state_base.update(
            {
                "recognition_mode": recognition_mode_name,
                "recognition_mode_label": normalize_text(
                    str(recognition_profile.get("label", recognition_mode_name))
                )
                or recognition_mode_name,
                "transcription_backend": transcription_backend,
                "whisper_model": str(wmodel),
                "whisper_realtime_model": str(realtime_wmodel),
                "wake_model": str(wake_wmodel),
            }
        )

    def speak(text: str, out_name: str, speaker_override: int | None = None):
        nonlocal last_spoken_at, last_spoken_text
        out_wav = workdir / out_name
        if not tts_enabled or not text:
            return
        last_spoken_text = normalize_text(text)
        last_spoken_at = time.time()
        speak_with_voicevox(
            text,
            audio_out,
            out_wav,
            vv_host,
            speaker_override if speaker_override is not None else vv_speaker,
            vv_speed,
            vv_pitch,
            vv_intonation,
            vv_volume,
        )

    def recent_self_echo_similarity(*texts: str) -> float:
        if not last_spoken_text:
            return 0.0
        elapsed = time.time() - last_spoken_at
        if elapsed > debug_echo_cooldown_sec:
            return 0.0
        best = 0.0
        for text in texts:
            normalized = normalize_text(text)
            if not normalized:
                continue
            best = max(best, score_chunk(normalized, last_spoken_text))
        return best

    def play_effect(name: str, effect_cfg: dict):
        if not effect_cfg.get("enabled", False):
            return
        file_value = normalize_text(str(effect_cfg.get("file", "")))
        if file_value:
            out_wav = Path(file_value)
            if not out_wav.is_absolute():
                out_wav = ROOT / out_wav
            if out_wav.exists():
                play_wav_file(out_wav, audio_out)
                return
        out_wav = workdir / f"effect_{name}.wav"
        synth_tone_wav(
            out_wav,
            sample_rate=sr,
            duration_ms=int(effect_cfg.get("duration_ms", 90)),
            frequency_hz=float(effect_cfg.get("frequency_hz", 880)),
            volume=float(effect_cfg.get("volume", 0.25)),
            gap_ms=int(effect_cfg.get("gap_ms", 0)),
            sequence=effect_cfg.get("sequence"),
        )
        play_wav_file(out_wav, audio_out)

    vad = webrtcvad.Vad(vad_aggr)

    print("========================================================================")
    print("VOICE CHAT")
    print("========================================================================")
    print(f"INFO: run_mode={run_mode}")
    print(f"INFO: wake_word={wake_word}")
    print(f"INFO: wake_words={wake_words}")
    print(f"INFO: wake_model={wake_wmodel}")
    print(f"INFO: wake_backend={wake_backend}")
    if wake_backend_fallback_reason:
        level = "WARN"
        if wake_backend == "porcupine":
            level = "INFO"
        print(
            f"{level}: requested wake backend '{requested_wake_backend}': {wake_backend_fallback_reason}"
        )
    print(f"INFO: wake_pre_vad_backend={wake_pre_vad_backend}")
    print(f"INFO: wake_inline_command={'on' if wake_inline_command_enabled else 'off'}")
    print(
        "INFO: recognition_mode="
        f"{recognition_mode_name} ({normalize_text(str(recognition_profile.get('label', recognition_mode_name))) or recognition_mode_name})"
    )
    print(f"INFO: stt_backend={transcription_backend}")
    print(f"INFO: whisper_model={wmodel}")
    print(f"INFO: whisper_realtime_model={realtime_wmodel}")
    if transcription_backend == "vosk" and vosk_model:
        print(f"INFO: vosk_model={vosk_model}")
    print(f"INFO: llm_provider={llm_cfg['provider']} model={llm_cfg['model']}")
    print(f"INFO: tts={'voicevox' if tts_enabled else 'disabled'}")
    print(f"INFO: mode={active_mode}")
    print(f"INFO: ai_correction={'on' if ai_correction_enabled else 'off'}")
    print(f"INFO: summary_memory={'on' if summary_enabled else 'off'}")
    print(f"INFO: memory_search={'on' if memory_search_enabled else 'off'}")
    print(f"INFO: rag={'on' if rag_enabled else 'off'} chunks={len(rag_corpus)}")
    print("INFO: 監視開始")
    runtime_state_base = {
        "pid": os.getpid(),
        "run_mode": run_mode,
        "active_mode": active_mode,
        "recognition_mode": recognition_mode_name,
        "recognition_mode_label": normalize_text(
            str(recognition_profile.get("label", recognition_mode_name))
        )
        or recognition_mode_name,
        "transcription_backend": transcription_backend,
        "wake_word": wake_word,
        "wake_words": wake_words,
        "whisper_model": str(wmodel),
        "whisper_realtime_model": str(realtime_wmodel),
        "wake_model": str(wake_wmodel),
        "vosk_model": str(vosk_model) if vosk_model else None,
        "requested_wake_backend": requested_wake_backend,
        "wake_backend": wake_backend,
        "wake_pre_vad_backend": wake_pre_vad_backend,
        "wake_inline_command": wake_inline_command_enabled,
        "llm_provider": llm_cfg["provider"],
        "llm_model": llm_cfg["model"],
        "tts_enabled": tts_enabled,
    }
    apply_recognition_mode(recognition_mode_name)
    write_runtime_state(runtime_state_path, {**runtime_state_base, "state": "startup"})
    if tts_enabled and startup_greeting and announce_startup:
        speak(startup_greeting, "reply_startup.wav")
    if tts_enabled and startup_briefing_cfg.get("enabled", False):
        briefing = build_briefing(startup_briefing_cfg.get("topic_source", {}))
        if briefing:
            print(f"INFO: briefing={briefing}")
            speak(briefing, "reply_briefing.wav")
    if active_mode not in {"debug_stt", "phrase_test", "ai_duo"}:
        play_effect("ready", ready_effect_cfg)
        if startup_arm_delay_sec > 0:
            time.sleep(startup_arm_delay_sec)

    if active_mode == "ai_duo":
        run_duo_mode(
            mode_cfg=mode_cfg,
            speak=speak,
            llm_cfg=llm_cfg,
        )
        return

    if run_mode == "timed_record":
        seconds = int(timed_record_cfg.get("seconds", 300))
        announce_start = normalize_text(
            str(timed_record_cfg.get("announce_start", "スタート"))
        )
        announce_end = normalize_text(
            render_timed_record_end_text(
                str(timed_record_cfg.get("announce_end_template", "{minutes}分終了")),
                seconds,
            )
        )
        tag = time.strftime("%Y%m%d_%H%M%S")
        wav_path = workdir / f"timed_record_{tag}.wav"
        if tts_enabled and announce_start:
            speak(announce_start, f"timed_record_{tag}_start.wav")
        rec_cmd = [
            "arecord",
            "-D",
            audio_in,
            "-f",
            "S16_LE",
            "-r",
            str(sr),
            "-c",
            "1",
            "-d",
            str(seconds),
            str(wav_path),
        ]
        if run(rec_cmd).returncode != 0:
            die("NG: timed record failed")
        if tts_enabled and announce_end:
            speak(announce_end, f"timed_record_{tag}_end.wav")

        raw_text, corrected_text, elapsed_sec, model_name = transcribe_audio(
            backend=transcription_backend,
            whisper_bin=wbin,
            whisper_model=wmodel,
            vosk_model=vosk_model,
            wav=wav_path,
            lang=lang,
            threads=threads,
            beam=beam,
            best=best,
            temp=temp,
            workdir=workdir,
            remote_cfg=remote_cfg,
            llm_cfg=llm_cfg,
            google_cfg=google_cfg,
            speech_recognition_cfg=speech_recognition_cfg,
        )
        final_text = corrected_text if ai_correction_enabled else raw_text
        if transcription_backend == "local" and ai_correction_enabled:
            final_text = correct_transcript(llm_cfg, raw_text)

        raw_path = workdir / f"timed_record_{tag}_raw.txt"
        corrected_path = workdir / f"timed_record_{tag}_corrected.txt"
        meta_path = workdir / f"timed_record_{tag}.json"
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        corrected_path.write_text(final_text + "\n", encoding="utf-8")
        payload = {
            "ts": int(time.time()),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "timed_record",
            "mode": run_mode,
            "backend": transcription_backend,
            "model": model_name,
            "wav": str(wav_path),
            "raw_path": str(raw_path),
            "corrected_path": str(corrected_path),
            "recognized": final_text,
            "raw_user": raw_text,
            "corrected_user": final_text,
            "elapsed_sec": elapsed_sec,
            "seconds": seconds,
        }
        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        append_event_logs(
            payload=payload,
            event_type="timed_record",
            jsonl_enabled=event_jsonl_enabled,
            jsonl_path=event_jsonl_path,
            sqlite_enabled=event_sqlite_enabled,
            sqlite_path=event_sqlite_path,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    command_retry_active = False
    pending_command_confirmation: dict[str, Any] | None = None

    while True:
        inline_command_text = ""
        wake_text = ""
        wake_matched = False
        matched_wake_word = ""
        wake_payload: dict[str, Any] | None = None
        if active_mode not in {"debug_stt", "phrase_test"}:
            if not command_retry_active:
                print("INFO: ウェイクワード待機中...")
                write_runtime_state(
                    runtime_state_path, {**runtime_state_base, "state": "wake_wait"}
                )
                pcm = arecord_chunk_pcm(audio_in, window_sec, sr)
                win_wav = workdir / "wake_window.wav"
                wav_write_pcm16_mono(win_wav, [pcm], sr)
                wake_prepared_wav = maybe_apply_rnnoise(
                    win_wav, workdir, rnnoise_cfg, tag="wake_window"
                )
                wake_started_at = time.time()
                wake_meta: dict[str, Any] = {}

                if wake_pre_vad_backend == "silero":
                    has_speech, silero_meta = detect_silero_speech(
                        wake_prepared_wav, silero_cfg
                    )
                    wake_meta["silero"] = silero_meta
                    if not has_speech and wake_pre_vad_hard_gate:
                        wake_elapsed_sec = round(time.time() - wake_started_at, 3)
                        wake_payload = {
                            "ts": int(time.time()),
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "wake_check",
                            "mode": active_mode,
                            "backend": "silero_precheck",
                            "model": str(wake_wmodel),
                            "recognized": "",
                            "expected": wake_word,
                            "wake_words": wake_words,
                            "elapsed_sec": wake_elapsed_sec,
                            "wake_word": wake_word,
                            "matched_wake_word": "",
                            "wake_window_sec": window_sec,
                            "wake_threads": wake_threads,
                            "matched": False,
                            "speech_detected": False,
                            "wake_backend": wake_backend,
                            "pre_vad_backend": wake_pre_vad_backend,
                            "audio_path": str(wake_prepared_wav),
                            **wake_meta,
                        }
                        append_event_logs(
                            payload=wake_payload,
                            event_type="wake_check",
                            jsonl_enabled=event_jsonl_enabled,
                            jsonl_path=event_jsonl_path,
                            sqlite_enabled=event_sqlite_enabled,
                            sqlite_path=event_sqlite_path,
                        )
                        continue

                if wake_backend == "porcupine":
                    wake_matched, matched_wake_word, porcupine_meta = (
                        detect_porcupine_wake(
                            wake_prepared_wav,
                            porcupine_cfg,
                            wake_words_porcupine or wake_words,
                        )
                    )
                    wake_meta["porcupine"] = porcupine_meta
                    wake_text = matched_wake_word if wake_matched else ""
                    if not wake_matched and wake_words_whisper_fallback:
                        wake_text = whisper_transcribe_txt(
                            wbin,
                            wake_wmodel,
                            wake_prepared_wav,
                            lang,
                            wake_threads,
                            beam,
                            best,
                            temp,
                        )
                        if wake_text:
                            print(f"WAKECHK-FALLBACK: {wake_text}")
                        wake_matched, matched_wake_word = contains_any_wake(
                            wake_text,
                            wake_words_whisper_fallback,
                            wake_cfg,
                            wake_aliases,
                        )
                        wake_meta["whisper_fallback"] = {
                            "wake_words": wake_words_whisper_fallback,
                            "used": True,
                        }
                elif wake_backend == "google":
                    try:
                        wake_text, _ = _google_transcribe(wake_prepared_wav, google_cfg)
                        if wake_text:
                            print(f"WAKECHK-GOOGLE: {wake_text}")
                    except Exception as exc:
                        print(f"WARN: Google wake transcription failed: {exc}")
                        wake_text = ""
                    wake_matched, matched_wake_word = contains_any_wake(
                        wake_text,
                        wake_words,
                        wake_cfg,
                        wake_aliases,
                    )
                elif wake_backend == "vosk":
                    raw_text, elapsed, model_name = transcribe_vosk(
                        vosk_model=vosk_model,
                        wav=wake_prepared_wav,
                        boost_volume=10.0,
                    )
                    wake_text = raw_text
                    if wake_text:
                        print(f"WAKECHK-VOSK: {wake_text}")
                    wake_matched, matched_wake_word = contains_any_wake(
                        wake_text,
                        wake_words,
                        wake_cfg,
                        wake_aliases,
                    )
                    print(f"DEBUG: wake_matched={wake_matched}, wake_text={wake_text}")
                elif wake_backend == "speech_recognition":
                    wake_text, _, model_name = transcribe_speech_recognition(
                        wav=wake_prepared_wav,
                        speech_recognition_cfg=speech_recognition_cfg,
                        boost_volume=10.0,
                    )
                    if wake_text.startswith("NG:"):
                        print(f"WARN: wake speech_recognition failed: {wake_text}")
                    elif wake_text:
                        print(f"WAKECHK-SR: {wake_text}")
                    wake_matched, matched_wake_word = contains_any_wake(
                        wake_text,
                        wake_words,
                        wake_cfg,
                        wake_aliases,
                    )
                else:
                    wake_text = whisper_transcribe_txt(
                        wbin,
                        wake_wmodel,
                        wake_prepared_wav,
                        lang,
                        wake_threads,
                        beam,
                        best,
                        temp,
                    )
                    if wake_text:
                        print(f"WAKECHK: {wake_text}")
                    wake_matched, matched_wake_word = contains_any_wake(
                        wake_text, wake_words, wake_cfg, wake_aliases
                    )
                wake_elapsed_sec = round(time.time() - wake_started_at, 3)
                wake_payload = {
                    "ts": int(time.time()),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "wake_check",
                    "mode": active_mode,
                    "backend": f"wake_{wake_backend}",
                    "model": str(wake_wmodel),
                    "recognized": wake_text,
                    "expected": wake_word,
                    "wake_words": wake_words,
                    "elapsed_sec": wake_elapsed_sec,
                    "wake_word": wake_word,
                    "matched_wake_word": matched_wake_word,
                    "wake_window_sec": window_sec,
                    "wake_threads": wake_threads,
                    "matched": wake_matched,
                    "speech_detected": True,
                    "wake_backend": wake_backend,
                    "pre_vad_backend": wake_pre_vad_backend,
                    "audio_path": str(wake_prepared_wav),
                    **wake_meta,
                }
            if wake_payload is not None:
                append_event_logs(
                    payload=wake_payload,
                    event_type="wake_check",
                    jsonl_enabled=event_jsonl_enabled,
                    jsonl_path=event_jsonl_path,
                    sqlite_enabled=event_sqlite_enabled,
                    sqlite_path=event_sqlite_path,
                )

                if not wake_matched:
                    continue

                music_restore_volume: int | None = None
                if is_music_playing():
                    print("INFO: music playback detected; ducking volume")
                    music_restore_volume = duck_music_volume(10)

                if wake_inline_command_enabled:
                    inline_command_text, _ = extract_inline_command_from_wake(
                        wake_text,
                        wake_words,
                        wake_aliases,
                    )
                    if inline_command_text:
                        print(f"INFO: inline command detected: {inline_command_text}")
                        write_runtime_state(
                            runtime_state_path,
                            {
                                **runtime_state_base,
                                "state": "command_inline",
                                "recognized_fast": inline_command_text,
                                "recognized_final": inline_command_text,
                                "wake_recognized": wake_text,
                            },
                        )

                if inline_command_text:
                    print("INFO: wake transcript will be used as command input")
                else:
                    print("INFO: コマンド再入力待機中...")
                    play_effect("ready", ready_effect_cfg)
                    write_runtime_state(
                        runtime_state_path,
                        {**runtime_state_base, "state": "command_retry_wait"},
                    )
                command_retry_active = False
        else:
            if active_mode == "phrase_test":
                if not phrase_test_items:
                    raise RuntimeError("phrase_test mode requires phrase_test.phrases")
                current_phrase = phrase_test_items[
                    phrase_index % len(phrase_test_items)
                ]
                current_style = (
                    phrase_style_cycle[phrase_index % len(phrase_style_cycle)]
                    if phrase_style_cycle
                    else {"speaker": vv_speaker, "name": "default"}
                )
                expected_phrase = current_phrase["text"]
                print(
                    f"INFO: phrase_test prompt={current_phrase['id']} {expected_phrase} "
                    f"speaker={current_style['speaker']} style={current_style['name']}"
                )
                if tts_enabled:
                    speak(
                        phrase_prompt_template.format(phrase=expected_phrase),
                        f"reply_phrase_prompt_{phrase_index % len(phrase_test_items)}.wav",
                        int(current_style["speaker"]),
                    )
                phrase_index += 1
            else:
                print("INFO: 連続音声認識待機中...")
                write_runtime_state(
                    runtime_state_path,
                    {**runtime_state_base, "state": "continuous_listen"},
                )

        if inline_command_text:
            heard_speech = True
            user_text = inline_command_text
            prefetched_corrected_text = inline_command_text
            transcribe_elapsed_sec = wake_elapsed_sec
            transcribe_model = f"wake_{wake_backend}"
            final_user_text = inline_command_text
            final_prefetched_corrected_text = inline_command_text
            final_transcribe_elapsed_sec = wake_elapsed_sec
            final_transcribe_model = f"wake_{wake_backend}"
        else:
            print("INFO: 録音開始")
            write_runtime_state(
                runtime_state_path,
                {
                    **runtime_state_base,
                    "state": "recording",
                    "command_retry_active": command_retry_active,
                },
            )
            all_pcm: list[bytes] = []
            silent_chunks = 0
            heard_speech = False
            no_speech_timeout = False
            need_silent_chunks = max(1, round(silence_ms / max(1.0, chunk_sec * 1000)))
            started = time.time()

            while True:
                pcm = arecord_chunk_pcm(audio_in, chunk_sec, sr)
                all_pcm.append(pcm)

                if vad_enabled:
                    speech_frames, total_frames = is_speech_frame(
                        vad, pcm, sr, frame_ms=30
                    )
                    speaking = (
                        total_frames > 0
                        and (speech_frames / total_frames) >= speech_ratio
                    )
                    if speaking:
                        heard_speech = True
                        silent_chunks = 0
                        print("VAD: speech")
                    elif heard_speech:
                        silent_chunks += 1
                        print(f"VAD: silence ({silent_chunks}/{need_silent_chunks})")
                    else:
                        print("VAD: waiting speech")

                    if heard_speech and silent_chunks >= need_silent_chunks:
                        print("INFO: end-of-speech detected")
                        break

                if (
                    active_mode not in {"debug_stt", "phrase_test"}
                    and not heard_speech
                    and no_speech_timeout_sec > 0
                    and (time.time() - started) >= no_speech_timeout_sec
                ):
                    print(f"INFO: no speech timeout reached ({no_speech_timeout_sec}s)")
                    no_speech_timeout = True
                    break

                if (time.time() - started) >= max_record_sec:
                    print("INFO: max_record_sec reached")
                    break

            if no_speech_timeout:
                restore_music_volume(music_restore_volume)
                append_event_logs(
                    payload={
                        "ts": int(time.time()),
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "command_reset_no_speech",
                        "mode": active_mode,
                        "backend": transcription_backend,
                        "elapsed_sec": round(time.time() - started, 3),
                        "no_speech_timeout_sec": no_speech_timeout_sec,
                    },
                    event_type="command_reset_no_speech",
                    jsonl_enabled=event_jsonl_enabled,
                    jsonl_path=event_jsonl_path,
                    sqlite_enabled=event_sqlite_enabled,
                    sqlite_path=event_sqlite_path,
                )
                play_effect("recorded", recorded_effect_cfg)
                pending_command_confirmation = None
                command_retry_active = False
                write_runtime_state(
                    runtime_state_path,
                    {
                        **runtime_state_base,
                        "state": "wake_wait",
                        "last_reset": "no_speech",
                    },
                )
                continue

            rec_wav = workdir / f"rec_final_{int(time.time())}.wav"
            wav_write_pcm16_mono(rec_wav, all_pcm, sr)
            if active_mode not in {"debug_stt", "phrase_test"}:
                try:
                    reset_as_no_speech, silero_gate_meta = (
                        should_reset_command_as_no_speech(
                            rec_wav,
                            command_router_cfg=command_router_cfg,
                            silero_cfg=silero_cfg,
                        )
                    )
                except Exception as exc:
                    print(f"WARN: command Silero gate skipped: {exc}")
                    reset_as_no_speech = False
                    silero_gate_meta = {
                        "enabled": True,
                        "error": normalize_text(str(exc)),
                    }
                if reset_as_no_speech:
                    print("INFO: command reset by Silero gate (no speech)")
                    restore_music_volume(music_restore_volume)
                    append_event_logs(
                        payload={
                            "ts": int(time.time()),
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "command_reset_no_speech",
                            "mode": active_mode,
                            "backend": transcription_backend,
                            "elapsed_sec": round(time.time() - started, 3),
                            "no_speech_timeout_sec": no_speech_timeout_sec,
                            "detector": "silero_gate",
                            "silero": silero_gate_meta,
                        },
                        event_type="command_reset_no_speech",
                        jsonl_enabled=event_jsonl_enabled,
                        jsonl_path=event_jsonl_path,
                        sqlite_enabled=event_sqlite_enabled,
                        sqlite_path=event_sqlite_path,
                    )
                    play_effect("recorded", recorded_effect_cfg)
                    pending_command_confirmation = None
                    command_retry_active = False
                    write_runtime_state(
                        runtime_state_path,
                        {
                            **runtime_state_base,
                            "state": "wake_wait",
                            "last_reset": "silero_no_speech",
                            "silero": silero_gate_meta,
                        },
                    )
                    continue
                play_effect("recorded", recorded_effect_cfg)

            if active_mode in {"debug_stt", "phrase_test"} and not heard_speech:
                print(f"INFO: {active_mode} skip (no speech detected)")
                continue

            user_text = ""
            prefetched_corrected_text = ""
            transcribe_elapsed_sec = 0.0
            transcribe_model = str(wmodel)
            final_user_text = ""
            final_prefetched_corrected_text = ""
            final_transcribe_elapsed_sec = 0.0
            final_transcribe_model = str(wmodel)

            if (
                active_mode not in {"debug_stt", "phrase_test"}
                and transcription_backend == "local"
            ):
                write_runtime_state(
                    runtime_state_path,
                    {**runtime_state_base, "state": "transcribing_realtime"},
                )
                (
                    user_text,
                    prefetched_corrected_text,
                    transcribe_elapsed_sec,
                    transcribe_model,
                ) = transcribe_audio(
                    backend=transcription_backend,
                    whisper_bin=wbin,
                    whisper_model=realtime_wmodel,
                    vosk_model=vosk_model,
                    wav=rec_wav,
                    lang=lang,
                    threads=threads,
                    beam=beam,
                    best=best,
                    temp=temp,
                    workdir=workdir,
                    remote_cfg=remote_cfg,
                    llm_cfg=llm_cfg,
                    google_cfg=google_cfg,
                    speech_recognition_cfg=speech_recognition_cfg,
                    rnnoise_cfg=rnnoise_cfg,
                    preprocess_tag="command_realtime",
                )
                write_runtime_state(
                    runtime_state_path,
                    {**runtime_state_base, "state": "transcribing_final"},
                )
                (
                    final_user_text,
                    final_prefetched_corrected_text,
                    final_transcribe_elapsed_sec,
                    final_transcribe_model,
                ) = transcribe_audio(
                    backend=transcription_backend,
                    whisper_bin=wbin,
                    whisper_model=wmodel,
                    vosk_model=vosk_model,
                    wav=rec_wav,
                    lang=lang,
                    threads=threads,
                    beam=beam,
                    best=best,
                    temp=temp,
                    workdir=workdir,
                    remote_cfg=remote_cfg,
                    llm_cfg=llm_cfg,
                    google_cfg=google_cfg,
                    speech_recognition_cfg=speech_recognition_cfg,
                    rnnoise_cfg=rnnoise_cfg,
                    preprocess_tag="command_final",
                )
            else:
                write_runtime_state(
                    runtime_state_path, {**runtime_state_base, "state": "transcribing"}
                )
                (
                    user_text,
                    prefetched_corrected_text,
                    transcribe_elapsed_sec,
                    transcribe_model,
                ) = transcribe_audio(
                    backend=transcription_backend,
                    whisper_bin=wbin,
                    whisper_model=wmodel,
                    vosk_model=vosk_model,
                    wav=rec_wav,
                    lang=lang,
                    threads=threads,
                    beam=beam,
                    best=best,
                    temp=temp,
                    workdir=workdir,
                    remote_cfg=remote_cfg,
                    llm_cfg=llm_cfg,
                    google_cfg=google_cfg,
                    speech_recognition_cfg=speech_recognition_cfg,
                    rnnoise_cfg=rnnoise_cfg,
                    preprocess_tag="command_single",
                )
                final_user_text = user_text
                final_prefetched_corrected_text = prefetched_corrected_text
                final_transcribe_elapsed_sec = transcribe_elapsed_sec
                final_transcribe_model = transcribe_model

        user_text = normalize_transcript_text(user_text)
        prefetched_corrected_text = normalize_transcript_text(prefetched_corrected_text)
        final_user_text = normalize_transcript_text(final_user_text)
        final_prefetched_corrected_text = normalize_transcript_text(
            final_prefetched_corrected_text
        )
        print("===== USER =====")
        print(user_text if user_text else "(no transcript)")
        print("================")
        write_runtime_state(
            runtime_state_path,
            {
                **runtime_state_base,
                "state": "command_mode",
                "recognized_fast": user_text,
                "recognized_final": final_user_text or user_text,
            },
        )
        if final_user_text and compact_text(final_user_text) != compact_text(user_text):
            print("===== USER FINAL =====")
            print(final_user_text)
            print("======================")

        if not user_text:
            restore_music_volume(music_restore_volume)
            if active_mode == "phrase_test" and tts_enabled:
                speak(phrase_repeat_prompt, "reply_phrase_retry.wav")
            elif tts_enabled and active_mode != "debug_stt":
                speak("聞き取れませんでした。もう一度お願いします。", "reply_retry.wav")
            if active_mode not in {"debug_stt", "phrase_test"}:
                command_retry_active = True
                write_runtime_state(
                    runtime_state_path,
                    {
                        **runtime_state_base,
                        "state": "command_retry_wait",
                        "last_reset": "empty_transcript",
                    },
                )
            continue

        if active_mode not in {"debug_stt", "phrase_test"}:
            self_echo_similarity = recent_self_echo_similarity(
                user_text, final_user_text or user_text
            )
            if self_echo_similarity >= debug_echo_similarity:
                print(
                    "INFO: command input skipped as self echo "
                    f"(similarity={self_echo_similarity:.2f})"
                )
                restore_music_volume(music_restore_volume)
                append_event_logs(
                    payload={
                        "ts": int(time.time()),
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "command_skip_self_echo",
                        "mode": active_mode,
                        "backend": transcription_backend,
                        "model": final_transcribe_model,
                        "recognized": user_text,
                        "raw_user": final_user_text or user_text,
                        "recognized_fast": user_text,
                        "recognized_final": final_user_text or user_text,
                        "last_spoken_text": last_spoken_text,
                        "self_echo_similarity": round(self_echo_similarity, 4),
                        "elapsed_sec": final_transcribe_elapsed_sec
                        or transcribe_elapsed_sec,
                    },
                    event_type="command_skip_self_echo",
                    jsonl_enabled=event_jsonl_enabled,
                    jsonl_path=event_jsonl_path,
                    sqlite_enabled=event_sqlite_enabled,
                    sqlite_path=event_sqlite_path,
                )
                pending_command_confirmation = None
                command_retry_active = False
                write_runtime_state(
                    runtime_state_path,
                    {
                        **runtime_state_base,
                        "state": "wake_wait",
                        "last_reset": "self_echo",
                        "recognized_fast": user_text,
                        "recognized_final": final_user_text or user_text,
                    },
                )
                continue

        if active_mode in {"debug_stt", "phrase_test"}:
            if len(compact_text(user_text)) < debug_min_chars:
                print(f"INFO: {active_mode} skip (too short)")
                restore_music_volume(music_restore_volume)
                continue

            if (
                last_spoken_text
                and (time.time() - last_spoken_at) <= debug_echo_cooldown_sec
            ):
                similarity = score_chunk(user_text, last_spoken_text)
                if similarity >= debug_echo_similarity:
                    print(
                        f"INFO: {active_mode} skip (echo similarity={similarity:.2f})"
                    )
                    restore_music_volume(music_restore_volume)
                    continue

            corrected_text = user_text
            if ai_correction_enabled:
                if transcription_backend == "ssh_remote":
                    corrected_text = prefetched_corrected_text or user_text
                else:
                    corrected_text = correct_transcript(llm_cfg, user_text)
            print("===== CORRECTED =====")
            print(corrected_text if corrected_text else "(empty corrected text)")
            print("=====================")

            if len(compact_text(corrected_text)) < debug_min_chars:
                print(f"INFO: {active_mode} skip (corrected too short)")
                restore_music_volume(music_restore_volume)
                continue

            if (
                last_spoken_text
                and (time.time() - last_spoken_at) <= debug_echo_cooldown_sec
            ):
                similarity = score_chunk(corrected_text, last_spoken_text)
                if similarity >= debug_echo_similarity:
                    print(
                        f"INFO: {active_mode} skip (corrected echo similarity={similarity:.2f})"
                    )
                    restore_music_volume(music_restore_volume)
                    continue

            log_payload = {
                "ts": int(time.time()),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": active_mode,
                "backend": transcription_backend,
                "model": transcribe_model,
                "recognized": corrected_text,
                "elapsed_sec": transcribe_elapsed_sec,
                "raw_user": user_text,
                "corrected_user": corrected_text,
            }

            if active_mode == "phrase_test":
                current_phrase = phrase_test_items[
                    (phrase_index - 1) % len(phrase_test_items)
                ]
                current_style = (
                    phrase_style_cycle[(phrase_index - 1) % len(phrase_style_cycle)]
                    if phrase_style_cycle
                    else {"speaker": vv_speaker, "name": "default"}
                )
                expected_phrase = current_phrase["text"]
                phrase_result = {
                    "phrase_id": current_phrase["id"],
                    "phrase_note": current_phrase.get("note", ""),
                    "prompt_speaker": int(current_style["speaker"]),
                    "prompt_style": current_style["name"],
                    "expected": expected_phrase,
                    "similarity": round(
                        score_chunk(expected_phrase, corrected_text), 4
                    ),
                    "ollama_judgement": judge_phrase_result(
                        llm_cfg, expected_phrase, corrected_text
                    ),
                }
                log_payload.update(phrase_result)
                print("===== PHRASE TEST =====")
                print(log_payload)
                print("=======================")

            if log_enabled:
                append_jsonl(log_path, log_payload)
            append_event_logs(
                payload=log_payload,
                event_type=active_mode,
                jsonl_enabled=event_jsonl_enabled,
                jsonl_path=event_jsonl_path,
                sqlite_enabled=event_sqlite_enabled,
                sqlite_path=event_sqlite_path,
            )

            if corrected_text and tts_enabled:
                speak(corrected_text, "reply_corrected.wav")
            continue

        effective_user_text = user_text
        if ai_correction_enabled:
            if transcription_backend == "ssh_remote":
                effective_user_text = prefetched_corrected_text or user_text
            else:
                effective_user_text = correct_transcript(llm_cfg, user_text)
        final_effective_user_text = final_user_text or effective_user_text
        if ai_correction_enabled:
            if transcription_backend == "ssh_remote":
                final_effective_user_text = (
                    final_prefetched_corrected_text or final_effective_user_text
                )
            elif final_effective_user_text:
                final_effective_user_text = correct_transcript(
                    llm_cfg, final_effective_user_text
                )

        normalized_command_text, command_candidates = normalize_command_text(
            effective_user_text,
            command_phrase_index,
            command_router_cfg,
            llm_cfg,
        )
        command_input_text = normalized_command_text or effective_user_text

        command_hit = match_command(command_input_text, command_router_cfg)
        if pending_command_confirmation and not command_hit:
            if is_confirmation_yes(command_input_text, command_router_cfg):
                pending_item = pending_command_confirmation.get("item")
                if isinstance(pending_item, dict):
                    command_hit = pending_item
                    phrases = pending_item.get("phrases", [])
                    if isinstance(phrases, list) and phrases:
                        command_input_text = normalize_text(str(phrases[0]))
            elif is_confirmation_no(command_input_text, command_router_cfg):
                pending_command_confirmation = None
                restore_music_volume(music_restore_volume)
                if short_input_reset_reply and tts_enabled:
                    speak(short_input_reset_reply, "reply_command_retry.wav")
                    play_effect("ready", ready_effect_cfg)
                command_retry_active = True
                write_runtime_state(
                    runtime_state_path,
                    {
                        **runtime_state_base,
                        "state": "command_retry_wait",
                        "last_reset": "candidate_rejected",
                        "recognized_fast": user_text,
                        "recognized_final": final_user_text or user_text,
                    },
                )
                continue
            else:
                pending_command_confirmation = None
        if command_hit:
            confirmation_source_text = ""
            if (
                pending_command_confirmation
                and command_hit == pending_command_confirmation.get("item")
            ):
                confirmation_source_text = normalize_text(
                    str(pending_command_confirmation.get("source_text", ""))
                )
            pending_command_confirmation = None
            (
                internal_handled,
                updated_llm_cfg,
                internal_reply,
                updated_recognition_mode_name,
            ) = execute_internal_command(
                command_hit=command_hit,
                llm_cfg=llm_cfg,
                ai_alias_map=ai_alias_map,
                recognition_mode_name=recognition_mode_name,
                recognition_profiles=recognition_profiles,
            )
            llm_cfg = updated_llm_cfg
            if updated_recognition_mode_name != recognition_mode_name:
                apply_recognition_mode(updated_recognition_mode_name)
            if internal_handled:
                command_result = {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "internal": True,
                }
            else:
                command_result = execute_command_action(
                    command_hit, command_router_cfg, command_execution_enabled
                )
            reply_text = render_command_reply(
                str(command_hit.get("reply", "")), command_hit
            ) or normalize_text(str(command_router_cfg.get("fallback_reply", "")))
            if command_hit.get("id") == "help":
                reply_text = build_help_reply(command_router_cfg)
            if internal_reply:
                reply_text = internal_reply
            elif normalize_text(str(command_result.get("message", ""))):
                reply_text = normalize_text(str(command_result.get("message", "")))
            if command_hit.get("id") == "recent_raw_transcript":
                reply_text = (
                    fetch_recent_raw_transcript(event_sqlite_path)
                    or "まだ生文字起こしの記録がないのだ。"
                )
            elif command_hit.get("id") == "today_raw_transcript":
                today = time.strftime("%Y-%m-%d")
                reply_text = (
                    fetch_today_raw_transcripts(event_sqlite_path, today)
                    or "今日はまだ生文字起こしの記録が少ないのだ。"
                )
            command_payload = {
                "ts": int(time.time()),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "command",
                "mode": active_mode,
                "backend": transcription_backend,
                "model": final_transcribe_model,
                "recognized": command_input_text,
                "raw_user": final_user_text or user_text,
                "corrected_user": final_effective_user_text,
                "recognized_fast": user_text,
                "recognized_final": final_user_text or user_text,
                "model_fast": transcribe_model,
                "model_final": final_transcribe_model,
                "command_input_text": command_input_text,
                "command_candidates": command_candidates,
                "elapsed_sec": final_transcribe_elapsed_sec or transcribe_elapsed_sec,
                "command_id": command_hit.get("id"),
                "command_reply": reply_text,
                "command_ok": command_result.get("ok"),
                "command_returncode": command_result.get("returncode"),
                "command_stdout": normalize_text(str(command_result.get("stdout", ""))),
                "command_stderr": normalize_text(str(command_result.get("stderr", ""))),
                "command_skipped": bool(command_result.get("skipped", False)),
                "llm_provider_after": llm_cfg.get("provider"),
                "llm_model_after": llm_cfg.get("model"),
            }
            append_event_logs(
                payload=command_payload,
                event_type="command",
                jsonl_enabled=event_jsonl_enabled,
                jsonl_path=event_jsonl_path,
                sqlite_enabled=event_sqlite_enabled,
                sqlite_path=event_sqlite_path,
            )
            if (
                event_sqlite_enabled
                and effective_user_text
                and compact_text(effective_user_text).lower()
                != compact_text(command_input_text).lower()
            ):
                save_recognition_alias(
                    event_sqlite_path,
                    alias_type="command",
                    target=normalize_text(str(command_hit.get("id", ""))),
                    alias=effective_user_text,
                    source="command_success",
                )
            if event_sqlite_enabled and confirmation_source_text:
                save_recognition_alias(
                    event_sqlite_path,
                    alias_type="command",
                    target=normalize_text(str(command_hit.get("id", ""))),
                    alias=confirmation_source_text,
                    source="command_confirmed",
                )
            if reply_text and tts_enabled:
                speak(reply_text, "reply_command.wav")
            success_effect_name = normalize_text(
                str(command_hit.get("success_effect", ""))
            ) or "ready"
            success_effect_cfg = (
                command_effects_cfg.get(success_effect_name, {})
                if success_effect_name
                else {}
            )
            if success_effect_name and isinstance(success_effect_cfg, dict):
                play_effect(success_effect_name, success_effect_cfg)
            if command_hit.get("id") == "shutdown":
                restore_music_volume(music_restore_volume)
                print("INFO: shutdown command received")
                write_runtime_state(
                    runtime_state_path,
                    {
                        **runtime_state_base,
                        "state": "stopped",
                        "last_command": "shutdown",
                    },
                )
                return
            command_retry_active = False
            restore_music_volume(music_restore_volume)
            write_runtime_state(
                runtime_state_path,
                {
                    **runtime_state_base,
                    "state": "wake_wait",
                    "last_command": command_hit.get("id"),
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                },
            )
            continue

        if (
            short_input_reset_chars
            and len(compact_text(command_input_text)) <= short_input_reset_chars
        ):
            print(f"INFO: command reset (too short <= {short_input_reset_chars})")
            restore_music_volume(music_restore_volume)
            append_event_logs(
                payload={
                    "ts": int(time.time()),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "command_reset_short_input",
                    "mode": active_mode,
                    "backend": transcription_backend,
                    "model": final_transcribe_model,
                    "recognized": command_input_text,
                    "raw_user": final_user_text or user_text,
                    "corrected_user": final_effective_user_text,
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                    "model_fast": transcribe_model,
                    "model_final": final_transcribe_model,
                    "elapsed_sec": final_transcribe_elapsed_sec
                    or transcribe_elapsed_sec,
                    "short_input_reset_chars": short_input_reset_chars,
                },
                event_type="command_reset_short_input",
                jsonl_enabled=event_jsonl_enabled,
                jsonl_path=event_jsonl_path,
                sqlite_enabled=event_sqlite_enabled,
                sqlite_path=event_sqlite_path,
            )
            if short_input_reset_reply and tts_enabled:
                speak(short_input_reset_reply, "reply_command_reset.wav")
                play_effect("ready", ready_effect_cfg)
            command_retry_active = True
            write_runtime_state(
                runtime_state_path,
                {
                    **runtime_state_base,
                    "state": "command_retry_wait",
                    "last_reset": "short_input",
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                },
            )
            continue

        if is_command_reset_phrase(command_input_text, command_router_cfg):
            restore_music_volume(music_restore_volume)
            append_event_logs(
                payload={
                    "ts": int(time.time()),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "command_reset_to_wake",
                    "mode": active_mode,
                    "backend": transcription_backend,
                    "model": final_transcribe_model,
                    "recognized": command_input_text,
                    "raw_user": final_user_text or user_text,
                    "corrected_user": final_effective_user_text,
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                    "elapsed_sec": final_transcribe_elapsed_sec
                    or transcribe_elapsed_sec,
                },
                event_type="command_reset_to_wake",
                jsonl_enabled=event_jsonl_enabled,
                jsonl_path=event_jsonl_path,
                sqlite_enabled=event_sqlite_enabled,
                sqlite_path=event_sqlite_path,
            )
            pending_command_confirmation = None
            command_retry_active = False
            write_runtime_state(
                runtime_state_path,
                {
                    **runtime_state_base,
                    "state": "wake_wait",
                    "last_reset": "command_reset_phrase",
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                },
            )
            continue

        candidate_command_item, candidate_reply_text = (
            choose_command_confirmation_candidate(
                command_candidates, command_router_cfg
            )
        )
        if candidate_command_item and candidate_reply_text:
            pending_command_confirmation = {
                "item": candidate_command_item,
                "source_text": command_input_text,
            }
            append_event_logs(
                payload={
                    "ts": int(time.time()),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "command_candidate_confirmation",
                    "mode": active_mode,
                    "backend": transcription_backend,
                    "model": final_transcribe_model,
                    "recognized": command_input_text,
                    "raw_user": final_user_text or user_text,
                    "corrected_user": final_effective_user_text,
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                    "command_candidates": command_candidates,
                    "candidate_command_id": candidate_command_item.get("id"),
                    "command_reply": candidate_reply_text,
                    "elapsed_sec": final_transcribe_elapsed_sec
                    or transcribe_elapsed_sec,
                },
                event_type="command_candidate_confirmation",
                jsonl_enabled=event_jsonl_enabled,
                jsonl_path=event_jsonl_path,
                sqlite_enabled=event_sqlite_enabled,
                sqlite_path=event_sqlite_path,
            )
            if candidate_reply_text and tts_enabled:
                speak(candidate_reply_text, "reply_command_candidate.wav")
            command_retry_active = True
            restore_music_volume(music_restore_volume)
            write_runtime_state(
                runtime_state_path,
                {
                    **runtime_state_base,
                    "state": "command_retry_wait",
                    "last_reset": "candidate_confirmation",
                    "pending_command_id": candidate_command_item.get("id"),
                    "recognized_fast": user_text,
                    "recognized_final": final_user_text or user_text,
                },
            )
            continue

        unknown_reply_text = unknown_command_reply.format(
            recognized=command_input_text or "それ"
        )
        print(f"INFO: command miss: {command_input_text}")
        append_event_logs(
            payload={
                "ts": int(time.time()),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "command_unknown",
                "mode": active_mode,
                "backend": transcription_backend,
                "model": final_transcribe_model,
                "recognized": command_input_text,
                "raw_user": final_user_text or user_text,
                "corrected_user": final_effective_user_text,
                "recognized_fast": user_text,
                "recognized_final": final_user_text or user_text,
                "model_fast": transcribe_model,
                "model_final": final_transcribe_model,
                "command_candidates": command_candidates,
                "elapsed_sec": final_transcribe_elapsed_sec or transcribe_elapsed_sec,
                "command_reply": unknown_reply_text,
            },
            event_type="command_unknown",
            jsonl_enabled=event_jsonl_enabled,
            jsonl_path=event_jsonl_path,
            sqlite_enabled=event_sqlite_enabled,
            sqlite_path=event_sqlite_path,
        )
        if unknown_reply_text and tts_enabled:
            speak(unknown_reply_text, "reply_command_unknown.wav")
        pending_command_confirmation = None
        command_retry_active = False
        restore_music_volume(music_restore_volume)
        write_runtime_state(
            runtime_state_path,
            {
                **runtime_state_base,
                "state": "wake_wait",
                "recognized_fast": user_text,
                "recognized_final": final_user_text or user_text,
                "last_reply": unknown_reply_text,
                "last_reset": "unknown_command",
            },
        )
        continue


if __name__ == "__main__":
    main()

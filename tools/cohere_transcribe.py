from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]


def die(message: str, code: int = 1) -> None:
    print(message)
    raise SystemExit(code)


def load_cfg() -> dict[str, Any]:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def load_cohere_cfg() -> dict[str, Any]:
    cfg = load_cfg()
    section = cfg.get("cohere_transcribe", {})
    if not isinstance(section, dict):
        section = {}
    return section


def require_command(command: str) -> None:
    if not shutil_which(command):
        die(f"NG: command not found: {command}")


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)


def resolve_api_key(cli_value: str | None = None) -> str:
    key = (cli_value or "").strip() or os.environ.get("COHERE_API_KEY", "").strip()
    if not key:
        die("NG: COHERE_API_KEY is not set")
    return key


def merge_cli_config(
    *,
    language: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    api_url: str | None = None,
    timeout_sec: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = load_cohere_cfg()
    if language:
        cfg["language"] = language
    if model is not None:
        cfg["model"] = model
    if prompt is not None:
        cfg["prompt"] = prompt
    if api_url:
        cfg["api_url"] = api_url
    if timeout_sec is not None:
        cfg["timeout_sec"] = timeout_sec
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    return cfg


def output_dir_from_cfg(cfg: dict[str, Any]) -> Path:
    raw = str(cfg.get("output_dir", "/tmp/voicechat/cohere_transcribe")).strip()
    return Path(raw).expanduser()


def ffmpeg_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("ffmpeg", {})
    return value if isinstance(value, dict) else {}


def arecord_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("arecord", {})
    return value if isinstance(value, dict) else {}


def ffmpeg_normalize(src: Path, dst: Path, cfg: dict[str, Any]) -> None:
    require_command("ffmpeg")
    ffmpeg = ffmpeg_cfg(cfg)
    sample_rate = str(ffmpeg.get("sample_rate", 16000))
    channels = str(ffmpeg.get("channels", 1))
    codec = str(ffmpeg.get("codec", "pcm_s16le")).strip() or "pcm_s16le"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        codec,
        "-ar",
        sample_rate,
        "-ac",
        channels,
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        die(proc.stdout or f"NG: ffmpeg failed rc={proc.returncode}")


def record_wav(dst: Path, seconds: int, cfg: dict[str, Any], device: str | None = None) -> None:
    require_command("arecord")
    arecord = arecord_cfg(cfg)
    audio_device = (device or str(arecord.get("device", ""))).strip()
    sample_rate = str(arecord.get("sample_rate", 16000))
    channels = str(arecord.get("channels", 1))
    audio_format = str(arecord.get("format", "S16_LE")).strip() or "S16_LE"
    cmd = ["arecord"]
    if audio_device:
        cmd += ["-D", audio_device]
    cmd += [
        "-f",
        audio_format,
        "-r",
        sample_rate,
        "-c",
        channels,
        "-d",
        str(seconds),
        str(dst),
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        die(f"NG: arecord failed rc={proc.returncode}")


def build_payload(cfg: dict[str, Any]) -> dict[str, str]:
    payload: dict[str, str] = {}
    model = str(cfg.get("model", "")).strip()
    language = str(cfg.get("language", "ja")).strip()
    prompt = str(cfg.get("prompt", "")).strip()
    if model:
        payload["model"] = model
    if language:
        payload["language"] = language
    if prompt:
        payload["prompt"] = prompt
    return payload


def parse_transcript_response(body: dict[str, Any]) -> str:
    direct_candidates = [
        body.get("text"),
        body.get("transcript"),
        body.get("transcription"),
    ]
    for item in direct_candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()

    results = body.get("results")
    if isinstance(results, list):
        parts: list[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            for key in ("text", "transcript"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
            else:
                alternatives = item.get("alternatives")
                if isinstance(alternatives, list):
                    for alt in alternatives:
                        if not isinstance(alt, dict):
                            continue
                        value = alt.get("text") or alt.get("transcript")
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                            break
        if parts:
            return "\n".join(parts)

    segments = body.get("segments")
    if isinstance(segments, list):
        parts = []
        for item in segments:
            if isinstance(item, dict):
                value = item.get("text")
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        if parts:
            return "\n".join(parts)

    return ""


def transcribe_audio(src: Path, cfg: dict[str, Any], api_key: str) -> tuple[str, dict[str, Any], float]:
    api_url = str(cfg.get("api_url", "https://api.cohere.ai/v1/audio/transcriptions")).strip()
    if not api_url:
        die("NG: cohere_transcribe.api_url is empty")
    timeout_sec = int(cfg.get("timeout_sec", 300))
    mime_type = mimetypes.guess_type(src.name)[0] or "audio/wav"
    payload = build_payload(cfg)
    headers = {"Authorization": f"Bearer {api_key}"}
    started = time.perf_counter()
    with src.open("rb") as handle:
        response = requests.post(
            api_url,
            headers=headers,
            data=payload,
            files={"file": (src.name, handle, mime_type)},
            timeout=timeout_sec,
        )
    elapsed = round(time.perf_counter() - started, 3)
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text}
    if response.status_code >= 400:
        pretty = json.dumps(body, ensure_ascii=False, indent=2) if isinstance(body, dict) else str(body)
        die(f"NG: Cohere API error {response.status_code}\n{pretty}")
    if not isinstance(body, dict):
        die("NG: Cohere API response is not JSON object")
    transcript = parse_transcript_response(body)
    if not transcript:
        pretty = json.dumps(body, ensure_ascii=False, indent=2)
        die(f"NG: transcript not found in response\n{pretty}")
    return transcript, body, elapsed


def save_result(
    *,
    tag: str,
    original_audio: Path,
    normalized_audio: Path,
    transcript: str,
    response_json: dict[str, Any],
    elapsed_sec: float,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{tag}.txt"
    json_path = out_dir / f"{tag}.json"
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_audio": str(original_audio),
        "normalized_audio": str(normalized_audio),
        "elapsed_sec": elapsed_sec,
        "text": transcript,
        "response": response_json,
    }
    txt_path.write_text(transcript + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"txt": txt_path, "json": json_path}


def run_transcription(
    *,
    src_audio: Path,
    cfg: dict[str, Any],
    api_key: str,
    tag: str,
    keep_normalized: bool = True,
) -> dict[str, Any]:
    if not src_audio.exists():
        die(f"NG: audio not found: {src_audio}")
    out_dir = output_dir_from_cfg(cfg)
    ffmpeg = ffmpeg_cfg(cfg)
    normalized_suffix = str(ffmpeg.get("format", "wav")).strip() or "wav"
    normalized_path = out_dir / f"{tag}_normalized.{normalized_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_normalize(src_audio, normalized_path, cfg)
    transcript, response_json, elapsed_sec = transcribe_audio(normalized_path, cfg, api_key)
    saved = save_result(
        tag=tag,
        original_audio=src_audio,
        normalized_audio=normalized_path,
        transcript=transcript,
        response_json=response_json,
        elapsed_sec=elapsed_sec,
        out_dir=out_dir,
    )
    if not keep_normalized:
        normalized_path.unlink(missing_ok=True)
    return {
        "text": transcript,
        "elapsed_sec": elapsed_sec,
        "normalized_audio": normalized_path,
        "txt_path": saved["txt"],
        "json_path": saved["json"],
    }


def temporary_wav_path(prefix: str = "cohere_record_") -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    return Path(path)

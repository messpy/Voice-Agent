#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VOICEVOX のずんだもん各スタイルで、id とスタイル名だけ読み上げる。"
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:50021",
        help="VOICEVOX engine host",
    )
    parser.add_argument(
        "--audio-out",
        default="plughw:CARD=B01,DEV=0",
        help="aplay の出力デバイス",
    )
    parser.add_argument(
        "--pause-sec",
        type=float,
        default=1.0,
        help="各読み上げの間隔",
    )
    parser.add_argument(
        "--volume-scale",
        type=float,
        default=0.1,
        help="VOICEVOX の volumeScale",
    )
    return parser.parse_args()


def get_zundamon_styles(host: str) -> list[dict]:
    resp = requests.get(host.rstrip("/") + "/speakers", timeout=10)
    resp.raise_for_status()
    speakers = resp.json()
    for speaker in speakers:
        if speaker.get("name") == "ずんだもん":
            return list(speaker.get("styles", []))
    raise RuntimeError("ずんだもん が speakers に見つからない")


def synthesize(host: str, speaker_id: int, text: str, volume_scale: float) -> bytes:
    host = host.rstrip("/")
    query_resp = requests.post(
        host + "/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=30,
    )
    query_resp.raise_for_status()
    query = query_resp.json()
    query["volumeScale"] = volume_scale

    synth_resp = requests.post(
        host + "/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=120,
    )
    synth_resp.raise_for_status()
    return synth_resp.content


def play_wav(wav_bytes: bytes, audio_out: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = Path(tmp.name)
    try:
        cmd = ["aplay"]
        if audio_out:
            cmd += ["-D", audio_out]
        cmd.append(str(tmp_path))
        subprocess.run(cmd, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    styles = get_zundamon_styles(args.host)

    print(json.dumps(styles, ensure_ascii=False, indent=2))
    for style in styles:
        style_id = int(style["id"])
        style_name = str(style["name"])
        text = f"{style_id} {style_name}"
        print(f"PLAY: {text}", flush=True)
        wav_bytes = synthesize(args.host, style_id, text, args.volume_scale)
        play_wav(wav_bytes, args.audio_out)
        time.sleep(max(0.0, args.pause_sec))


if __name__ == "__main__":
    main()

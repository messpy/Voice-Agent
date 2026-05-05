#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VOICEVOX の各話者で、話者名だけを読み上げるテスト。"
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
        default=0.8,
        help="各読み上げの間隔",
    )
    parser.add_argument(
        "--volume-scale",
        type=float,
        default=0.1,
        help="VOICEVOX の volumeScale",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        help="この speaker id だけを読む。省略すると全話者を順番に読む。",
    )
    return parser.parse_args()


def get_speakers(host: str) -> list[dict]:
    resp = requests.get(host.rstrip("/") + "/speakers", timeout=10)
    resp.raise_for_status()
    speakers = resp.json()
    if not isinstance(speakers, list):
        raise RuntimeError("VOICEVOX /speakers の応答が list ではない")
    return speakers


def pick_representative_styles(speakers: list[dict], speaker_id: int | None) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for speaker in speakers:
        name = str(speaker.get("name", "")).strip()
        styles = speaker.get("styles", [])
        if not name or not isinstance(styles, list) or not styles:
            continue
        if speaker_id is not None:
            if any(int(style.get("id", -1)) == speaker_id for style in styles if isinstance(style, dict)):
                items.append((name, speaker_id))
            continue
        style = next((item for item in styles if isinstance(item, dict) and item.get("type") == "talk"), styles[0])
        try:
            style_id = int(style["id"])
        except Exception:
            continue
        items.append((name, style_id))
    return items


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
    speakers = get_speakers(args.host)
    items = pick_representative_styles(speakers, args.speaker)
    if not items:
        raise SystemExit("話者が見つかりませんでした")

    print(f"count={len(items)}")
    for name, speaker_id in items:
        text = f"{name}"
        print(f"PLAY: [{speaker_id}] {text}", flush=True)
        wav_bytes = synthesize(args.host, speaker_id, text, args.volume_scale)
        play_wav(wav_bytes, args.audio_out)
        time.sleep(max(0.0, args.pause_sec))


if __name__ == "__main__":
    main()

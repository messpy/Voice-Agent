from __future__ import annotations
from pathlib import Path
import subprocess
import time

def run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode

def countdown(sec: int) -> None:
    for i in range(sec, 0, -1):
        print(f"INFO: start in {i}...")
        time.sleep(1)

def record_wav(out_wav: Path, audio_in: str, sec: int, rate: int, ch: int, fmt: str, countdown_sec: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # if: arecord exists
    if subprocess.run(["bash", "-lc", "command -v arecord >/dev/null"]).returncode != 0:
        raise RuntimeError("NG: arecord not found (alsa-utils)")

    print("========================================================================")
    print("RECORD")
    print("========================================================================")
    print(f"INFO: audio_in={audio_in} rate={rate} ch={ch} fmt={fmt} sec={sec}")
    if countdown_sec > 0:
        countdown(countdown_sec)

    cmd = [
        "arecord",
        "-D", audio_in,
        "-f", fmt,
        "-r", str(rate),
        "-c", str(ch),
        "-d", str(sec),
        str(out_wav),
    ]
    rc = run(cmd)
    if rc != 0:
        raise RuntimeError(f"NG: arecord failed rc={rc}")

    size = out_wav.stat().st_size
    print(f"OK: REC {out_wav.name} ({size} bytes)")

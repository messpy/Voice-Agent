from __future__ import annotations
from pathlib import Path
import subprocess
import time

def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)

def whisper_once(
    whisper_bin: Path,
    whisper_model: Path,
    wav: Path,
    lang: str,
    common_args: list[str],
    beam: int,
    best: int,
    temp: float,
    log_path: Path,
) -> tuple[int, float, str]:
    ensure(whisper_bin.exists(), f"NG: whisper bin missing: {whisper_bin}")
    ensure(whisper_model.exists(), f"NG: whisper model missing: {whisper_model}")
    ensure(wav.exists() and wav.stat().st_size > 0, f"NG: wav missing/empty: {wav}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(whisper_bin),
        "-m", str(whisper_model),
        "-f", str(wav),
        "-l", lang,
        "--beam-size", str(beam),
        "--best-of", str(best),
        "--temperature", str(temp),
        *common_args,
    ]

    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    out = (p.stdout or "").strip()

    log_path.write_text(out + "\n", encoding="utf-8")
    # whisperの最終テキストは末尾に出ることが多いので、最後の数行を拾う
    tail = "\n".join(out.splitlines()[-5:]).strip()
    return p.returncode, dt, tail

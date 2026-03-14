from __future__ import annotations
from pathlib import Path
import subprocess

def play_wav(wav: Path, audio_out: str) -> None:
    # if: aplay exists
    if subprocess.run(["bash", "-lc", "command -v aplay >/dev/null"]).returncode != 0:
        print("WARN: aplay not found (skip playback)")
        return
    print("========================================================================")
    print("PLAYBACK")
    print("========================================================================")
    print("INFO: 再生する（聞こえるか確認）")
    subprocess.run(["aplay", "-D", audio_out, str(wav)])

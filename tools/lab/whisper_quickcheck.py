import os
import sys
import time
import subprocess
from pathlib import Path

def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    raise SystemExit(code)

def get_env_path(name: str) -> str:
    v = os.environ.get(name, "").strip()
    return v

def main():
    whisper_bin = get_env_path("WHISPER_BIN") or str(Path.cwd() / "whisper.cpp/build/bin/whisper-cli")
    whisper_model = get_env_path("WHISPER_MODEL") or str(Path.cwd() / "whisper.cpp/models/ggml-base.bin")
    audio_in = get_env_path("AUDIO_IN") or "default"
    rec_sec = int(get_env_path("REC_SEC") or "10")
    lang = get_env_path("LANG_CODE") or "ja"

    wb = Path(whisper_bin)
    wm = Path(whisper_model)
    if not wb.exists():
        die(f"NG: WHISPER_BIN not found: {wb}")
    if not wm.exists():
        die(f"NG: WHISPER_MODEL not found: {wm}")

    out_dir = Path("/tmp/voicechat_quick")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"rec_{rec_sec}s.wav"
    outbase = out_dir / "out"

    print(f"INFO: 録音 {rec_sec}s（{audio_in}）")
    for i in (3,2,1):
        print(f"INFO: {i}...")
        time.sleep(1)

    # arecord: 16kHz mono
    cmd_rec = [
        "arecord",
        "-D", audio_in,
        "-d", str(rec_sec),
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        str(wav),
    ]
    rc = subprocess.run(cmd_rec).returncode
    if rc != 0:
        die(f"NG: arecord failed exit={rc}")

    if not wav.exists() or wav.stat().st_size == 0:
        die("NG: recorded wav missing/empty")

    # whisper: out.txt を必ず生成させる
    # -nt: no timestamps（テキストだけ）
    cmd_wh = [
        str(wb),
        "-m", str(wm),
        "-f", str(wav),
        "-l", lang,
        "-nt",
        "-otxt", "-of", str(outbase),
    ]

    # out を掃除
    for suf in (".txt", ".json", ".srt", ".vtt"):
        try:
            (Path(str(outbase) + suf)).unlink()
        except FileNotFoundError:
            pass

    rc = subprocess.run(cmd_wh).returncode
    if rc != 0:
        die(f"NG: whisper-cli failed exit={rc}")

    txt = Path(str(outbase) + ".txt")
    print("===== TRANSCRIPT =====")
    if txt.exists() and txt.stat().st_size > 0:
        print(txt.read_text(encoding="utf-8", errors="replace").strip())
    else:
        print("(no transcript)")
    print("======================")
    print(f"OK: WAV={wav}")
    print(f"OK: TXT={txt}")

if __name__ == "__main__":
    main()

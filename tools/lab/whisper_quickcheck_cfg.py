import time
import subprocess
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]

def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    raise SystemExit(code)

def load_cfg():
    p = ROOT / "config" / "config.yaml"
    if not p.exists():
        die(f"NG: config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def main():
    cfg = load_cfg()

    workdir = Path(cfg["paths"]["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    audio_in = cfg["audio"]["input"]
    sr = int(cfg["audio"]["sample_rate"])
    ch = int(cfg["audio"]["channels"])
    fmt = cfg["audio"]["format"]

    wb = Path(cfg["whisper"]["bin"])
    wm = Path(cfg["whisper"]["model"])
    lang = cfg["whisper"]["lang"]
    beam = int(cfg["whisper"]["beam"])
    best = int(cfg["whisper"]["best"])
    temp = float(cfg["whisper"]["temp"])
    threads = int(cfg["whisper"]["threads"])

    rec_sec = 10  # まず固定（あとで wake/vad 統合）

    if not wb.exists():
        die(f"NG: whisper bin not found: {wb}")
    if not wm.exists():
        die(f"NG: whisper model not found: {wm}")

    wav = workdir / f"rec_{rec_sec}s.wav"
    outbase = workdir / "out"

    print("INFO: whisper quickcheck (config.yaml)")
    print(f"INFO: AUDIO_IN={audio_in} sr={sr} ch={ch} fmt={fmt}")
    print(f"INFO: WHISPER lang={lang} beam={beam} best={best} temp={temp} threads={threads}")
    print(f"INFO: 録音 {rec_sec}s（話して）")

    for i in (3, 2, 1):
        print(f"INFO: {i}...")
        time.sleep(1)

    cmd_rec = [
        "arecord",
        "-D", audio_in,
        "-d", str(rec_sec),
        "-f", fmt,
        "-r", str(sr),
        "-c", str(ch),
        str(wav),
    ]
    rc = subprocess.run(cmd_rec).returncode
    if rc != 0:
        die(f"NG: arecord failed exit={rc}")

    if not wav.exists() or wav.stat().st_size == 0:
        die("NG: recorded wav missing/empty")

    # 既存出力掃除
    for suf in (".txt", ".json", ".srt", ".vtt"):
        try:
            Path(str(outbase) + suf).unlink()
        except FileNotFoundError:
            pass

    cmd_wh = [
        str(wb),
        "-m", str(wm),
        "-f", str(wav),
        "-l", lang,
        "-t", str(threads),
        "-bo", str(best),
        "-bs", str(beam),
        "-tp", str(temp),
        "-nt",
        "-otxt", "-of", str(outbase),
    ]
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

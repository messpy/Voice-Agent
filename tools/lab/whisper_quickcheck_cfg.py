import time
from pathlib import Path
import sys
import yaml

from localagent.recorder import record_wav
from localagent.whisper_runner import whisper_cpp_transcribe

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

    record_wav(wav, audio_in, rec_sec, sr, ch, fmt, 0)
    text, _elapsed = whisper_cpp_transcribe(
        whisper_bin=wb,
        model_path=wm,
        wav=wav,
        out_prefix=outbase,
        lang=lang,
        threads=threads,
        beam=beam,
        temperature=temp,
        extra_args=["-bo", str(best), "-nt"],
    )
    print("===== TRANSCRIPT =====")
    if text:
        print(text)
    else:
        print("(no transcript)")
    print("======================")
    print(f"OK: WAV={wav}")
    print(f"OK: TXT={outbase}.txt")

if __name__ == "__main__":
    main()

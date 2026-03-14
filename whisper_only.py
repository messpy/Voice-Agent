import os
import sys
import subprocess
from pathlib import Path
import time

TMP = Path("/tmp/voicechat_whisper")
TMP.mkdir(parents=True, exist_ok=True)

AUDIO_IN = os.environ.get("AUDIO_IN", "default")  # 例: hw:2,0 / plughw:2,0 / default
SECONDS  = float(os.environ.get("REC_SEC", "5"))

def die(msg: str, code: int = 1):
    print(f"NG: {msg}", file=sys.stderr)
    raise SystemExit(code)

def which(cmd: str) -> str:
    p = subprocess.run(["bash", "-lc", f"command -v {cmd}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return p.stdout.strip()

def find_whisper_bin() -> str:
    # env優先
    wb = os.environ.get("WHISPER_BIN", "").strip()
    if wb and Path(wb).exists():
        return wb

    # PATH上
    for name in ("whisper-cli", "main"):
        p = which(name)
        if p and Path(p).exists():
            return p

    # よくある場所を軽く探索（深すぎる探索は避ける）
    home = Path.home()
    for base in (home, home / "whisper.cpp", home / "src", home / "bin"):
        if base.exists():
            hits = []
            try:
                hits = list(base.rglob("whisper-cli")) + list(base.rglob("main"))
            except Exception:
                pass
            for h in hits:
                if h.is_file() and os.access(h, os.X_OK):
                    return str(h)

    return ""

def find_whisper_model() -> str:
    wm = os.environ.get("WHISPER_MODEL", "").strip()
    if wm and Path(wm).exists():
        return wm

    home = Path.home()
    # よくある場所
    candidates = []
    for base in (home, home / "whisper.cpp", home / ".cache", home / ".cache/whispercpp"):
        if base.exists():
            try:
                candidates += list(base.rglob("ggml-*.bin"))
            except Exception:
                pass

    # tiny/base/small を優先
    prio = ("ggml-tiny", "ggml-base", "ggml-small")
    candidates.sort(key=lambda p: (next((i for i, k in enumerate(prio) if k in p.name), 999), len(str(p))))
    for c in candidates:
        if c.is_file() and c.stat().st_size > 1024 * 1024:
            return str(c)
    return ""

def if_check():
    if not which("arecord"):
        die("arecord not found")
    wb = find_whisper_bin()
    wm = find_whisper_model()
    if not wb:
        die("whisper.cpp binary not found. Set WHISPER_BIN or put whisper-cli/main in PATH")
    if not wm:
        die("whisper model not found. Set WHISPER_MODEL (ggml-*.bin)")
    return wb, wm

def record_wav(out_wav: Path, seconds: float):
    # 16kHz mono wav
    cmd = [
        "arecord",
        "-D", AUDIO_IN,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", str(int(seconds)),
        str(out_wav),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        die(f"arecord failed: {r.stderr.strip()}")

def transcribe(wb: str, wm: str, wav: Path) -> str:
    out_prefix = TMP / "out"
    txt = out_prefix.with_suffix(".txt")
    if txt.exists():
        txt.unlink()

    # whisper.cpp: main/whisper-cli の一般的な形式
    cmd = [wb, "-m", wm, "-f", str(wav), "-l","ja","-otxt", "-of", str(out_prefix)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        die(f"whisper failed: {r.stderr.strip() or r.stdout.strip()}")

    if txt.exists():
        return txt.read_text(errors="ignore").strip()

    # 実装差異: stdoutに出る場合
    s = (r.stdout or "").strip()
    if s:
        return s
    return ""

def main():
    wb, wm = if_check()
    print(f"OK: WHISPER_BIN={wb}")
    print(f"OK: WHISPER_MODEL={wm}")
    print(f"OK: AUDIO_IN={AUDIO_IN} REC_SEC={int(SECONDS)}")

    wav = TMP / "rec.wav"
    print("INFO: 録音開始（話して）...")
    record_wav(wav, seconds=SECONDS)
    print(f"OK: RECORDED {wav} ({wav.stat().st_size} bytes)")

    print("INFO: 文字起こし中...")
    text = transcribe(wb, wm, wav)
    print("===== TRANSCRIPT =====")
    print(text if text else "(empty)")
    print("======================")

if __name__ == "__main__":
    main()

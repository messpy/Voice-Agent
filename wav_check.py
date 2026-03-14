import argparse
import math
import os
import subprocess
import wave
from pathlib import Path

def analyze_wav(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty: {path}")

    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        dur = nframes / fr if fr else 0.0
        raw = wf.readframes(nframes)

    # PCM16/PCM32/PCM8 のざっくり対応（piperは基本PCM16）
    if sampwidth == 2:
        import struct
        it = struct.iter_unpack("<h", raw)
        max_abs = 0
        sum_sq = 0.0
        count = 0
        for (v,) in it:
            av = abs(v)
            if av > max_abs:
                max_abs = av
            sum_sq += v * v
            count += 1
        peak = max_abs / 32768.0 if count else 0.0
        rms = math.sqrt(sum_sq / count) / 32768.0 if count else 0.0
    elif sampwidth == 1:
        # unsigned 8bit
        max_abs = 0
        sum_sq = 0.0
        count = 0
        for b in raw:
            v = b - 128
            av = abs(v)
            if av > max_abs:
                max_abs = av
            sum_sq += v * v
            count += 1
        peak = max_abs / 128.0 if count else 0.0
        rms = math.sqrt(sum_sq / count) / 128.0 if count else 0.0
    elif sampwidth == 4:
        import struct
        it = struct.iter_unpack("<i", raw)
        max_abs = 0
        sum_sq = 0.0
        count = 0
        for (v,) in it:
            av = abs(v)
            if av > max_abs:
                max_abs = av
            sum_sq += v * v
            count += 1
        peak = max_abs / 2147483648.0 if count else 0.0
        rms = math.sqrt(sum_sq / count) / 2147483648.0 if count else 0.0
    else:
        peak = float("nan")
        rms = float("nan")

    return {
        "path": str(path),
        "channels": nch,
        "sample_width": sampwidth,
        "sample_rate": fr,
        "frames": nframes,
        "duration_sec": dur,
        "peak": peak,
        "rms": rms,
        "size_bytes": path.stat().st_size,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="/tmp/tts.wav")
    ap.add_argument("--log", default="audio_check.txt")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--device", default="plughw:3,0")  # USB B01想定
    args = ap.parse_args()

    wav = Path(args.wav)
    log = Path(args.log)

    # if: ファイル存在
    if not wav.exists() or wav.stat().st_size == 0:
        log.write_text(f"NG: wav missing or empty: {wav}\n", encoding="utf-8")
        raise SystemExit(1)

    info = analyze_wav(wav)

    # 無音っぽい判定（厳密じゃないが実用）
    judge = "WAV has AUDIO"
    if info["rms"] != info["rms"]:  # NaN
        judge = "UNKNOWN (unsupported sample width)"
    elif info["rms"] < 0.001 and info["peak"] < 0.01:
        judge = "LIKELY SILENCE"

    out = []
    out.append("===== WAV ANALYZE =====")
    for k in ["path","size_bytes","channels","sample_width","sample_rate","frames","duration_sec","peak","rms"]:
        out.append(f"{k}: {info[k]}")
    out.append(f"JUDGE: {judge}")
    out.append("")

    # 再生（必要なら）
    if args.play:
        out.append("===== PLAYBACK =====")
        out.append(f"cmd: aplay -D {args.device} {wav}")
        try:
            cp = subprocess.run(
                ["aplay", "-D", args.device, str(wav)],
                text=True, capture_output=True
            )
            out.append("--- stdout ---")
            out.append(cp.stdout.strip())
            out.append("--- stderr ---")
            out.append(cp.stderr.strip())
            out.append(f"exit_code: {cp.returncode}")
        except FileNotFoundError:
            out.append("NG: aplay not found")
        out.append("")

    log.write_text("\n".join(out) + "\n", encoding="utf-8")

    # 画面にも要点だけ出す
    print("\n".join(out[-20:]))

if __name__ == "__main__":
    main()

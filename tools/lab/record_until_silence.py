import os
import sys
import time
import wave
import struct
import subprocess
from pathlib import Path

import webrtcvad
from src.config_loader import load_cfg

def ensure(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def arecord_cmd(cfg):
    dev = cfg["audio"].get("input", "default")
    sr = int(cfg["audio"].get("sample_rate", 16000))
    ch = int(cfg["audio"].get("channels", 1))
    fmt = cfg["audio"].get("format", "S16_LE")
    # arecord: raw PCM を stdout に吐かせる
    return ["arecord", "-D", dev, "-f", fmt, "-r", str(sr), "-c", str(ch), "-t", "raw", "-q"]

def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # S16_LE
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def main():
    cfg = load_cfg()

    sr = int(cfg["audio"].get("sample_rate", 16000))
    ch = int(cfg["audio"].get("channels", 1))
    ensure(sr == 16000, "NG: VADはまず16kHz固定で運用（config audio.sample_rate=16000 にして）")
    ensure(ch == 1, "NG: VADはまずmono固定で運用（config audio.channels=1 にして）")

    vad_cfg = cfg.get("vad", {})
    ensure(vad_cfg.get("enabled", True), "NG: vad.enabled=false")

    aggr = int(vad_cfg.get("aggressiveness", 2))
    silence_ms = int(vad_cfg.get("silence_ms", 900))
    max_sec = int(vad_cfg.get("max_record_sec", 20))

    # VAD frame: 20ms が扱いやすい (webrtcvad対応)
    frame_ms = 20
    frame_samples = sr * frame_ms // 1000
    frame_bytes = frame_samples * 2  # mono S16_LE

    vad = webrtcvad.Vad(aggr)

    outdir = Path(cfg.get("paths", {}).get("workdir", "/tmp/voicechat")) / "rec"
    outdir.mkdir(parents=True, exist_ok=True)
    out_wav = outdir / f"utt_{time.strftime('%Y%m%d_%H%M%S')}.wav"

    print(f"INFO: record_until_silence start (max={max_sec}s, silence={silence_ms}ms, aggr={aggr})")
    print("INFO: 3...")
    time.sleep(1)
    print("INFO: 2...")
    time.sleep(1)
    print("INFO: 1...")
    time.sleep(1)

    proc = subprocess.Popen(
        arecord_cmd(cfg),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    ensure(proc.stdout is not None, "NG: arecord stdout missing")

    buf = bytearray()
    silence_acc_ms = 0
    started = time.time()

    try:
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk or len(chunk) < frame_bytes:
                break

            buf.extend(chunk)

            # VAD判定
            is_speech = vad.is_speech(chunk, sr)
            if is_speech:
                silence_acc_ms = 0
            else:
                silence_acc_ms += frame_ms

            elapsed = time.time() - started
            if elapsed >= max_sec:
                print("INFO: stop by max_record_sec")
                break
            if silence_acc_ms >= silence_ms and elapsed > 0.6:
                # 最低0.6sは録る（誤判定で即停止しない）
                print("INFO: stop by silence")
                break
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        proc.wait(timeout=2)

    pcm = bytes(buf)
    write_wav(out_wav, pcm, sr, ch)
    print(f"OK: saved {out_wav} bytes={len(pcm)}")

if __name__ == "__main__":
    main()

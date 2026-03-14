from __future__ import annotations

import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

from src.config_loader import load_cfg, ensure

def which(name: str) -> str | None:
    from shutil import which as _w
    return _w(name)

def resolve_whisper_bin() -> str:
    # env優先
    wb = os.environ.get("WHISPER_BIN", "").strip()
    if wb and Path(wb).exists():
        return wb

    # PATH上（whisper-cli / main）
    for name in ("whisper-cli", "main"):
        p = which(name)
        if p and Path(p).exists():
            return p

    # よくある場所（深掘りしない）
    root = Path.cwd()
    candidates = [
        root / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise RuntimeError("NG: whisper.cpp binary not found. Set WHISPER_BIN or put whisper-cli/main in PATH.")

def resolve_model() -> str:
    wm = os.environ.get("WHISPER_MODEL", "").strip()
    if wm and Path(wm).exists():
        return wm

    root = Path.cwd()
    candidates = [
        root / "whisper.cpp" / "models" / "ggml-base.bin",
        Path.home() / "whisper.cpp" / "models" / "ggml-base.bin",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise RuntimeError("NG: whisper model not found. Set WHISPER_MODEL.")

def sh(cmd: list[str], *, input_bytes: bytes | None = None, out_log: Path | None = None) -> int:
    if out_log:
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with out_log.open("wb") as f:
            p = subprocess.run(cmd, input=input_bytes, stdout=f, stderr=subprocess.STDOUT)
            return p.returncode
    p = subprocess.run(cmd, input=input_bytes)
    return p.returncode

def countdown(sec: int) -> None:
    for i in range(sec, 0, -1):
        print(f"COUNTDOWN: {i}...")
        time.sleep(1)

def record_wav(out_wav: Path, seconds: int, rate: int, ch: int, in_dev: str) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # arecord: ALSA capture
    cmd = [
        "arecord",
        "-D", in_dev,
        "-f", "S16_LE",
        "-r", str(rate),
        "-c", str(ch),
        "-d", str(seconds),
        str(out_wav),
    ]
    print("INFO: 録音開始（話して）...")
    rc = sh(cmd)
    ensure(rc == 0, f"NG: arecord failed rc={rc}")
    ensure(out_wav.exists() and out_wav.stat().st_size > 0, f"NG: rec wav missing: {out_wav}")
    print(f"OK: REC {out_wav.name} ({out_wav.stat().st_size} bytes)")

def play_wav(wav: Path, out_dev: str) -> None:
    # 対話型：実際に聞こえるか確認用に、そのまま aplay
    cmd = ["aplay"]
    if out_dev and out_dev != "default":
        cmd += ["-D", out_dev]
    cmd += [str(wav)]
    print("INFO: 再生（聞こえるか確認）...")
    sh(cmd)

def whisper_cmd(wbin: str, wmodel: str, wav: Path, lang: str, threads: int, p: dict, out_prefix: Path, timestamps: bool) -> list[str]:
    # whisper-cli 出力は -of PREFIX -otxt を使い、TEXT は PREFIX.txt を読む
    cmd = [
        wbin,
        "-m", wmodel,
        "-f", str(wav),
        "-l", lang,
        "-t", str(threads),
        "-of", str(out_prefix),
        "-otxt",
    ]

    # timestamps OFF の場合、SRT的な行を減らしたいが whisper-cli の出力形式は実装依存。
    # ここでは transcript は txt を読んで「認識文だけ」を抽出する。
    # パラメータ適用（壊れない組）
    temp = float(p.get("temp", 0.0))
    beam = int(p.get("beam", 5))
    best = int(p.get("best", 5))

    if temp == 0.0:
        # beam search
        cmd += ["--beam-size", str(beam)]
        # best-of は sampling 用なので入れない（exit=10回避）
    else:
        # sampling
        cmd += ["--temperature", str(temp), "--best-of", str(best), "--beam-size", "1"]

    return cmd

def read_transcript_txt(txt_path: Path) -> str:
    if not txt_path.exists():
        return ""
    raw = txt_path.read_text(errors="replace")

    # 余計なものを削る（timingsや"output_txt:" 等が混ざる環境があるため）
    # 1) whisper_print_timings 以降を除外
    raw = raw.split("whisper_print_timings:", 1)[0]

    # 2) "output_txt:" 行を削除
    raw = re.sub(r"^output_txt:.*$", "", raw, flags=re.M)

    # 3) timestamps行が入ってたら [..] を除去して本文だけ取り出す
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", line)
        # それでも残るノイズは弾く
        if "whisper_model_load" in line or "system_info" in line:
            continue
        lines.append(line)

    # 連結
    return "\n".join(lines).strip()

def prompt_rate() -> tuple[int, str]:
    while True:
        s = input("評価(1-5) > ").strip()
        if re.fullmatch(r"[1-5]", s):
            break
        print("NG: 1〜5 の数字で入力して")
    memo = input("メモ(任意。空でOK) > ").rstrip("\n")
    return int(s), memo

def main() -> None:
    cfg = load_cfg()

    # ===== INFO =====
    print("=" * 72)
    print("WHISPER BENCH: record -> play -> transcribe -> rate")
    print("=" * 72)

    wbin = resolve_whisper_bin()
    wmodel = resolve_model()

    audio = cfg.get("audio", {})
    whisper = cfg.get("whisper", {})
    meta = cfg.get("meta", {})

    in_dev = str(audio.get("in_dev", "default"))
    out_dev = str(audio.get("out_dev", "default"))
    rate = int(audio.get("rate", 16000))
    ch = int(audio.get("channels", 1))
    sec = int(audio.get("seconds", 10))

    lang = str(whisper.get("lang", "ja"))
    threads = int(whisper.get("threads", 4))
    timestamps = bool(whisper.get("print_timestamps", False))

    params = cfg.get("params", [])
    ensure(isinstance(params, list) and len(params) > 0, "NG: パラメータセットが0件")

    print(f"INFO: name={meta.get('name','')}")
    print(f"INFO: note={meta.get('note','')}")
    print(f"INFO: WHISPER_BIN={wbin}")
    print(f"INFO: WHISPER_MODEL={wmodel}")
    print(f"INFO: AUDIO_IN={in_dev} AUDIO_OUT={out_dev} rate={rate} ch={ch} sec={sec}")
    print(f"INFO: lang={lang} threads={threads}")
    print("INFO: params:")
    for p in params:
        print(f"  - {p.get('id')} beam={p.get('beam')} best={p.get('best')} temp={p.get('temp')}")
    print()

    # ===== WORKDIR =====
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = Path("/tmp/voice_bench") / f"bench_{ts}"
    rec_dir = bench_dir / "rec"
    log_dir = bench_dir / "log"
    out_dir = bench_dir / "out"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # ===== RECORD (10s x 1) =====
    print("=" * 72)
    print("RECORD")
    print("=" * 72)

    wav = rec_dir / "rec_10s_take1.wav"
    print("INFO: 録音前カウントダウン（3秒）")
    countdown(3)
    record_wav(wav, seconds=sec, rate=rate, ch=ch, in_dev=in_dev)

    # ===== PLAYBACK (確認) =====
    print("=" * 72)
    print("PLAYBACK + TRANSCRIBE")
    print("=" * 72)

    play_wav(wav, out_dev=out_dev)

    # ===== RUN GRID =====
    results_path = bench_dir / "results.jsonl"
    n_total = len(params) * 2
    idx = 0

    for p in params:
        pid = str(p.get("id", "PX"))
        for run_i in (1, 2):
            idx += 1
            prefix = out_dir / f"{wav.stem}__{pid}__run{run_i}"
            out_txt = Path(str(prefix) + ".txt")
            out_log = log_dir / f"log_{wav.stem}__{pid}__run{run_i}.txt"

            cmd = whisper_cmd(
                wbin=wbin,
                wmodel=wmodel,
                wav=wav,
                lang=lang,
                threads=threads,
                p=p,
                out_prefix=prefix,
                timestamps=timestamps,
            )

            t0 = time.time()
            rc = sh(cmd, out_log=out_log)
            dt = time.time() - t0

            # TEXT は transcript(txt) のみ
            text = read_transcript_txt(out_txt)

            print("-" * 72)
            print(f"[{idx}/{n_total}]")
            print(f"WAV   : {wav.name} (10s take1)")
            print(f"PARAM : {pid} beam={p.get('beam')} best={p.get('best')} temp={p.get('temp')}")
            print(f"RUN   : {run_i}/2")
            print(f"EXIT  : {rc}")
            print(f"TIME  : {dt:.2f}s")
            print(f"TEXT  : {text if text else '(no transcript)'}")
            print(f"LOG   : {out_log}")

            # 評価（失敗でも評価は入れられるが、基準はあなたに任せる）
            rate_score, memo = prompt_rate()

            rec = {
                "ts": ts,
                "wav": str(wav),
                "param_id": pid,
                "param": p,
                "run": run_i,
                "exit": rc,
                "sec": sec,
                "time_sec": round(dt, 3),
                "text": text,
                "log": str(out_log),
                "out_txt": str(out_txt),
                "rating": rate_score,
                "memo": memo,
            }
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"OK: saved {results_path}")
    print(f"OK: bench_dir {bench_dir}")

if __name__ == "__main__":
    # src import を安定化（`cd ~/voicechat` 前提）
    # もしカレントがズレたら、ここで root を sys.path に入れる
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    main()

#!/usr/bin/env python3
import os
import re
import csv
import json
import time
import shlex
import shutil
import subprocess
from pathlib import Path

TMP = Path("/tmp/voice_bench")
TMP.mkdir(parents=True, exist_ok=True)

def which(x): return shutil.which(x)

def run(cmd, *, input_text=None, capture=True):
    p = subprocess.run(
        cmd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    out = (p.stdout.decode("utf-8", errors="replace") if capture and p.stdout is not None else "")
    return p.returncode, out

def ask(prompt, default=None):
    if default is None:
        return input(prompt).strip()
    s = input(f"{prompt} [{default}] ").strip()
    return s if s else default

def header(t):
    print("\n" + "="*70)
    print(t)
    print("="*70, flush=True)

def ensure(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def normalize_text(s: str) -> str:
    # ざっくり比較用（記号/空白を寄せる）
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[。．\.]+", "。", s)
    s = re.sub(r"[、,]+", "、", s)
    return s

def extract_transcript(stdout: str) -> str:
    # whisper-cli の出力は色々混ざるので、本文っぽい行だけ拾って連結
    lines = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("whisper_print_timings:") or s.startswith("main:"):
            continue
        # 明らかにログっぽいのは除外
        if any(k in s.lower() for k in ["threads", "processors", "load time", "total time", "processing"]):
            if "この日は" in s:  # 実際の本文になることがあるので例外
                pass
            else:
                continue
        # timestamp形式 [00:00.000 --> ...] を落とす
        s = re.sub(r"^\[[0-9:\.\s\-–>]+\]\s*", "", s)
        # 日本語 or 英字が含まれてたら本文扱い
        if re.search(r"[ぁ-んァ-ン一-龯A-Za-z0-9]", s):
            lines.append(s)
    return normalize_text(" ".join(lines))

def record_wav(audio_in: str, sec: int, out_wav: Path):
    ensure(which("arecord"), "NG: arecord not found")
    cmd = ["arecord", "-D", audio_in, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(sec), str(out_wav)]
    rc, out = run(cmd, capture=True)
    if rc != 0:
        raise RuntimeError(f"record failed: {out}")
    return out_wav

def whisper_once(wbin: str, wmodel: str, wav: Path, lang: str, args: list[str], out_log: Path):
    cmd = [wbin, "-m", wmodel, "-f", str(wav), "--no-timestamps"]
    if lang and lang != "auto":
        cmd += ["-l", lang]
    cmd += args
    t0 = time.time()
    rc, out = run(cmd, capture=True)
    dt = time.time() - t0
    out_log.write_text(out, encoding="utf-8")
    if rc != 0:
        # 重要そうなところだけ拾って返す
        err_lines = []
        for line in out.splitlines()[-120:]:
            if any(k in line.lower() for k in ["error", "failed", "abort", "invalid"]):
                err_lines.append(line)
        raise RuntimeError(f"whisper failed exit={rc} log={out_log}\n" + "\n".join(err_lines))
    text = extract_transcript(out)
    return text, dt

def parse_param_sets(text: str):
    """
    入力例:
      beam=5 best=5 temp=0.0
      beam=10 best=10 temp=0.0
      beam=5 best=5 temp=0.2
    """
    sets = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kv = {}
        for token in line.split():
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            kv[k.strip()] = v.strip()
        # whisper-cli 引数へ
        args = []
        if "beam" in kv: args += ["--beam-size", kv["beam"]]
        if "best" in kv: args += ["--best-of", kv["best"]]
        if "temp" in kv: args += ["--temperature", kv["temp"]]
        if "max_tokens" in kv: args += ["--max-tokens", kv["max_tokens"]]
        if "threads" in kv: args += ["-t", kv["threads"]]
        # 記録用
        sets.append({"raw": line, "args": args})
    return sets

def main():
    header("WHISPER BENCH (録音→パラメータ総当り→比較用ログ保存)")

    # if: whisper bin/model
    wbin = os.environ.get("WHISPER_BIN", "").strip() or ask("WHISPER_BIN", str(Path.cwd()/ "whisper.cpp/build/bin/whisper-cli"))
    wmodel = os.environ.get("WHISPER_MODEL", "").strip() or ask("WHISPER_MODEL", str(Path.cwd()/ "whisper.cpp/models/ggml-base.bin"))
    ensure(Path(wbin).exists(), f"NG: whisper bin not found: {wbin}")
    ensure(Path(wmodel).exists(), f"NG: whisper model not found: {wmodel}")

    # if: arecord
    ensure(which("arecord"), "NG: arecord not found")

    audio_in = os.environ.get("AUDIO_IN", "").strip() or ask("AUDIO_IN (例: default / hw:2,0 / plughw:2,0)", "default")
    lang = ask("言語 (ja/en/auto)", "ja")

    # 録音設定
    durations_str = ask("録音秒数リスト (例: 5,20,40)", "5,20")
    durations = [int(x.strip()) for x in durations_str.split(",") if x.strip()]
    takes = int(ask("各秒数あたりの録音回数 (例: 2)", "2"))

    # 同一パラメータを何回回すか（温度>0 の揺れ確認）
    runs_per_param = int(ask("同一パラメータを同じwavに何回実行する？", "2"))

    header("パラメータセット入力（複数行。空行で終了）")
    print("例:")
    print("  beam=5 best=5 temp=0.0")
    print("  beam=10 best=10 temp=0.0")
    print("  beam=5 best=5 temp=0.2")
    print("対応キー: beam best temp max_tokens threads")
    lines = []
    while True:
        s = input("> ").strip()
        if not s:
            break
        lines.append(s)
    ensure(lines, "NG: パラメータセットが0件")

    param_sets = parse_param_sets("\n".join(lines))
    ensure(param_sets, "NG: パラメータ解析に失敗")

    # 期待文字列（任意）: WERっぽい比較をしたいなら入力
    ref = ask("期待する文章（空ならスキップ）", "")
    ref_norm = normalize_text(ref) if ref else ""

    tag = now_tag()
    out_dir = TMP / f"bench_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": tag,
        "wbin": wbin,
        "wmodel": wmodel,
        "audio_in": audio_in,
        "lang": lang,
        "durations": durations,
        "takes": takes,
        "runs_per_param": runs_per_param,
        "param_sets": [p["raw"] for p in param_sets],
        "ref": ref,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    header(f"録音開始: {out_dir}")
    wavs = []
    for sec in durations:
        for i in range(1, takes+1):
            wav = out_dir / f"rec_{sec}s_take{i}.wav"
            print(f"INFO: 録音 {sec}s take{i}/{takes}（話して）...")
            record_wav(audio_in, sec, wav)
            print(f"OK: {wav} ({wav.stat().st_size} bytes)")
            wavs.append(wav)

    header("総当り開始")
    rows = []
    # CSV (比較用)
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "wav","param_raw","run_idx","transcript",
            "transcript_norm","ref_norm",
            "match_exact_norm","elapsed_sec",
            "log_path"
        ])

        for wav in wavs:
            for p in param_sets:
                for r in range(1, runs_per_param+1):
                    log = out_dir / f"log_{wav.stem}__p{param_sets.index(p)+1}__run{r}.txt"
                    text, dt = whisper_once(wbin, wmodel, wav, lang, p["args"], log)
                    text_norm = normalize_text(text)
                    match = (text_norm == ref_norm) if ref_norm else ""
                    w.writerow([
                        str(wav), p["raw"], r, text,
                        text_norm, ref_norm,
                        match, f"{dt:.3f}",
                        str(log)
                    ])
                    print(f"OK: {wav.name} | {p['raw']} | run{r} | {dt:.2f}s | {text_norm}")

    header("完了")
    print(f"OK: results={csv_path}")
    print(f"OK: dir={out_dir}")
    print("次: results.csv を見て、良さそうな param を絞って再実行して。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"NG: {e}")
        raise

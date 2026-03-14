#!/usr/bin/env python3
import os
import re
import time
import json
import shutil
import signal
import subprocess
from pathlib import Path

import requests

# =========================
# CONFIG (必要ならここだけ調整)
# =========================

# wake word（完全一致に近い判定。Whisperのブレも考慮して複数候補）
WAKE_WORDS = ["べりべり", "ベリベリ", "very very", "beri beri", "veri veri"]

# 録音設定（wake検出用は短く、質問取り込みは長め）
REC_RATE = int(os.getenv("REC_RATE", "16000"))
REC_CH = int(os.getenv("REC_CH", "1"))
WAKE_SEC = float(os.getenv("WAKE_SEC", "1.6"))
QUERY_SEC = float(os.getenv("QUERY_SEC", "6.0"))

# 再生デバイス（例: "plughw:3,0" や "default"）
AUDIO_OUT = os.getenv("AUDIO_OUT", "default")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# コマンド実行を許可するか（危険なのでデフォルト禁止）
ALLOW_COMMANDS = os.getenv("ALLOW_COMMANDS", "0") == "1"

# open_jtalk（自動検出するが、固定したいなら環境変数で上書き）
OJT_BIN = os.getenv("OJT_BIN", "open_jtalk")
OJT_VOICE = os.getenv("OJT_VOICE", "")   # 空なら自動探索
OJT_DIC = os.getenv("OJT_DIC", "")       # 空なら自動探索

# whisper.cpp（自動検出するが、固定したいなら環境変数で上書き）
WHISPER_BIN = os.getenv("WHISPER_BIN", "")     # 空なら自動探索
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "") # 空なら自動探索

# =========================
# Helpers
# =========================

def die(msg: str, code: int = 1):
    print(f"NG: {msg}")
    raise SystemExit(code)

def run(cmd, *, inp: bytes | None = None, check=True, capture=False):
    if capture:
        return subprocess.run(cmd, input=inp, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, input=inp, check=check)

def which_or_empty(name: str) -> str:
    p = shutil.which(name)
    return p or ""

def find_first_existing(cands):
    for c in cands:
        if c and Path(c).exists():
            return str(Path(c))
    return ""

def detect_whisper_bin() -> str:
    if WHISPER_BIN and Path(WHISPER_BIN).exists():
        return WHISPER_BIN

    cands = [
        which_or_empty("whisper-cli"),
        which_or_empty("main"),
        str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        str(Path.home() / "whisper.cpp" / "build" / "bin" / "main"),
        str(Path.home() / "whisper.cpp" / "main"),
    ]
    p = find_first_existing(cands)
    if not p:
        die("whisper.cpp binary not found. Set WHISPER_BIN or put whisper-cli/main in PATH.")
    return p

def detect_whisper_model() -> str:
    if WHISPER_MODEL and Path(WHISPER_MODEL).exists():
        return WHISPER_MODEL

    base = Path.home() / "whisper.cpp" / "models"
    cands = [
        base / "ggml-small.bin",
        base / "ggml-base.bin",
        base / "ggml-tiny.bin",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    die("whisper model not found. Set WHISPER_MODEL (e.g., ~/whisper.cpp/models/ggml-small.bin).")
    return ""

def detect_open_jtalk_voice() -> str:
    if OJT_VOICE and Path(OJT_VOICE).exists():
        return OJT_VOICE

    # よくある配置
    cands = [
        "/usr/share/hts-voice/hts_voice_nitech_jp_atr503_m001.htsvoice",
        "/usr/share/hts-voice/hts-voice-nitech-jp-atr503-m001/hts_voice_nitech_jp_atr503_m001.htsvoice",
    ]
    p = find_first_existing(cands)
    if p:
        return p

    # 再帰探索（遅いので /usr/share/hts-voice を優先）
    base = Path("/usr/share/hts-voice")
    if base.is_dir():
        hits = list(base.rglob("*.htsvoice"))
        if hits:
            hits.sort(key=lambda x: ("nitech" not in x.name.lower(), len(str(x))))
            return str(hits[0])

    die("open_jtalk voice (.htsvoice) not found. Install hts voice or set OJT_VOICE.")
    return ""

def detect_open_jtalk_dic() -> str:
    if OJT_DIC and Path(OJT_DIC).is_dir():
        return OJT_DIC

    cands = [
        "/var/lib/mecab/dic/open-jtalk/naist-jdic",
        "/usr/share/mecab/dic/open-jtalk/naist-jdic",
        "/usr/share/open_jtalk/dic",
        "/usr/share/open_jtalk/dic/naist-jdic",
    ]
    for c in cands:
        if Path(c).is_dir():
            return c

    die("open_jtalk dictionary not found. Install open-jtalk + naist-jdic or set OJT_DIC.")
    return ""

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_wake(s: str) -> bool:
    t = s.lower()
    for w in WAKE_WORDS:
        if w.lower() in t:
            return True
    return False

def record_wav(out_wav: str, seconds: float):
    # arecord: S16_LE, mono, 16k
    cmd = [
        "arecord",
        "-q",
        "-f", "S16_LE",
        "-r", str(REC_RATE),
        "-c", str(REC_CH),
        "-d", str(seconds),
        out_wav,
    ]
    run(cmd, check=True)

def whisper_transcribe(wav_path: str) -> str:
    # whisper.cpp: 出力はstdoutにテキストを出す設定で取得
    # -nt: timestamps無効, -of -: stdout出力（whisper-cliは -of - が通る想定）
    # 実装差異があるので、stdout取得して最後のテキストっぽい行を拾う
    cmd = [
        WHISPER_BIN_PATH,
        "-m", WHISPER_MODEL_PATH,
        "-f", wav_path,
        "-nt",
    ]

    p = run(cmd, capture=True, check=False)
    out = (p.stdout or b"").decode("utf-8", errors="ignore")
    err = (p.stderr or b"").decode("utf-8", errors="ignore")

    # 失敗時はstderrも含めて解析
    merged = "\n".join([out, err]).strip()

    # それっぽい最終行を抽出
    lines = [ln.strip() for ln in merged.splitlines() if ln.strip()]
    if not lines:
        return ""
    # whisper.cppは進捗ログが混じるので、短く日本語/英字がある行を優先
    cand = ""
    for ln in reversed(lines):
        if any(ch.isalnum() for ch in ln) or any("\u3040" <= ch <= "\u30ff" for ch in ln):
            cand = ln
            break
    return normalize_text(cand or lines[-1])

def ollama_chat(user_text: str) -> str:
    # /api/chat を使う（なければ /api/generate に落とす）
    system = (
        "あなたは音声アシスタント。短く分かりやすく回答する。"
        "ユーザーが『コマンド』と言った場合は、実行コマンド案を1行で提案する（危険な操作は提案しない）。"
    )

    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
    }

    r = requests.post(url, json=payload, timeout=120)
    if r.status_code == 404:
        # fallback: generate
        url2 = f"{OLLAMA_HOST}/api/generate"
        payload2 = {
            "model": OLLAMA_MODEL,
            "prompt": system + "\n\n" + user_text,
            "stream": False,
        }
        r2 = requests.post(url2, json=payload2, timeout=120)
        r2.raise_for_status()
        data2 = r2.json()
        return normalize_text(data2.get("response", ""))

    r.raise_for_status()
    data = r.json()
    # {"message":{"content":...}} 形式を想定
    msg = data.get("message", {}) or {}
    return normalize_text(msg.get("content", ""))

def speak_open_jtalk(text: str):
    if not text:
        return
    wav = "/tmp/ojt.wav"
    # open_jtalk は stdin からテキストを読む
    cmd = [
        OJT_BIN,
        "-m", OJT_VOICE_PATH,
        "-x", OJT_DIC_PATH,
        "-ow", wav,
    ]
    run(cmd, inp=text.encode("utf-8"), check=True)
    # 再生
    run(["aplay", "-D", AUDIO_OUT, wav], check=False)

def maybe_command_mode(user_text: str) -> bool:
    t = user_text.strip()
    return t.startswith("コマンド") or t.startswith("command") or t.startswith("cmd")

def propose_command(user_text: str) -> str:
    # まずはollamaに「1行コマンド案だけ」返させる
    system = (
        "あなたはLinuxコマンド提案アシスタント。"
        "ユーザー要望に対し、危険な操作（rm -rf 等）や破壊的変更は提案しない。"
        "提案はコマンド1行のみ。説明文は禁止。"
    )
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code == 404:
        url2 = f"{OLLAMA_HOST}/api/generate"
        payload2 = {"model": OLLAMA_MODEL, "prompt": system + "\n\n" + user_text, "stream": False}
        r2 = requests.post(url2, json=payload2, timeout=120)
        r2.raise_for_status()
        return normalize_text(r2.json().get("response", ""))
    r.raise_for_status()
    return normalize_text((r.json().get("message", {}) or {}).get("content", ""))

def exec_command_interactive(cmdline: str):
    cmdline = cmdline.strip()
    if not cmdline:
        return
    print(f"PROPOSED: {cmdline}")
    if not ALLOW_COMMANDS:
        print("SKIP: ALLOW_COMMANDS=0 (execute disabled)")
        return
    ans = input("EXECUTE? type 'yes' to run: ").strip()
    if ans != "yes":
        print("CANCEL")
        return
    # シェル解釈は危険なので、bash -lc で1行だけ（ログは表示）
    p = subprocess.run(["bash", "-lc", cmdline])
    print(f"EXIT={p.returncode}")

# =========================
# Main loop
# =========================

STOP = False
def handle_sigint(sig, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# Resolve paths
WHISPER_BIN_PATH = detect_whisper_bin()
WHISPER_MODEL_PATH = detect_whisper_model()
OJT_VOICE_PATH = detect_open_jtalk_voice()
OJT_DIC_PATH = detect_open_jtalk_dic()

def sanity():
    # if: 必須コマンドがあるか
    for name in ["arecord", "aplay", OJT_BIN]:
        if not shutil.which(name):
            die(f"required command not found in PATH: {name}")
    if not Path(WHISPER_BIN_PATH).exists():
        die(f"WHISPER_BIN not found: {WHISPER_BIN_PATH}")
    if not Path(WHISPER_MODEL_PATH).exists():
        die(f"WHISPER_MODEL not found: {WHISPER_MODEL_PATH}")
    if not Path(OJT_VOICE_PATH).exists():
        die(f"OJT_VOICE not found: {OJT_VOICE_PATH}")
    if not Path(OJT_DIC_PATH).is_dir():
        die(f"OJT_DIC not dir: {OJT_DIC_PATH}")

def main():
    sanity()
    print("OK: voice assistant starting")
    print(f"  whisper: {WHISPER_BIN_PATH}")
    print(f"  model  : {WHISPER_MODEL_PATH}")
    print(f"  ojtvoice: {OJT_VOICE_PATH}")
    print(f"  ojtdic  : {OJT_DIC_PATH}")
    print(f"  audio out: {AUDIO_OUT}")
    print(f"  ollama: {OLLAMA_HOST} model={OLLAMA_MODEL}")
    print("CTRL+C to stop")

    tmp_wake = "/tmp/wake.wav"
    tmp_q = "/tmp/query.wav"

    while not STOP:
        try:
            # wake 判定
            record_wav(tmp_wake, WAKE_SEC)
            t = whisper_transcribe(tmp_wake)
            if t:
                # print(f"DEBUG wake: {t}")
                pass
            if not contains_wake(t):
                continue

            speak_open_jtalk("はい。")

            # 質問を録音して認識
            record_wav(tmp_q, QUERY_SEC)
            q = whisper_transcribe(tmp_q)
            q = normalize_text(q)
            if not q:
                speak_open_jtalk("聞き取れませんでした。")
                continue

            # wake word 自体が入ってたら除去
            for w in WAKE_WORDS:
                q = q.replace(w, "").strip()

            if not q:
                speak_open_jtalk("どうしましたか。")
                continue

            print(f"USER: {q}")

            # コマンドモード（将来想定）
            if maybe_command_mode(q):
                cmd = propose_command(q)
                print(f"CMD: {cmd}")
                speak_open_jtalk("コマンド案を表示しました。")
                exec_command_interactive(cmd)
                continue

            # 通常会話
            ans = ollama_chat(q)
            print(f"ASSIST: {ans}")
            speak_open_jtalk(ans)

        except Exception as e:
            print(f"ERR: {e}")
            time.sleep(0.5)

    print("BYE")

if __name__ == "__main__":
    main()

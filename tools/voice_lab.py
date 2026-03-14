#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

# ----------------------------
# Utils
# ----------------------------
def run(cmd, *, input_text=None, check=False, capture=True, env=None):
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    p = subprocess.run(
        cmd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
        shell=shell,
        env=env,
    )
    out = (p.stdout.decode("utf-8", errors="replace") if capture and p.stdout is not None else "")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed (exit={p.returncode}): {cmd}\n{out}")
    return p.returncode, out

def which(name):
    return shutil.which(name)

def exists(path):
    return Path(path).exists()

def say(s):
    print(s, flush=True)

def ask(prompt, default=None):
    if default is None:
        s = input(prompt).strip()
        return s
    s = input(f"{prompt} [{default}] ").strip()
    return s if s else default

def if_cmd(cmd):
    p = which(cmd)
    if not p:
        say(f"NG: command not found: {cmd}")
        return False
    say(f"OK: {cmd} -> {p}")
    return True

def header(title):
    say("")
    say("=" * 60)
    say(title)
    say("=" * 60)

# ----------------------------
# Paths / defaults
# ----------------------------
ROOT = Path.cwd()

# whisper.cpp (default guess)
WHISPER_BIN_DEFAULT = os.environ.get("WHISPER_BIN", "").strip() or str(ROOT / "whisper.cpp" / "build" / "bin" / "whisper-cli")
WHISPER_MODEL_DEFAULT = os.environ.get("WHISPER_MODEL", "").strip() or str(ROOT / "whisper.cpp" / "models" / "ggml-base.bin")

# open_jtalk (voice guess by searching)
VOICE_HINT = os.environ.get("OJT_VOICE", "").strip()
DIC_HINT = os.environ.get("OJT_DIC", "").strip() or "/var/lib/mecab/dic/open-jtalk/naist-jdic"

# audio in/out
AUDIO_IN_DEFAULT = os.environ.get("AUDIO_IN", "default").strip()
AUDIO_OUT_DEFAULT = os.environ.get("AUDIO_OUT", "default").strip()

TMP = Path("/tmp/voice_lab")
TMP.mkdir(parents=True, exist_ok=True)

def find_htsvoice():
    # env優先
    if VOICE_HINT and exists(VOICE_HINT):
        return VOICE_HINT
    # typical dirs
    candidates = [
        Path("/usr/share/hts-voice"),
        Path("/usr/share"),
        Path("/usr/local/share"),
    ]
    hits = []
    for base in candidates:
        if base.exists():
            hits.extend(base.rglob("*.htsvoice"))
    if not hits:
        return ""
    # prefer "nitech"
    hits.sort(key=lambda x: ("nitech" not in x.name.lower(), len(str(x))))
    return str(hits[0])

def show_versions():
    header("1) バージョン/環境チェック")
    say(f"cwd: {ROOT}")
    say(f"time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    run("uname -a", capture=False)
    say("")
    # commands
    for c in ["uv", "python", "arecord", "aplay", "sox", "open_jtalk", "curl"]:
        if_cmd(c)
    # open_jtalk voice/dic
    voice = find_htsvoice()
    say("")
    say(f"OJT_VOICE (auto): {voice or 'NG (not found)'}")
    say(f"OJT_DIC (default): {DIC_HINT} {'OK' if exists(DIC_HINT) else 'NG'}")
    say("")
    # whisper
    say(f"WHISPER_BIN (default): {WHISPER_BIN_DEFAULT} {'OK' if exists(WHISPER_BIN_DEFAULT) else 'NG'}")
    say(f"WHISPER_MODEL(default): {WHISPER_MODEL_DEFAULT} {'OK' if exists(WHISPER_MODEL_DEFAULT) else 'NG'}")
    say("")
    # ollama
    code, _ = run("curl -sS http://127.0.0.1:11434/api/tags >/dev/null")
    say("ollama: " + ("OK" if code == 0 else "NG"))

def list_audio_devices():
    header("2) 音声デバイス一覧（対話型）")
    run("arecord -l", capture=False)
    say("")
    run("aplay -l", capture=False)

def record_wav():
    header("3) 10秒録音（AUDIO_IN を選択）")
    audio_in = ask("AUDIO_IN を指定 (例: default / hw:2,0 / plughw:2,0)", AUDIO_IN_DEFAULT)
    sec = int(ask("録音秒数", "10"))
    wav = TMP / f"rec{sec}.wav"

    # if: arecord
    if not if_cmd("arecord"):
        return

    say(f"INFO: 録音開始: {sec}s / device={audio_in}")
    # 16kHz mono S16_LE (whisper向け)
    cmd = ["arecord", "-D", audio_in, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(sec), str(wav)]
    rc, out = run(cmd, capture=True)
    if rc != 0:
        say("NG: record failed")
        say(out)
        return
    say(f"OK: RECORDED {wav} ({wav.stat().st_size} bytes)")

    # sox stat
    if if_cmd("sox"):
        say("----- SOX STAT -----")
        _, soxout = run(["sox", str(wav), "-n", "stat"], capture=True)
        # 重要行だけ
        for line in soxout.splitlines():
            if any(k in line for k in ["Length (seconds)", "Sample Rate", "RMS     amplitude", "Maximum amplitude", "Mean    amplitude"]):
                say(line)

    # playback
    if if_cmd("aplay"):
        say("INFO: 再生します（聞こえるか確認）")
        run(["aplay", str(wav)], capture=False)

def whisper_transcribe():
    header("4) Whisper 文字起こし（直近録音ファイルを選択）")
    # if: whisper bin/model
    wb = ask("WHISPER_BIN", WHISPER_BIN_DEFAULT)
    wm = ask("WHISPER_MODEL", WHISPER_MODEL_DEFAULT)
    if not exists(wb):
        say(f"NG: whisper bin not found: {wb}")
        return
    if not exists(wm):
        say(f"NG: whisper model not found: {wm}")
        return

    # list wavs
    wavs = sorted(TMP.glob("rec*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        say("NG: 録音ファイルがありません。まず 3) で録音して。")
        return
    say("録音候補:")
    for i, w in enumerate(wavs[:10], 1):
        say(f"  {i}) {w} ({w.stat().st_size} bytes)")
    idx = int(ask("どれを使う？", "1"))
    wav = wavs[idx-1]

    lang = ask("言語 (ja/en/auto)", "ja")
    sec = ask("オプション: beam/best/temperature (例: 10 10 0.0)", "10 10 0.0")
    beam, best, temp = sec.split()

    log = TMP / f"whisper_{wav.stem}.log"
    cmd = [
        wb, "-m", wm, "-f", str(wav),
        "--no-timestamps",
        "--beam-size", str(beam),
        "--best-of", str(best),
        "--temperature", str(temp),
    ]
    if lang != "auto":
        cmd += ["-l", lang]

    say("INFO: 文字起こし中...")
    rc, out = run(cmd, capture=True)
    log.write_text(out, encoding="utf-8")
    if rc != 0:
        say(f"NG: whisper failed (exit={rc}) log={log}")
        # 重要そうな行だけ
        for line in out.splitlines()[-80:]:
            if any(k in line.lower() for k in ["error", "failed", "abort", "invalid"]):
                say(line)
        return

    # 結果のそれっぽい行だけ表示（日本語/英字行）
    say("----- RESULT (抜粋) -----")
    shown = 0
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        # whisper-cli は進捗行も多いので、全文らしい行だけ拾う
        if any(ch in s for ch in "ぁあいうえおアイウエオ一二三四五六七八九十") or (s and s[0].isalpha()):
            if "whisper_print_timings" in s or "main:" in s:
                continue
            say(s)
            shown += 1
            if shown >= 30:
                break
    say(f"OK: log={log}")

def openjtalk_tts():
    header("5) OpenJTalk 読み上げ（wav生成→再生）")
    if not if_cmd("open_jtalk"):
        say("NG: open_jtalk が無い。 apt で入れる必要あり。")
        return
    if not if_cmd("aplay"):
        return

    voice = ask("HTS voice path (空なら自動検索)", find_htsvoice() or "")
    if not voice or not exists(voice):
        say(f"NG: VOICE not found: {voice}")
        return
    dic = ask("辞書パス (-x)", DIC_HINT)
    if not exists(dic):
        say(f"NG: DIC not found: {dic}")
        return

    text = ask("読み上げテキスト", "文章を読み上げればいいんじゃないの？とりあえず")
    wav = TMP / "ojt.wav"

    cmd = ["open_jtalk", "-m", voice, "-x", dic, "-ow", str(wav)]
    rc, out = run(cmd, input_text=text, capture=True)
    if rc != 0:
        say("NG: open_jtalk failed")
        say(out)
        return

    say(f"OK: WAV={wav} ({wav.stat().st_size} bytes)")
    run(["aplay", str(wav)], capture=False)

def menu():
    say("")
    say("0) 終了")
    say("1) バージョン/環境チェック")
    say("2) 音声デバイス一覧（arecord/aplay -l）")
    say("3) 10秒録音→sox stat→再生（対話型）")
    say("4) Whisper 文字起こし（対話型）")
    say("5) OpenJTalk 読み上げ（対話型）")

def main():
    while True:
        menu()
        sel = ask("番号を選んで", "1")
        if sel == "0":
            return
        try:
            if sel == "1":
                show_versions()
            elif sel == "2":
                list_audio_devices()
            elif sel == "3":
                record_wav()
            elif sel == "4":
                whisper_transcribe()
            elif sel == "5":
                openjtalk_tts()
            else:
                say("NG: その番号は無い")
        except Exception as e:
            say(f"NG: {e}")

if __name__ == "__main__":
    main()

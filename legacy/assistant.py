#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from vosk import Model, KaldiRecognizer


def eprint(*args):
    print(*args, file=sys.stderr)


def find_hts_voice() -> Path:
    # よくある配置を総当たり（パッケージ差異に強くする）
    candidates = [
        Path("/usr/share/hts-voice"),
        Path("/usr/share/hts-voice/hts-voice-nitech-jp-atr503-m001"),
        Path("/usr/share/hts-voice/hts_voice_nitech_jp_atr503_m001.htsvoice"),
    ]
    # 直接ファイル
    for p in candidates:
        if p.is_file() and p.suffix == ".htsvoice":
            return p
    # ディレクトリ探索
    for base in candidates:
        if base.is_dir():
            hits = list(base.rglob("*.htsvoice"))
            if hits:
                # それっぽい名前を優先
                hits.sort(key=lambda x: ("nitech" not in x.name.lower(), len(str(x))))
                return hits[0]
    raise FileNotFoundError("HTS voice (*.htsvoice) not found under /usr/share/hts-voice")


def find_openjtalk_dict() -> Path:
    base = Path("/var/lib/mecab/dic/open-jtalk/naist-jdic")
    if base.is_dir():
        return base
    # 念のため探索
    root = Path("/var/lib/mecab/dic/open-jtalk")
    if root.is_dir():
        for p in root.rglob("naist-jdic"):
            if p.is_dir():
                return p
    raise FileNotFoundError("open_jtalk dictionary (naist-jdic) not found under /var/lib/mecab/dic/open-jtalk")


def run_openjtalk(text: str, out_wav: Path, voice: Path, dic: Path):
    # open_jtalkはstdin対応してないので一時テキストファイル
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
        tf.write(text)
        tf.flush()
        txt_path = tf.name

    try:
        cmd = [
            "open_jtalk",
            "-m", str(voice),
            "-x", str(dic),
            "-ow", str(out_wav),
            txt_path,
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"open_jtalk failed: rc={r.returncode}\n{r.stderr.strip()}")
    finally:
        try:
            os.unlink(txt_path)
        except OSError:
            pass


def aplay_wav(path: Path, device: str | None):
    cmd = ["aplay"]
    if device:
        cmd += ["-D", device]
    cmd += [str(path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ollama_generate(host: str, model: str, prompt: str, system: str, timeout_sec: int = 120) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vosk_model", default=str(Path("~/voicechat/vosk-model-ja").expanduser()))
    ap.add_argument("--wake", default="べりべり")
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--ollama_model", default="llama3.2")
    ap.add_argument("--aplay_device", default=None, help="例: plughw:3,0 / default")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # if: 前提ファイル
    vosk_path = Path(args.vosk_model)
    if not vosk_path.is_dir():
        raise RuntimeError(f"Vosk model dir not found: {vosk_path}")

    voice = find_hts_voice()
    dic = find_openjtalk_dict()

    # if: ollama到達確認（落ちてるなら喋らずに終了）
    try:
        _ = requests.get(args.ollama_host.rstrip("/") + "/api/version", timeout=2)
    except Exception as ex:
        raise RuntimeError(f"ollama not reachable: {args.ollama_host} ({ex})")

    if args.debug:
        eprint(f"VOSK={vosk_path}")
        eprint(f"VOICE={voice}")
        eprint(f"DIC={dic}")
        eprint(f"APLAY_DEVICE={args.aplay_device}")
        eprint(f"OLLAMA={args.ollama_host} model={args.ollama_model}")
        eprint(f"WAKE={args.wake}")

    model = Model(str(vosk_path))
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(False)

    # arecord を raw で流して STT
    arecord_cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"]
    p = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def say(text: str):
        out = Path("/tmp/ojt_tts.wav")
        run_openjtalk(text, out, voice, dic)
        aplay_wav(out, args.aplay_device)

    system_prompt = (
        "あなたは日本語の音声アシスタント。回答は短く、結論から。"
        "ユーザーが『コマンド』と言った場合は、実行手順ではなく command 文字列をJSONで返す。"
        "それ以外は通常回答。"
        "JSON出力が必要な場合は必ず以下形式: {\"type\":\"command\",\"command\":\"...\"} または {\"type\":\"chat\",\"answer\":\"...\"}"
    )

    say("起動しました。")

    mode = "idle"  # idle -> armed
    armed_start = 0.0
    captured = []

    try:
        while True:
            chunk = p.stdout.read(4000)
            if not chunk:
                raise RuntimeError("arecord stream ended")

            if rec.AcceptWaveform(chunk):
                j = json.loads(rec.Result())
                text = normalize_text(j.get("text", ""))
                if args.debug and text:
                    eprint(f"[final] {text}")

                if not text:
                    continue

                if mode == "idle":
                    if args.wake in text:
                        mode = "armed"
                        armed_start = time.time()
                        captured = []
                        say("はい。")
                        continue

                elif mode == "armed":
                    # ウェイク後、次の発話を1回確定で拾う（最大10秒）
                    # べりべり自体が混ざるのは除外
                    if args.wake in text:
                        continue
                    captured.append(text)
                    merged = normalize_text(" ".join(captured))
                    if merged:
                        # ollamaへ
                        prompt = merged

                        # コマンド想定：ユーザーが「コマンド」と言ったらJSONを要求
                        if "コマンド" in merged:
                            prompt = (
                                "次の要求から、実行コマンド1行だけを生成してJSONで返して。\n"
                                f"要求: {merged}"
                            )

                        ans = ollama_generate(args.ollama_host, args.ollama_model, prompt, system_prompt)
                        if not ans:
                            say("すみません、回答できませんでした。")
                        else:
                            # JSONっぽい場合は読み上げ用に整形
                            if ans.lstrip().startswith("{"):
                                try:
                                    obj = json.loads(ans)
                                    if obj.get("type") == "command":
                                        # 実行はしない（将来拡張ポイント）
                                        cmd = obj.get("command", "")
                                        say(f"コマンド候補です。{cmd}")
                                    else:
                                        say(obj.get("answer", ans))
                                except Exception:
                                    say(ans)
                            else:
                                say(ans)

                    mode = "idle"
                    captured = []

            else:
                # partial はログだけ（誤爆防止で判定しない）
                if args.debug:
                    pj = json.loads(rec.PartialResult())
                    pt = normalize_text(pj.get("partial", ""))
                    if pt:
                        eprint(f"[partial] {pt}")

            # armed timeout
            if mode == "armed" and (time.time() - armed_start) > 10:
                say("聞き取れませんでした。")
                mode = "idle"
                captured = []

    finally:
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()

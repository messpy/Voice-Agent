#!/usr/bin/env python3
"""
STT 完全自動ベンチマーク — 全パターン網羅 + リソース監視 + 結果保存

使い方:
  uv run python tools/stt_full_benchmark.py
  uv run python tools/stt_full_benchmark.py --audio /path/to/audio.mp3 --ref /path/to/lyrics.txt
  uv run python tools/stt_full_benchmark.py --quick  # 代表パターンのみ
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\u3000", " ")).strip()


def compact(text: str) -> str:
    return normalize(text).replace(" ", "")


def char_ngrams(text: str, *, min_n: int = 2, max_n: int = 3) -> set[str]:
    c = compact(text)
    return set(c[i:i+n] for n in range(min_n, max_n + 1) for i in range(len(c) - n + 1))


def similarity(q: str, a: str) -> float:
    qg, ag = char_ngrams(q), char_ngrams(a)
    if not qg or not ag:
        return 0.0
    ov = len(qg & ag)
    if ov == 0:
        return 0.0
    return round((ov / max(1, len(qg)) * 0.8) + (ov / max(1, len(ag)) * 0.2), 4)


# ── リソース監視 ─────────────────────────────────────────────

_monitor_data: dict = {"cpu": [], "mem": [], "gpu": []}
_monitor_stop = threading.Event()


def _monitor_loop():
    while not _monitor_stop.is_set():
        try:
            cpu = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
            if cpu:
                _monitor_data["cpu"].append(float(cpu))
        except:
            pass
        try:
            mem = os.popen("free -m | grep Mem | awk '{print $3}'").read().strip()
            if mem:
                _monitor_data["mem"].append(int(mem))
        except:
            pass
        try:
            gpu = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null").read().strip()
            if gpu:
                _monitor_data["gpu"].append(float(gpu.replace("%", "")))
        except:
            pass
        time.sleep(1)


def start_monitor():
    global _monitor_data
    _monitor_data = {"cpu": [], "mem": [], "gpu": []}
    _monitor_stop.clear()
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    return t


def stop_monitor(mon_thread: threading.Thread):
    time.sleep(1)
    _monitor_stop.set()
    mon_thread.join(timeout=2)
    d = _monitor_data
    return {
        "avg_cpu": sum(d["cpu"]) / len(d["cpu"]) if d["cpu"] else 0,
        "peak_cpu": max(d["cpu"]) if d["cpu"] else 0,
        "avg_mem": sum(d["mem"]) / len(d["mem"]) if d["mem"] else 0,
        "peak_mem": max(d["mem"]) if d["mem"] else 0,
        "avg_gpu": sum(d["gpu"]) / len(d["gpu"]) if d["gpu"] else 0,
        "has_gpu": len(d["gpu"]) > 0,
    }


# ── 前処理 ──────────────────────────────────────────────────────

def preprocess_audio(wav_in: Path, wav_out: Path, mode: str) -> None:
    if mode == "none":
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-ar", "16000", "-ac", "1",
                        "-sample_fmt", "s16", str(wav_out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif mode == "vocal_eq":
        af = "highpass=f=150,lowpass=f=8000,compand=attacks=0.02:decays=0.05:points=-70/-70|-40/-40|-20/-10|0/-3|20/-3:soft-knee=6:gain=6"
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-af", af, "-ar", "16000",
                        "-ac", "1", "-sample_fmt", "s16", str(wav_out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Whisper.cpp 実行 ─────────────────────────────────────────────

def run_whisper_cpp(wav: Path, model: str, threads: int, prompt: str, beam: int = 5) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / "whisper.cpp" / "build" / "src") + ":" + str(ROOT / "whisper.cpp" / "build" / "ggml" / "src")

    model_map = {
        "tiny": ".runtime/models/ggml-tiny.bin",
        "base": ".runtime/models/ggml-base.bin",
        "small": ".runtime/models/ggml-small.bin",
    }
    model_path = ROOT / model_map[model]
    if not model_path.exists():
        return {"error": f"モデル未: {model_path}"}

    cmd = [
        str(ROOT / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        "-m", str(model_path), "-f", str(wav), "-l", "ja", "-t", str(threads),
        "--no-timestamps", "--beam-size", str(beam), "--temperature", "0.0",
    ]
    if prompt:
        cmd += ["--prompt", prompt]

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    elapsed = time.time() - t0

    combined = p.stdout + "\n" + p.stderr
    combined = combined.split("whisper_print_timings:", 1)[0]
    lines = []
    for line in combined.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(k in s.lower() for k in ["whisper_", "system_info:", "main:", "n_threads",
                                         "whisper_init", "whisper_model_load", "whisper_backend",
                                         "whisper_init_state", "compute buffer", "kv ", "adding ",
                                         "loading model", "model size"]):
            continue
        s = re.sub(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", s)
        if re.search(r"[ぁ-んァ-ン一-龯A-Za-z0-9]", s):
            lines.append(s)

    text = normalize(" ".join(lines))
    return {"text": text, "elapsed": round(elapsed, 2), "chars": len(compact(text))}


# ── faster-whisper 実行 ────────────────────────────────────────

def run_faster_whisper(wav: Path, model_name: str, device: str, compute_type: str, vad: bool) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "faster-whisper未インストール"}

    fw_root = ROOT / ".runtime" / "faster_whisper_models"
    fw_root.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type,
                            download_root=str(fw_root))
    except Exception as e:
        return {"error": f"モデルロード失敗: {e}"}
    load_time = time.time() - t_load

    kwargs = dict(language="ja", beam_size=5, temperature=0.0)
    if vad:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)

    t0 = time.time()
    try:
        segments, _ = model.transcribe(str(wav), **kwargs)
        seg_list = list(segments)
    except Exception as e:
        return {"error": f"認識失敗: {e}"}
    elapsed = time.time() - t0

    text = normalize(" ".join(s.text for s in seg_list).strip())
    return {"text": text, "elapsed": round(elapsed, 2), "chars": len(compact(text)),
            "load_time": round(load_time, 1)}


# ── プロンプト定義 ───────────────────────────────────────────────

PROMPTS = {
    "none": "",
    "keyword": "♪音楽♪ 帝京平成大学 帝京魂 グランドライン 大大学時代 キャンパス 合格者数 実学教育 健康 医療 スポーツ 経営学 ナミ 航海士 トナカイ バケモノ 仲間 創立30周年 新型コロナウイルス めざましじゃんけん 医者の俺 男たちは夢を追い続け",
    "lyrics_head": "男たちはグランドラインを目指し 夢を追い続け 世はまさに大大学時代 キャンパキャンパスデスデス 全国合格数みんな 帝京平成大学を知ってるか 実学の総合大学 帝京平成大学 帝京魂 東京 千葉にある4つのキャンパス 健康 医療 スポーツ 経営学など 幅広い学問を学べるんだぞ 私は航海士のナミ 帝京平成大学はここもすごいのよ 徹底した実学教育 全国有数の合格者数",
}


# ── ベンチマークパターン ───────────────────────────────────────

def build_patterns(quick: bool = False) -> list[dict]:
    """全テストパターンを生成"""
    patterns = []

    # whisper.cpp パターン
    for model in ["small", "base", "tiny"]:
        for prompt_key in ["none", "keyword"]:
            for preprocess in ["none"]:
                threads = 4
                patterns.append({
                    "type": "whisper.cpp",
                    "id": f"wcpp_{model}_{prompt_key}_{preprocess}",
                    "model": model,
                    "prompt_key": prompt_key,
                    "preprocess": preprocess,
                    "threads": threads,
                })

    if not quick:
        for model in ["small"]:
            for prompt_key in ["lyrics_head"]:
                patterns.append({
                    "type": "whisper.cpp",
                    "id": f"wcpp_{model}_{prompt_key}_none",
                    "model": model,
                    "prompt_key": prompt_key,
                    "preprocess": "none",
                    "threads": 4,
                })

    # faster-whisper パターン
    fw_models = [
        ("small", "cpu", "int8"),
    ]
    if not quick:
        fw_models += [
            ("kotoba-tech/kotoba-whisper-v1.0", "cpu", "int8"),
            ("base", "cpu", "int8"),
        ]

    for model_name, device, compute_type in fw_models:
        for vad in [True, False]:
            fw_id = f"fw_{model_name.replace('/', '_')}_vad{'1' if vad else '0'}"
            patterns.append({
                "type": "faster-whisper",
                "id": fw_id,
                "model_name": model_name,
                "device": device,
                "compute_type": compute_type,
                "vad": vad,
            })

    return patterns


# ── 実行 ─────────────────────────────────────────────────────────

def run_single(pattern: dict, wav: Path, lyrics: str) -> dict:
    print(f"\n  ▶ {pattern['id']}")

    # 前処理
    pre_wav = wav
    if pattern.get("preprocess", "none") != "none":
        pre_wav = wav.with_name(wav.stem + f"_{pattern['preprocess']}.wav")
        preprocess_audio(wav, pre_wav, pattern["preprocess"])

    # リソース監視開始
    mon = start_monitor()

    # 実行
    if pattern["type"] == "whisper.cpp":
        result = run_whisper_cpp(
            wav=pre_wav,
            model=pattern["model"],
            threads=pattern["threads"],
            prompt=PROMPTS.get(pattern["prompt_key"], ""),
        )
    elif pattern["type"] == "faster-whisper":
        result = run_faster_whisper(
            wav=pre_wav,
            model_name=pattern["model_name"],
            device=pattern["device"],
            compute_type=pattern["compute_type"],
            vad=pattern["vad"],
        )
    else:
        result = {"error": "不明なタイプ"}

    # リソース集計
    resources = stop_monitor(mon)

    # 評価
    text = result.get("text", "")
    sim = similarity(lyrics, text) if text else 0.0

    return {
        "id": pattern["id"],
        "type": pattern["type"],
        "status": "ok" if "text" in result else "error",
        "elapsed": result.get("elapsed", 0),
        "load_time": result.get("load_time", 0),
        "chars": result.get("chars", 0),
        "similarity": sim,
        "text_preview": text[:120] if text else "",
        "text_full": text,
        "resources": resources,
        "error": result.get("error", ""),
        "config": {k: v for k, v in pattern.items() if k != "type"},
    }


def main():
    parser = argparse.ArgumentParser(description="STT 完全ベンチマーク")
    parser.add_argument("--audio", type=Path, default=Path("/home/kennypi/data/audio/music/テイキョウヘイセイダイガク.mp3"))
    parser.add_argument("--ref", type=Path, default=Path("/home/kennypi/data/audio/music/teikyoheiseidaigaku.txt"))
    parser.add_argument("--quick", action="store_true", help="代表パターンのみ")
    args = parser.parse_args()

    wav_in = args.audio.expanduser().resolve()
    ref_file = args.ref.expanduser().resolve()

    if not wav_in.exists():
        raise SystemExit(f"NG: 音声ファイルなし {wav_in}")
    if not ref_file.exists():
        raise SystemExit(f"NG: 参照ファイルなし {ref_file}")

    lyrics = normalize(ref_file.read_text(encoding="utf-8"))
    patterns = build_patterns(quick=args.quick)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if args.quick else "full"
    out_file = RESULTS_DIR / f"stt_bench_{mode}_{tag}.json"

    print("=" * 72)
    print(f"  STT 完全ベンチマーク ({mode} mode)")
    print("=" * 72)
    print(f"  音声: {wav_in.name}")
    print(f"  参照: {ref_file.name} ({len(compact(lyrics))}文字)")
    print(f"  パターン数: {len(patterns)}")
    print(f"  結果: {out_file}")

    results = []
    t_all = time.time()

    for i, pat in enumerate(patterns, 1):
        print(f"\n[{i}/{len(patterns)}] {pat['id']}")
        r = run_single(pat, wav_in, lyrics)
        results.append(r)
        status_icon = "✅" if r["status"] == "ok" else "❌"
        print(f"  {status_icon} {r['elapsed']:.1f}s | {r['chars']}文字 | 類似度 {r['similarity']:.4f}")
        if r.get("error"):
            print(f"  エラー: {r['error']}")

    total = time.time() - t_all

    # ── 結果テーブル ──
    print("\n" + "=" * 72)
    print("  結果まとめ")
    print("=" * 72)
    header = f"{'ID':45s} {'時間(s)':>8s} {'一致率':>8s} {'文字数':>6s} {'CPU%':>6s} {'Mem':>6s}"
    print(header)
    print("-" * 72)

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["similarity"], reverse=True)

    for r in ok_results:
        res = r["resources"]
        print(f"{r['id']:45s} {r['elapsed']:8.1f} {r['similarity']:8.4f} {r['chars']:6d} "
              f"{res['avg_cpu']:5.1f}% {res['peak_mem']:5.0f}MB")

    print("=" * 72)
    print(f"  総時間: {total:.0f}秒")

    # 最高精度
    if ok_results:
        best = ok_results[0]
        print(f"\n🏆 最高一致率: {best['id']} — 類似度 {best['similarity']:.4f} / {best['elapsed']:.1f}s")

    # 高速かつ高一致
    fast_good = [r for r in ok_results if r["similarity"] >= 0.35]
    if fast_good:
        fast_good.sort(key=lambda r: r["elapsed"])
        best_fast = fast_good[0]
        print(f"⚡ 高速かつ高一致: {best_fast['id']} — 類似度 {best_fast['similarity']:.4f} / {best_fast['elapsed']:.1f}s")

    # JSON保存
    save_data = {
        "date": datetime.now().isoformat(),
        "mode": mode,
        "audio": str(wav_in),
        "reference": str(ref_file),
        "reference_chars": len(compact(lyrics)),
        "total_time": round(total, 1),
        "results": [
            {
                "id": r["id"],
                "type": r["type"],
                "status": r["status"],
                "elapsed": r["elapsed"],
                "load_time": r.get("load_time", 0),
                "chars": r["chars"],
                "similarity": r["similarity"],
                "text_preview": r["text_preview"],
                "text_full": r["text_full"],
                "resources": r["resources"],
                "error": r.get("error", ""),
                "config": r["config"],
            }
            for r in results
        ],
    }

    out_file.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 保存: {out_file}")


if __name__ == "__main__":
    main()

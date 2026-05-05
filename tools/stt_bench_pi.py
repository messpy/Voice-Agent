#!/usr/bin/env python3
"""
whisper.cpp ベンチマーク + ラズパイ温度監視 + 複数音声対応
"""

import subprocess, time, re, os, json, threading, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMP_WARNING = 70
TEMP_CRITICAL = 80
COOLDOWN_MIN = 60
COOLDOWN_MAX = 180

PROMPTS = {
    "none": "",
    "keyword": "帝京平成大学 帝京魂 グランドライン キャンパス ナミ",
    "keyword_full": "帝京平成大学 帝京魂 グランドライン キャンパス ナミ ウォール街 自由の女神 リバティ島 エリス島 ハドソン川 エンパイアステートビル ダンボ地区 ワンワールドトレードセンター",
    "short": "帝京平成大学",
}


def normalize(text):
    return re.sub(r"\s+", " ", str(text).replace("\u3000", " ")).strip()


def compact(text):
    return normalize(text).replace(" ", "")


def char_ngrams(text, min_n=2, max_n=3):
    c = compact(text)
    return set(
        c[i : i + n] for n in range(min_n, max_n + 1) for i in range(len(c) - n + 1)
    )


def similarity(q, a):
    qg, ag = char_ngrams(q), char_ngrams(a)
    if not qg or not ag:
        return 0.0
    ov = len(qg & ag)
    if ov == 0:
        return 0.0
    return round((ov / max(1, len(qg)) * 0.8) + (ov / max(1, len(ag)) * 0.2), 4)


def get_system_info():
    info = {
        "cpu_temp": None,
        "gpu_temp": None,
        "cpu_usage": 0,
        "mem_used_mb": 0,
        "mem_total_mb": 0,
    }

    for path in [
        "/sys/class/thermal/thermal_zone0/temp",
        "/opt/vc/bin/vcgencmd measure_temp",
    ]:
        try:
            if "thermal_zone" in path:
                with open(path) as f:
                    info["cpu_temp"] = int(f.read().strip()) / 1000
            else:
                out = subprocess.run(path, capture_output=True, text=True).stdout
                m = re.search(r"(\d+\.?\d*)", out)
                if m:
                    info["cpu_temp"] = float(m.group(1))
            break
        except:
            pass

    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True
        ).stdout
        m = re.search(r"(\d+\.?\d*)", out)
        if m:
            info["gpu_temp"] = float(m.group(1))
    except:
        pass

    try:
        c = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
        if c:
            info["cpu_usage"] = float(c)
    except:
        pass

    try:
        out = os.popen("free -m").read()
        m = re.search(r"Mem:\s+(\d+)\s+(\d+)", out)
        if m:
            info["mem_total_mb"] = int(m.group(1))
            info["mem_used_mb"] = int(m.group(2))
    except:
        pass

    return info


def check_thermal_throttle():
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True
        ).stdout
        return "0x" not in out or out.strip() != "throttled=0x0"
    except:
        return False


def print_system_status(prefix="", info=None):
    if info is None:
        info = get_system_info()
    temp = info["cpu_temp"]
    throttled = " [THROTTLED]" if check_thermal_throttle() else ""
    temp_str = f"{temp:.1f}°C" if temp else "N/A"
    if info.get("gpu_temp") and info["gpu_temp"] != temp:
        temp_str += f" (GPU: {info['gpu_temp']:.1f}°C)"
    mem_pct = (
        info["mem_used_mb"] / info["mem_total_mb"] * 100
        if info["mem_total_mb"] > 0
        else 0
    )
    print(
        f"  {prefix}[{temp_str}{throttled}] CPU: {info['cpu_usage']:.0f}% | RAM: {info['mem_used_mb']}MB ({mem_pct:.0f}%)"
    )


def should_cooldown(info):
    temp = info.get("cpu_temp")
    if temp is None:
        return False
    if temp >= TEMP_CRITICAL:
        return True
    if temp >= TEMP_WARNING and check_thermal_throttle():
        return True
    return False


def wait_for_cooldown(current_temp):
    cooldown_sec = COOLDOWN_MAX if current_temp >= TEMP_CRITICAL else COOLDOWN_MIN
    print(
        f"\n  ⚠️  {'критичная' if current_temp >= TEMP_CRITICAL else '高温'} ({current_temp:.1f}°C) — クールダウン {cooldown_sec}秒"
    )
    for remaining in range(cooldown_sec, 0, -10):
        info = get_system_info()
        print_system_status(f"クールダウン ({remaining}s): ", info)
        if info.get("cpu_temp", 0) < TEMP_WARNING and not check_thermal_throttle():
            print("  ✅ 温度低下")
            return
        time.sleep(10)
    print("  ⏰ クールダウン終了")


_partial_save_lock = threading.Lock()
_partial_save_file = None


def _save_partial(results, results_dir):
    with _partial_save_lock:
        global _partial_save_file
        if _partial_save_file is None:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            _partial_save_file = results_dir / f"stt_bench_pi_{tag}.json"

        results["total_elapsed_sec"] = round(
            time.time() - results.get("_start_time", time.time()), 1
        )
        results["total_elapsed_min"] = round(results["total_elapsed_sec"] / 60, 1)
        _partial_save_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  💾 途中保存: {_partial_save_file.name}")


_mdata = {"cpu": [], "mem": []}
_mstop = threading.Event()


def _monitor():
    while not _mstop.is_set():
        try:
            c = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
            if c:
                _mdata["cpu"].append(float(c))
        except:
            pass
        try:
            m = os.popen("free -m | grep Mem | awk '{print $3}'").read().strip()
            if m:
                _mdata["mem"].append(int(m))
        except:
            pass
        time.sleep(1)


def start_mon():
    global _mdata
    _mdata = {"cpu": [], "mem": []}
    _mstop.clear()
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t


def stop_mon(t):
    time.sleep(1)
    _mstop.set()
    t.join(timeout=2)
    return {
        "avg_cpu": sum(_mdata["cpu"]) / len(_mdata["cpu"]) if _mdata["cpu"] else 0,
        "peak_cpu": max(_mdata["cpu"]) if _mdata["cpu"] else 0,
        "avg_mem": sum(_mdata["mem"]) / len(_mdata["mem"]) if _mdata["mem"] else 0,
        "peak_mem": max(_mdata["mem"]) if _mdata["mem"] else 0,
    }


def run_wcpp(model, wav, prompt="", threads=4, beam=5, timeout=300):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        str(ROOT / "whisper.cpp/build/src")
        + ":"
        + str(ROOT / "whisper.cpp/build/ggml/src")
    )
    model_path = ROOT / f".runtime/models/ggml-{model}.bin"
    if not model_path.exists():
        return {"error": f"モデルなし: {model_path}"}
    cmd = [
        str(ROOT / "whisper.cpp/build/bin/whisper-cli"),
        "-m",
        str(model_path),
        "-f",
        str(wav),
        "-l",
        "ja",
        "-t",
        str(threads),
        "--no-timestamps",
        "--beam-size",
        str(beam),
        "--temperature",
        "0.0",
    ]
    if prompt:
        cmd += ["--prompt", prompt]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"error": f"タイムアウト ({timeout}秒)", "elapsed": timeout}
    elapsed = time.time() - t0
    combined = (p.stdout + "\n" + p.stderr).split("whisper_print_timings:", 1)[0]
    lines = []
    for line in combined.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(
            k in s.lower()
            for k in [
                "whisper_",
                "system_info:",
                "main:",
                "n_threads",
                "whisper_init",
                "whisper_model_load",
                "compute buffer",
                "kv ",
            ]
        ):
            continue
        s = re.sub(
            r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", s
        )
        if re.search(r"[ぁ-んァ-ン一-龯A-Za-z0-9]", s):
            lines.append(s)
    return {"text": normalize(" ".join(lines)), "elapsed": round(elapsed, 2)}


def run_vosk(wav, model_name="vosk-model-small-ja", timeout=300):
    import wave
    import json

    model_path = ROOT / f".runtime/vosk/{model_name}"
    if not model_path.exists():
        return {"error": f"モデルなし: {model_path}"}

    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        return {"error": "vosk未インストール"}

    t0 = time.time()
    try:
        model = Model(str(model_path))
    except Exception as e:
        return {"error": f"モデルロード失敗: {e}"}

    try:
        rec = KaldiRecognizer(model, 16000)
        with wave.open(str(wav), "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
    except Exception as e:
        return {"error": f"認識失敗: {e}"}

    elapsed = time.time() - t0
    return {"text": normalize(text), "elapsed": round(elapsed, 2)}


def run_reazon(wav, timeout=300):
    try:
        from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
    except ImportError:
        return {"error": "reazonspeech未インストール. pip install ReazonSpeech"}

    t0 = time.time()
    try:
        model = load_model()
    except Exception as e:
        return {"error": f"モデルロード失敗: {e}"}

    try:
        audio = audio_from_path(str(wav))
        result = transcribe(model, audio)
        text = result.text
    except Exception as e:
        return {"error": f"認識失敗: {e}"}

    elapsed = time.time() - t0
    return {"text": normalize(text), "elapsed": round(elapsed, 2)}


def find_audio_pairs():
    pairs = []
    data_dir = Path("/home/kennypi/data/audio")

    known_refs = {
        "テイキョウヘイセイダイガク": "teikyoheiseidaigaku.txt",
    }

    test_dirs = [
        data_dir / "music",
        data_dir / "voice_recognition",
    ]

    for d in test_dirs:
        if not d.exists():
            continue
        for mp3 in d.glob("*.mp3"):
            ref = mp3.with_suffix(".txt")
            if not ref.exists() and mp3.stem in known_refs:
                ref = d / known_refs[mp3.stem]
            if ref.exists():
                pairs.append({"audio": str(mp3), "ref": str(ref), "name": mp3.stem})
            else:
                pairs.append({"audio": str(mp3), "ref": None, "name": mp3.stem})
        for wav in d.glob("*.wav"):
            ref = wav.with_suffix(".txt")
            if ref.exists():
                pairs.append({"audio": str(wav), "ref": str(ref), "name": wav.stem})

    return pairs


def build_patterns(quick=False, model_filter=None):
    patterns = []
    WCPP_MODELS = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
    }

    VOSK_MODELS = {
        "vosk-small-ja": "vosk-model-small-ja",
        "vosk-ja": "vosk-model-ja-0.22",
    }

    wcpp_models = ["tiny", "base", "small"]
    if model_filter:
        wcpp_models = [m for m in wcpp_models if m in model_filter]

    for model_id in wcpp_models:
        model_file = WCPP_MODELS.get(model_id, model_id)
        for prompt_key in ["none", "short"]:
            patterns.append(
                {
                    "type": "wcpp",
                    "model": model_id,
                    "model_file": model_file,
                    "prompt_key": prompt_key,
                    "threads": 4,
                    "beam": 5,
                }
            )

    if not quick:
        extra_models = ["vosk-small-ja", "vosk-ja", "reazon"]
        if model_filter:
            extra_models = [m for m in extra_models if m in model_filter]

        for model_id in extra_models:
            if model_id == "reazon":
                patterns.append(
                    {
                        "type": "reazon",
                        "model": "reazon-nemo-v2",
                        "model_file": "reazon-nemo-v2",
                        "prompt_key": "none",
                    }
                )
            elif model_id in VOSK_MODELS:
                patterns.append(
                    {
                        "type": "vosk",
                        "model": model_id,
                        "model_file": VOSK_MODELS[model_id],
                        "prompt_key": "none",
                    }
                )

        patterns.append(
            {
                "type": "wcpp",
                "model": "small",
                "model_file": "small",
                "prompt_key": "keyword_full",
                "threads": 4,
                "beam": 5,
            }
        )

    return patterns


def main():
    global TEMP_WARNING, TEMP_CRITICAL

    parser = argparse.ArgumentParser(description="STTベンチマーク + 温度監視")
    parser.add_argument("--audio", help="音声ファイル (デフォルト: ディレクトリ内全て)")
    parser.add_argument("--ref", help="参照テキスト")
    parser.add_argument("--quick", action="store_true", help="代表パターンのみ")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tiny", "base", "small", "vosk-small-ja", "vosk-ja", "reazon"],
        help="テストするモデル",
    )
    parser.add_argument(
        "--temp-warning",
        type=int,
        default=TEMP_WARNING,
        help=f"警告温度 (デフォルト: {TEMP_WARNING}°C)",
    )
    parser.add_argument(
        "--temp-critical",
        type=int,
        default=TEMP_CRITICAL,
        help=f" критичная温度 (デフォルト: {TEMP_CRITICAL}°C)",
    )
    parser.add_argument(
        "--list-audio", action="store_true", help="利用可能な音声ファイル一覧"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="1テストあたりのタイムアウト秒 (デフォルト: 300秒)",
    )
    args = parser.parse_args()

    TEMP_WARNING = args.temp_warning
    TEMP_CRITICAL = args.temp_critical

    if args.list_audio:
        print("利用可能な音声ファイル:")
        for pair in find_audio_pairs():
            ref_status = "✓" if pair["ref"] else "✗"
            print(f"  [{ref_status}] {pair['name']}")
            print(f"      {pair['audio']}")
            if pair["ref"]:
                print(f"      ref: {pair['ref']}")
        return

    patterns = build_patterns(quick=args.quick, model_filter=args.models)

    audio_pairs = []
    if args.audio:
        audio_pairs.append(
            {"audio": args.audio, "ref": args.ref, "name": Path(args.audio).stem}
        )
    else:
        audio_pairs = find_audio_pairs()

    if not audio_pairs:
        print("音声ファイルが見つかりません")
        return

    print("=" * 72)
    print(
        f"  STT ベンチマーク + 温度監視 ({len(audio_pairs)}音声 × {len(patterns)}パターン)"
    )
    print(f"  警告: {TEMP_WARNING}°C | критичная: {TEMP_CRITICAL}°C")
    print("=" * 72)

    info = get_system_info()
    print("開始前:")
    print_system_status("", info)
    print()

    t_start = time.time()
    all_results = {
        "date": datetime.now().isoformat(),
        "pairs": audio_pairs,
        "results": [],
        "_start_time": t_start,
    }

    for pair_idx, pair in enumerate(audio_pairs, 1):
        print(f"\n{'=' * 72}")
        print(f"  音声 [{pair_idx}/{len(audio_pairs)}]: {pair['name']}")
        print(f"{'=' * 72}")

        wav_norm = Path(f"/tmp/bench_{pair_idx}.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(pair["audio"]),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-sample_fmt",
                "s16",
                str(wav_norm),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        lyrics = ""
        if pair["ref"] and Path(pair["ref"]).exists():
            lyrics = normalize(Path(pair["ref"]).read_text(encoding="utf-8"))
            print(f"  参照: {len(compact(lyrics))}文字")

        pair_results = {
            "name": pair["name"],
            "audio": pair["audio"],
            "ref": pair["ref"],
            "tests": [],
        }

        for pat_idx, pat in enumerate(patterns, 1):
            prompt = PROMPTS.get(pat["prompt_key"], "")
            pid = f"{pat['model']}_{pat['prompt_key']}"
            print(
                f"\n[{pair_idx}/{len(audio_pairs)}][{pat_idx}/{len(patterns)}] {pid} ({pair['name']})"
            )

            mon = start_mon()
            pat_type = pat.get("type", "wcpp")
            if pat_type == "vosk":
                r = run_vosk(wav_norm, pat["model_file"], args.timeout)
            elif pat_type == "reazon":
                r = run_reazon(wav_norm, args.timeout)
            elif pat_type == "wcpp":
                threads = pat.get("threads", 4)
                beam = pat.get("beam", 5)
                r = run_wcpp(
                    pat["model_file"],
                    wav_norm,
                    prompt,
                    threads,
                    beam,
                    args.timeout,
                )
            else:
                r = {"error": f"未知のモデルタイプ: {pat_type}"}
            res = stop_mon(mon)

            if "error" in r:
                print(f"  ❌ {r['error']}")
                pair_results["tests"].append(
                    {"id": pid, "status": "error", "error": r["error"]}
                )
            else:
                text = r["text"]
                chars = len(compact(text))
                sim = similarity(lyrics, text) if lyrics else None

                print(f"  ✅ {r['elapsed']:.1f}s | {chars}文字", end="")
                if sim is not None:
                    print(f" | 類似度 {sim:.4f}", end="")
                print()
                print(f"  CPU: {res['avg_cpu']:.0f}% | Mem: {res['peak_mem']}MB")
                print(f"  → {text[:80]}...")

                pair_results["tests"].append(
                    {
                        "id": pid,
                        "status": "ok",
                        "elapsed": r["elapsed"],
                        "chars": chars,
                        "similarity": sim,
                        "text_preview": text[:120],
                        "text_full": text,
                        "resources": res,
                        "config": pat,
                    }
                )

            info = get_system_info()
            print_system_status("温度: ", info)

            if should_cooldown(info):
                wait_for_cooldown(info.get("cpu_temp", 0))

            _save_partial(all_results, RESULTS_DIR)

        all_results["results"].append(pair_results)

        if pair_idx < len(audio_pairs):
            print("\n  次の音声へ...")
            info = get_system_info()
            if should_cooldown(info):
                wait_for_cooldown(info.get("cpu_temp", 0))

    print("\n" + "=" * 72)
    print("  結果まとめ")
    print("=" * 72)

    for pair_result in all_results["results"]:
        print(f"\n【{pair_result['name']}】")
        ok = sorted(
            [t for t in pair_result["tests"] if t["status"] == "ok"],
            key=lambda x: x["similarity"] if x["similarity"] else 0,
            reverse=True,
        )
        if not ok:
            continue
        print(f"{'ID':25s} {'時間(s)':>8s} {'類似度':>8s} {'CPU%':>6s}")
        print("-" * 55)
        for t in ok[:5]:
            sim_str = f"{t['similarity']:.4f}" if t["similarity"] is not None else "N/A"
            print(
                f"{t['id']:25s} {t['elapsed']:8.1f} {sim_str:>8s} {t['resources']['avg_cpu']:5.0f}%"
            )

        best = ok[0]
        sim_str = f"{best['similarity']:.4f}" if best["similarity"] else "N/A"
        print(f"\n🏆 最高: {best['id']} — {sim_str} / {best['elapsed']:.1f}s")

    total_elapsed = time.time() - t_start
    print(f"\n⏱️  総実行時間: {total_elapsed:.0f}秒 ({total_elapsed / 60:.1f}分)")

    all_results["total_elapsed_sec"] = round(total_elapsed, 1)
    all_results["total_elapsed_min"] = round(total_elapsed / 60, 1)
    if _partial_save_file:
        _partial_save_file.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n💾 保存完了: {_partial_save_file}")


if __name__ == "__main__":
    main()

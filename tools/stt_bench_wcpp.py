#!/usr/bin/env python3
"""whisper.cpp 専用ベンチマーク — 全パターン + リソース監視 + 結果保存"""
import subprocess, time, re, os, json, threading, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = {
    "none": "",
    "keyword": "♪音楽♪ 帝京平成大学 帝京魂 グランドライン 大大学時代 キャンパス 合格者数 実学教育 健康 医療 スポーツ 経営学 ナミ 航海士 トナカイ バケモノ 仲間 創立30周年 新型コロナウイルス めざましじゃんけん 医者の俺 男たちは夢を追い続け",
    "lyrics_head": "男たちはグランドラインを目指し 夢を追い続け 世はまさに大大学時代 キャンパキャンパスデスデス 全国合格数みんな 帝京平成大学を知ってるか 実学の総合大学 帝京平成大学 帝京魂 東京 千葉にある4つのキャンパス 健康 医療 スポーツ 経営学など 幅広い学問を学べるんだぞ 私は航海士のナミ 帝京平成大学はここもすごいのよ 徹底した実学教育 全国有数の合格者数",
}

def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('\u3000', ' ')).strip()

def compact(text):
    return normalize(text).replace(' ', '')

def char_ngrams(text, min_n=2, max_n=3):
    c = compact(text)
    return set(c[i:i+n] for n in range(min_n, max_n+1) for i in range(len(c)-n+1))

def similarity(q, a):
    qg, ag = char_ngrams(q), char_ngrams(a)
    if not qg or not ag: return 0.0
    ov = len(qg & ag)
    if ov == 0: return 0.0
    return round((ov/max(1,len(qg))*0.8) + (ov/max(1,len(ag))*0.2), 4)

# リソース監視
_mdata = {"cpu": [], "mem": []}
_mstop = threading.Event()

def _monitor():
    while not _mstop.is_set():
        try:
            c = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
            if c: _mdata["cpu"].append(float(c))
        except: pass
        try:
            m = os.popen("free -m | grep Mem | awk '{print $3}'").read().strip()
            if m: _mdata["mem"].append(int(m))
        except: pass
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
        "avg_cpu": sum(_mdata["cpu"])/len(_mdata["cpu"]) if _mdata["cpu"] else 0,
        "peak_cpu": max(_mdata["cpu"]) if _mdata["cpu"] else 0,
        "avg_mem": sum(_mdata["mem"])/len(_mdata["mem"]) if _mdata["mem"] else 0,
        "peak_mem": max(_mdata["mem"]) if _mdata["mem"] else 0,
    }

def run_wcpp(model, threads, prompt, wav):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT/"whisper.cpp"/"build"/"src") + ":" + str(ROOT/"whisper.cpp"/"build"/"ggml"/"src")
    model_path = ROOT / f".runtime/models/ggml-{model}.bin"
    if not model_path.exists():
        return {"error": f"モデルなし: {model_path}"}
    cmd = [str(ROOT/"whisper.cpp"/"build"/"bin"/"whisper-cli"), "-m", str(model_path),
           "-f", str(wav), "-l", "ja", "-t", str(threads),
           "--no-timestamps", "--beam-size", "5", "--temperature", "0.0"]
    if prompt:
        cmd += ["--prompt", prompt]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    elapsed = time.time() - t0
    combined = (p.stdout + "\n" + p.stderr).split("whisper_print_timings:", 1)[0]
    lines = []
    for line in combined.splitlines():
        s = line.strip()
        if not s: continue
        if any(k in s.lower() for k in ["whisper_", "system_info:", "main:", "n_threads",
            "whisper_init", "whisper_model_load", "compute buffer", "kv "]): continue
        s = re.sub(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", s)
        if re.search(r"[ぁ-んァ-ン一-龯A-Za-z0-9]", s): lines.append(s)
    return {"text": normalize(" ".join(lines)), "elapsed": round(elapsed, 2), "chars": 0}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="/home/kennypi/data/audio/music/テイキョウヘイセイダイガク.mp3")
    p.add_argument("--ref", default="/home/kennypi/data/audio/music/teikyoheiseidaigaku.txt")
    args = p.parse_args()

    wav_in = Path(args.audio).expanduser()
    ref = Path(args.ref).expanduser()
    if not wav_in.exists(): raise SystemExit(f"NG: {wav_in}")
    if not ref.exists(): raise SystemExit(f"NG: {ref}")

    lyrics = normalize(ref.read_text(encoding="utf-8"))
    wav_norm = Path("/tmp/bench_norm.wav")
    subprocess.run(["ffmpeg","-y","-i",str(wav_in),"-ar","16000","-ac","1","-sample_fmt","s16",str(wav_norm)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    patterns = []
    # モデルIDと実際のファイル名のマッピング（デフォルトは ggml-{id}.bin）
    MODEL_MAP = {
        "kotoba": "kotoba-whisper-v2.0-q5_0"
    }

    for model_id in ["tiny", "base", "small", "kotoba"]:
        model_file = MODEL_MAP.get(model_id, model_id)
        for pk in ["none", "keyword"]:
            patterns.append({"model": model_id, "model_file": model_file, "prompt_key": pk, "threads": 4})
    
    # lyrics_head のみ small と kotoba
    patterns.append({"model": "small", "model_file": "small", "prompt_key": "lyrics_head", "threads": 4})
    patterns.append({"model": "kotoba", "model_file": MODEL_MAP["kotoba"], "prompt_key": "lyrics_head", "threads": 4})

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"stt_bench_wcpp_{tag}.json"

    print(f"=== whisper.cpp ベンチマーク ===")
    print(f"  {wav_in.name} / {len(compact(lyrics))}文字 / {len(patterns)}パターン\n")

    results = []
    for i, pat in enumerate(patterns, 1):
        pk = pat["prompt_key"]
        prompt = PROMPTS[pk]
        pid = f"wcpp_{pat['model']}_{pk}"
        print(f"[{i}/{len(patterns)}] {pid}")

        mon = start_mon()
        r = run_wcpp(pat["model_file"], pat["threads"], prompt, wav_norm)
        res = stop_mon(mon)

        if "error" in r:
            print(f"  ❌ {r['error']}")
            results.append({"id": pid, "status": "error", "error": r["error"]})
            continue

        text = r["text"]
        # charsを再計算
        r["chars"] = len(compact(text))
        sim = similarity(lyrics, text)

        print(f"  ✅ {r['elapsed']:.1f}s | {r['chars']}文字 | 類似度 {sim:.4f} | CPU {res['avg_cpu']:.0f}% | Mem {res['peak_mem']}MB")
        print(f"  {text[:100]}...")

        results.append({
            "id": pid, "status": "ok", "elapsed": r["elapsed"],
            "chars": r["chars"], "similarity": sim,
            "text_preview": text[:120], "text_full": text,
            "resources": res, "config": pat,
        })

    # テーブル表示
    print("\n" + "="*72)
    print("  結果")
    print("="*72)
    ok = sorted([r for r in results if r["status"]=="ok"], key=lambda r: r["similarity"], reverse=True)
    print(f"{'ID':30s} {'時間(s)':>8s} {'一致率':>8s} {'文字数':>6s} {'CPU%':>6s} {'Mem':>6s}")
    print("-"*72)
    for r in ok:
        res = r["resources"]
        print(f"{r['id']:30s} {r['elapsed']:8.1f} {r['similarity']:8.4f} {r['chars']:6d} {res['avg_cpu']:5.0f}% {res['peak_mem']:5.0f}MB")

    if ok:
        best = ok[0]
        print(f"\n🏆 最高一致率: {best['id']} — {best['similarity']:.4f} / {best['elapsed']:.1f}s")

    # 保存
    save = {
        "date": datetime.now().isoformat(),
        "audio": str(wav_in),
        "reference": str(ref),
        "results": results,
    }
    out.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {out}")

if __name__ == "__main__":
    main()

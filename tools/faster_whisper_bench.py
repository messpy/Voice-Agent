#!/usr/bin/env python3
"""faster-whisper ベンチマーク — リソース監視付き"""
import subprocess, time, re, os, json, threading
from faster_whisper import WhisperModel

def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('\u3000', ' ')).strip()

def compact(text):
    return normalize(text).replace(' ', '')

def char_ngrams(text, min_n=2, max_n=3):
    c = compact(text)
    return set(c[i:i+n] for n in range(min_n, max_n+1) for i in range(len(c)-n+1))

def similarity(q, a):
    qg = char_ngrams(q)
    ag = char_ngrams(a)
    if not qg or not ag: return 0.0
    ov = len(qg & ag)
    if ov == 0: return 0.0
    return round((ov/max(1,len(qg))*0.8) + (ov/max(1,len(ag))*0.2), 4)

monitor_data = {"cpu": [], "mem": [], "gpu": []}
monitor_stop = threading.Event()

def monitor_resources():
    while not monitor_stop.is_set():
        try:
            cpu = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
            if cpu: monitor_data["cpu"].append(float(cpu))
        except: pass
        try:
            mem = os.popen("free -m | grep Mem | awk '{print $3}'").read().strip()
            if mem: monitor_data["mem"].append(int(mem))
        except: pass
        try:
            gpu = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null").read().strip()
            if gpu: monitor_data["gpu"].append(float(gpu.replace('%','')))
        except: pass
        time.sleep(1)

def run_benchmark(label, model_name, device, compute_type, wav_path, lyrics):
    print(f"\n--- {label} 開始 ---")
    t_load = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type,
                        download_root=os.path.join(os.path.dirname(__file__), ".runtime", "faster_whisper_models"))
    load_time = time.time() - t_load
    print(f"  モデルロード時間: {load_time:.1f}秒")

    global monitor_data
    monitor_data = {"cpu": [], "mem": [], "gpu": []}
    monitor_stop.clear()
    mon = threading.Thread(target=monitor_resources, daemon=True)
    mon.start()

    t0 = time.time()
    segments, info = model.transcribe(
        wav_path, language="ja", vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5, temperature=0.0,
    )
    segments_list = list(segments)
    elapsed = time.time() - t0
    text = normalize(" ".join(s.text for s in segments_list).strip())
    sim = similarity(lyrics, text)

    time.sleep(1)
    monitor_stop.set()
    mon.join(timeout=2)

    avg_cpu = sum(monitor_data["cpu"]) / len(monitor_data["cpu"]) if monitor_data["cpu"] else 0
    peak_cpu = max(monitor_data["cpu"]) if monitor_data["cpu"] else 0
    avg_mem = sum(monitor_data["mem"]) / len(monitor_data["mem"]) if monitor_data["mem"] else 0
    peak_mem = max(monitor_data["mem"]) if monitor_data["mem"] else 0
    mem_used = peak_mem

    print(f"  処理時間: {elapsed:.1f}秒")
    print(f"  認識結果文字数: {len(compact(text))}")
    print(f"  一致率: {sim:.4f}")
    print(f"  CPU使用率: 平均 {avg_cpu:.1f}% / 最大 {peak_cpu:.1f}%")
    print(f"  メモリ使用量: 平均 {avg_mem:.0f}MB / 最大 {peak_mem:.0f}MB / 増加 {mem_used}MB")
    if monitor_data["gpu"]:
        avg_gpu = sum(monitor_data["gpu"]) / len(monitor_data["gpu"])
        print(f"  GPU使用率: 平均 {avg_gpu:.1f}%")
    else:
        print(f"  GPU: なし (CPU推論)")
    print(f"  認識結果: {text[:150]}...")
    return {"label": label, "elapsed": elapsed, "chars": len(compact(text)),
            "similarity": sim, "avg_cpu": avg_cpu, "peak_cpu": peak_cpu,
            "avg_mem": avg_mem, "peak_mem": peak_mem, "mem_used": mem_used, "text": text}

def main():
    wav_in = '/home/kennypi/data/audio/music/テイキョウヘイセイダイガク.mp3'
    wav_norm = '/tmp/teikyo_normalized.wav'
    subprocess.run(["ffmpeg", "-y", "-i", wav_in, "-ar", "16000", "-ac", "1",
                    "-sample_fmt", "s16", wav_norm],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open('/home/kennypi/data/audio/music/teikyoheiseidaigaku.txt') as f:
        lyrics = f.read()

    models = [
        ("small (faster-whisper)", "small", "cpu", "int8"),
        ("kotoba-v1.0 (faster-whisper)", "kotoba-tech/kotoba-whisper-v1.0", "cpu", "int8"),
    ]

    print("=" * 60)
    print("  faster-whisper ベンチマーク")
    print("=" * 60)

    results = []
    for label, name, dev, ct in models:
        r = run_benchmark(label, name, dev, ct, wav_norm, lyrics)
        results.append(r)

    print("\n" + "=" * 60)
    print("  結果まとめ")
    print("=" * 60)
    print(f"{'モデル':30s} {'時間(s)':>8s} {'一致率':>8s} {'文字数':>6s} {'CPU%':>8s} {'Mem(MB)':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:30s} {r['elapsed']:8.1f} {r['similarity']:8.4f} {r['chars']:6d} "
              f"{r['avg_cpu']:5.1f}%  {r['peak_mem']:5.0f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

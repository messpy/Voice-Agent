#!/usr/bin/env python3
"""
STT Benchmark — 音声ファイルを渡すだけで全モデルを自動比較

使い方:
  # 全モデルでベンチマーク
  uv run python tools/stt_benchmark.py audio.wav

  # 特定のモデルのみ
  uv run python tools/stt_benchmark.py audio.wav --ids tiny base small kotoba_v2_q5_0

  # 参照テキストを指定して類似度スコアも計算
  uv run python tools/stt_benchmark.py audio.wav --ref "こんにちは、テストです。"

  # スレッド数変更
  uv run python tools/stt_benchmark.py audio.wav --threads 4

結果は JSON + 見やすいテーブル形式で出力されます。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "whisper_models.yaml"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\u3000", " ")).strip()


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def char_ngrams(text: str, *, min_n: int = 2, max_n: int = 3) -> set[str]:
    compact = compact_text(text)
    grams: set[str] = set()
    for size in range(min_n, max_n + 1):
        if len(compact) < size:
            continue
        for idx in range(len(compact) - size + 1):
            grams.add(compact[idx: idx + size])
    return grams


def similarity_score(query: str, answer: str) -> float:
    qgrams = char_ngrams(query)
    agrams = char_ngrams(answer)
    if not qgrams or not agrams:
        return 0.0
    overlap = len(qgrams & agrams)
    if overlap == 0:
        return 0.0
    coverage = overlap / max(1, len(qgrams))
    density = overlap / max(1, len(agrams))
    return round((coverage * 0.8) + (density * 0.2), 4)


# ── 外部ツール ──────────────────────────────────────────────

def ffmpeg_normalize(wav_in: Path, wav_out: Path) -> None:
    """音声を16kHz mono 16bit PCMに変換"""
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_in),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(wav_out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_whisper_cpp(
    whisper_bin: Path,
    model_path: Path,
    wav: Path,
    out_prefix: Path,
    lang: str,
    threads: int,
) -> tuple[str, float]:
    """whisper-cli で文字起こし"""
    # 共有ライブラリのパスを設定
    env = os.environ.copy()
    lib_dirs = [
        whisper_bin.parent.parent / "src",
        whisper_bin.parent.parent / "ggml" / "src",
    ]
    existing_dirs = [str(d) for d in lib_dirs if d.exists()]
    if existing_dirs:
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(existing_dirs + ([current_ld] if current_ld else []))

    cmd = [
        str(whisper_bin),
        "-m", str(model_path),
        "-f", str(wav),
        "-l", lang,
        "-t", str(threads),
        "-of", str(out_prefix),
        "-otxt",
        "--no-timestamps",
        "--beam-size", "5",
        "--temperature", "0.0",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0

    # テキスト抽出
    txt_path = Path(str(out_prefix) + ".txt")
    # stdout + stderr を結合して解析
    combined_output = proc.stdout + "\n" + proc.stderr
    if txt_path.exists():
        raw = txt_path.read_text(errors="replace")
        raw = raw.split("whisper_print_timings:", 1)[0]
        raw = re.sub(r"^output_txt:.*$", "", raw, flags=re.M)
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", line)
            if any(k in line for k in ["whisper_model_load", "system_info", "main:"]):
                continue
            lines.append(line)
        text = normalize_text(" ".join(lines))
    else:
        # stdout/stderr から抽出
        # --no-timestamps の場合、認識結果がstderrの末尾に1行で出力される
        text = _extract_whisper_text_from_output(combined_output)
    return text, round(elapsed, 3)


def _extract_whisper_text_from_output(output: str) -> str:
    """whisper-cliの出力から認識テキストを抽出"""
    # timings以降を除外
    output = output.split("whisper_print_timings:", 1)[0]

    # 認識結果と思われる行を抽出
    lines = []
    for line in output.splitlines():
        s = line.strip()
        if not s:
            continue
        # タイミング行、info行を除外
        if any(k in s.lower() for k in [
            "whisper_", "system_info:", "main:", "n_threads",
            "whisper_init", "whisper_model_load", "whisper_backend",
            "whisper_init_state", "compute buffer", "kv ",
            "adding ", "loading model", "model size",
        ]):
            continue
        # timestamp行を削除
        s = re.sub(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", s)
        # 日本語または英数字が含まれていたら本文扱い
        if re.search(r"[ぁ-んァ-ン一-龯A-Za-z0-9]", s):
            lines.append(s)

    return normalize_text(" ".join(lines))


def transcribe_moonshine(
    python_bin: Path,
    model_name: str,
    wav: Path,
    cache_dir: Path | None = None,
) -> tuple[str, float]:
    """Moonshine ONNX で文字起こし"""
    env = os.environ.copy()
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(cache_dir)

    script = """
import json, time, os
from pathlib import Path
from moonshine_onnx import transcribe
wav = Path(os.environ["MOONSHINE_WAV"])
model = os.environ["MOONSHINE_MODEL"]
start = time.perf_counter()
text = transcribe(str(wav), model=model)[0]
elapsed = time.perf_counter() - start
print(json.dumps({"text": text, "elapsed": round(elapsed, 3)}))
"""
    env["MOONSHINE_WAV"] = str(wav)
    env["MOONSHINE_MODEL"] = model_name

    proc = subprocess.run(
        [str(python_bin), "-c", script],
        capture_output=True, text=True, env=env,
    )
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"Moonshine failed: {proc.stderr}")
    payload = json.loads(lines[-1])
    return normalize_text(payload["text"]), float(payload["elapsed"])


# ── モデルカタログ ─────────────────────────────────────────

def load_catalog() -> dict[str, Any]:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p


def find_existing_path(candidate_paths: list[str]) -> Path | None:
    for raw in candidate_paths:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    return None


# ── ベンチマーク実行 ──────────────────────────────────────────

@dataclass
class BenchmarkResult:
    model_id: str
    label: str
    family: str
    role: str
    available: bool
    status: str = "pending"  # ok / error / missing
    elapsed_sec: float = 0.0
    transcript: str = ""
    chars: int = 0
    similarity: float = 0.0
    error: str = ""
    model_path: str = ""


def run_benchmark(
    wav: Path,
    model_ids: list[str] | None = None,
    lang: str = "ja",
    threads: int = 4,
    reference_text: str = "",
) -> list[BenchmarkResult]:
    catalog = load_catalog()
    models = catalog.get("models", [])

    if model_ids:
        wanted = set(model_ids)
        models = [m for m in models if m["id"] in wanted]
    if not models:
        raise SystemExit("NG: 対象モデルがありません")

    # 正規化wav
    norm_wav = wav.with_name(wav.stem + "_normalized.wav")
    ffmpeg_normalize(wav, norm_wav)
    print(f"  正規化: {norm_wav.name}")

    whisper_bin_path = find_existing_path([
        str(ROOT / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        str(ROOT / ".runtime" / "whisper-cli"),
    ])

    results: list[BenchmarkResult] = []

    for row in models:
        rid = row["id"]
        label = row.get("label", rid)
        family = row.get("family", "")
        role = row.get("role", "")

        result = BenchmarkResult(
            model_id=rid, label=label, family=family, role=role, available=False,
        )

        if family == "whisper.cpp":
            model_path = find_existing_path(row.get("candidate_paths", []))
            if not model_path:
                result.status = "missing"
                result.error = "モデルファイルがありません"
                results.append(result)
                continue
            result.available = True
            result.model_path = str(model_path)

            try:
                out_prefix = wav.parent / f"out_{rid}"
                text, elapsed = transcribe_whisper_cpp(
                    whisper_bin=whisper_bin_path,
                    model_path=model_path,
                    wav=norm_wav,
                    out_prefix=out_prefix,
                    lang=lang,
                    threads=threads,
                )
                result.status = "ok"
                result.elapsed_sec = elapsed
                result.transcript = text
                result.chars = len(compact_text(text))
            except Exception as e:
                result.status = "error"
                result.error = str(e)

        elif family == "moonshine_onnx":
            python_bin = find_existing_path(row.get("python_bin_candidates", []))
            if not python_bin:
                result.status = "missing"
                result.error = "Python venv が見つかりません"
                results.append(result)
                continue
            result.available = True

            try:
                text, elapsed = transcribe_moonshine(
                    python_bin=python_bin,
                    model_name=row.get("model_name", "moonshine/tiny-ja"),
                    wav=norm_wav,
                    cache_dir=resolve_path(row.get("cache_dir")),
                )
                result.status = "ok"
                result.elapsed_sec = elapsed
                result.transcript = text
                result.chars = len(compact_text(text))
            except Exception as e:
                result.status = "error"
                result.error = str(e)
        else:
            result.status = "error"
            result.error = f"未対応ファミリー: {family}"

        if reference_text and result.status == "ok":
            result.similarity = similarity_score(reference_text, result.transcript)

        results.append(result)

    return results


# ── 結果表示 ──────────────────────────────────────────────

def print_table(results: list[BenchmarkResult], reference_text: str = "") -> None:
    """見やすいテーブル形式で出力"""
    # ヘッダー
    cols = ["ID", "モデル", "状態", "時間(s)", "文字数", "類似度", "認識結果"]
    if reference_text:
        cols.append("参照テキスト")

    # 幅計算
    max_id = max(len(r.model_id) for r in results)
    max_label = max(len(r.label) for r in results)
    max_transcript = max(len(r.transcript) for r in results) if results else 40
    max_transcript = min(max_transcript, 80)

    col_widths = [
        max(max_id, 6),
        max(max_label, 10),
        10,
        10,
        8,
        8,
        max(max_transcript, 20),
    ]

    # 区切り線
    def sep():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def row(values: list[str]):
        cells = []
        for val, w in zip(values, col_widths):
            cells.append(f" {val:<{w}} ")
        return "|" + "|".join(cells) + "|"

    print()
    print(sep())
    print(row(["ID", "モデル", "状態", "時間(s)", "文字数", "類似度", "認識結果"]))
    print(sep())

    for r in results:
        status_icon = {
            "ok": "✅ OK",
            "error": "❌ ERR",
            "missing": "⬜ 無し",
        }.get(r.status, r.status)

        time_str = f"{r.elapsed_sec:.2f}" if r.elapsed_sec else "-"
        chars_str = str(r.chars) if r.chars else "-"
        sim_str = f"{r.similarity:.3f}" if r.similarity else "-"

        # 認識結果は truncate
        transcript_display = r.transcript[:80] + ("..." if len(r.transcript) > 80 else "")
        if not transcript_display:
            transcript_display = r.error or "(空)"

        values = [
            r.model_id,
            r.label,
            status_icon,
            time_str,
            chars_str,
            sim_str,
            transcript_display,
        ]
        print(row(values))

    print(sep())
    print()


def print_summary(results: list[BenchmarkResult], reference_text: str = "") -> None:
    """推奨モデルを提示"""
    ok_results = [r for r in results if r.status == "ok"]
    if not ok_results:
        print("⚠️  成功したモデルはありません")
        return

    if reference_text:
        # 類似度が高い順
        ok_results.sort(key=lambda r: (r.similarity, -r.elapsed_sec), reverse=True)
        best = ok_results[0]
        print(f"🏆 推奨モデル (類似度優先): {best.model_id} — {best.label}")
        print(f"   類似度: {best.similarity:.3f} | 処理時間: {best.elapsed_sec:.2f}s")
    else:
        # 高速順
        ok_results.sort(key=lambda r: r.elapsed_sec)
        best = ok_results[0]
        print(f"🏆 推奨モデル (速度優先): {best.model_id} — {best.label}")
        print(f"   処理時間: {best.elapsed_sec:.2f}s")

    print()
    # 速度ランキング
    print("── 速度ランキング ──")
    for i, r in enumerate(ok_results, 1):
        print(f"  {i}. {r.model_id:20s} {r.elapsed_sec:8.2f}s  ({r.label})")
    print()


def save_results(results: list[BenchmarkResult], out_dir: Path, reference_text: str = "") -> Path:
    """JSON で結果保存"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    data = {
        "date": tag,
        "reference_text": reference_text,
        "results": [
            {
                "id": r.model_id,
                "label": r.label,
                "family": r.family,
                "role": r.role,
                "status": r.status,
                "elapsed_sec": r.elapsed_sec,
                "transcript": r.transcript,
                "chars": r.chars,
                "similarity": r.similarity,
                "error": r.error,
                "model_path": r.model_path,
            }
            for r in results
        ],
    }

    out_path = out_dir / f"benchmark_{tag}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ── CLI ───────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STT Benchmark — 音声ファイルを渡すだけで全モデルを自動比較",
    )
    parser.add_argument("input", type=Path, help="入力音声ファイル (wav/mp3/flac/m4a等)")
    parser.add_argument("--ids", nargs="+", help="対象モデルID（指定しない場合は全モデル）")
    parser.add_argument("--ref", default="", help="参照テキスト（類似度スコア計算用）")
    parser.add_argument("--ref-file", type=Path, help="参照テキストファイル")
    parser.add_argument("--lang", default="ja", help="言語 (デフォルト: ja)")
    parser.add_argument("--threads", type=int, default=4, help="スレッド数 (デフォルト: 4)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmark_results", help="出力ディレクトリ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 入力ファイルチェック
    if not args.input.exists():
        raise SystemExit(f"NG: 入力ファイルが見つかりません: {args.input}")

    # 参照テキスト
    ref_text = args.ref
    if args.ref_file:
        ref_text = normalize_text(args.ref_file.expanduser().read_text(encoding="utf-8"))

    print("=" * 72)
    print("  STT Benchmark")
    print("=" * 72)
    print(f"  入力: {args.input.name}")
    print(f"  言語: {args.lang}")
    print(f"  スレッド: {args.threads}")
    if ref_text:
        print(f"  参照テキスト: {ref_text[:60]}...")
    print()

    results = run_benchmark(
        wav=args.input,
        model_ids=args.ids,
        lang=args.lang,
        threads=args.threads,
        reference_text=ref_text,
    )

    # テーブル表示
    print_table(results, ref_text)
    print_summary(results, ref_text)

    # JSON保存
    out_path = save_results(results, args.out_dir, ref_text)
    print(f"💾 結果保存: {out_path}")
    print()


if __name__ == "__main__":
    main()

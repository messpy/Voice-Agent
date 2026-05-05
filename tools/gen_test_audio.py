#!/usr/bin/env python3
"""
テスト用音声ファイル生成スクリプト

VOICEVOX を使って日本語のテスト音声を生成します。
ベンチマーク用の標準テストケースとして利用できます。

使い方:
  # デフォルトテスト音声生成（ずんだもん）
  uv run python tools/gen_test_audio.py

  # 出力先を指定
  uv run python tools/gen_test_audio.py --out /tmp/test_audio.wav

  # テキストを指定
  uv run python tools/gen_test_audio.py --text "こんにちは、音声認識のテストです。"

  # テストスイート生成（複数のパターン）
  uv run python tools/gen_test_audio.py --suite
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
VOICEVOX_URL = "http://127.0.0.1:50021"
SPEAKER_ID = 3  # ずんだもん


# テスト用テキストスイート
TEST_CASES = {
    "short": "こんにちは、音声認識のテストです。",
    "greeting": "おはようございます。今日もいい天気ですね。一緒に散歩しませんか。",
    "numbers": "電話番号は零三の千百二十二番です。価格は二千五百円で、重さは百グラムです。",
    "mixed": "PythonでAIを開発しています。バージョンは3.12で、GPUはRTX 4090を使っています。",
    "long": "音声認識技術は近年大きく進歩しました。ディープラーニングの登場により、\
              従来よりもはるかに高い精度で音声をテキストに変換できるようになっています。\
              特にTransformerアーキテクチャの採用により、長文の認識精度が大幅に改善されました。\
              日本語の場合、同音異義語が多く存在するため、文脈を理解することが重要です。",
    "tech": " whisper は OpenAI が開発した音声認識モデルで、\
            多言語対応と高精度な文字起こしが特徴です。\
            コタバテックの kotoba-whisper は日本語に特化したファインチューニング版で、\
            固有名词や専門用語の認識精度が向上しています。",
    "tongue": "生麦生米生卵。赤巻紙青巻紙黄巻紙。東京特許許可局。",
}


def generate_audio(text: str, out_path: Path, speaker_id: int = SPEAKER_ID) -> None:
    """VOICEVOX で音声を生成して保存"""
    base = VOICEVOX_URL.rstrip("/")

    # アクセントクエリ取得
    print(f"  アクセントクエリ取得中...")
    query_resp = requests.post(
        base + "/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=30,
    )
    query_resp.raise_for_status()
    query = query_resp.json()

    # 音声合成
    print(f"  音声合成中...")
    synth_resp = requests.post(
        base + "/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=120,
    )
    synth_resp.raise_for_status()

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(synth_resp.content)

    # 再生時間確認
    duration = len(synth_resp.content) / (48000 * 2 * 2)  # 48kHz 16bit stereo 概算
    print(f"  ✅ 生成完了: {out_path.name} ({len(synth_resp.content) / 1024:.0f} KB, 約{duration:.1f}秒)")


def gen_default(out_path: Path, text: str | None = None) -> None:
    """デフォルトテスト音声生成"""
    t = text or TEST_CASES["short"]
    print(f"テキスト: {t}")
    generate_audio(t, out_path)


def gen_suite(out_dir: Path) -> None:
    """テストスイート生成"""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"テストスイート生成: {len(TEST_CASES)} 件")
    print()

    results = {}
    for name, text in TEST_CASES.items():
        out_path = out_dir / f"test_{name}.wav"
        print(f"[{name}] {text[:40]}...")
        try:
            generate_audio(text, out_path)
            results[name] = {
                "path": str(out_path),
                "text": text,
                "size_kb": round(out_path.stat().st_size / 1024, 1),
            }
        except requests.ConnectionError:
            print(f"  ❌ VOICEVOX に接続できません。VOICEVOXを起動してください。")
            return
        except Exception as e:
            print(f"  ❌ エラー: {e}")

    # マニフェスト保存
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speaker_id": SPEAKER_ID,
        "cases": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n💾 マニフェスト: {manifest}")
    print(f"\nベンチマーク実行例:")
    for name in TEST_CASES:
        print(f"  uv run python tools/stt_benchmark.py {out_dir}/test_{name}.wav --ref \"{TEST_CASES[name]}\"")


def check_voicevox() -> bool:
    """VOICEVOX が起動しているか確認"""
    try:
        resp = requests.get(f"{VOICEVOX_URL}/version", timeout=5)
        if resp.status_code == 200:
            print(f"  VOICEVOX バージョン: {resp.text.strip()}")
            return True
    except requests.ConnectionError:
        pass
    print(f"  ❌ VOICEVOX ({VOICEVOX_URL}) に接続できません")
    print(f"  VOICEVOX を起動してから実行してください")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="テスト用音声生成")
    parser.add_argument("--out", type=Path, default=ROOT / "test_audio.wav", help="出力ファイルパス")
    parser.add_argument("--text", type=str, help="合成するテキスト")
    parser.add_argument("--suite", action="store_true", help="テストスイート生成")
    parser.add_argument("--speaker", type=int, default=SPEAKER_ID, help="話者ID (デフォルト: 3=ずんだもん)")
    args = parser.parse_args()

    global SPEAKER_ID
    SPEAKER_ID = args.speaker

    print("=" * 60)
    print("  テスト用音声生成")
    print("=" * 60)

    if not check_voicevox():
        raise SystemExit(1)

    if args.suite:
        out_dir = args.out if args.out.is_dir() else ROOT / "test_audio_suite"
        gen_suite(out_dir)
    else:
        gen_default(args.out, args.text)

    print("\n" + "=" * 60)
    print("  完了")
    print("=" * 60)


if __name__ == "__main__":
    main()

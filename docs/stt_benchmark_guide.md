# STT ベンチマーク ガイド

voicechat の音声認識(STT)モデルをベンチマークするためのツール群です。

## ツール一覧

| ファイル | 用途 |
|---|---|
| `tools/stt_benchmark.py` | メインのベンチマークスクリプト |
| `tools/vad_benchmark.py` | WebRTC VAD / Silero VAD 比較 |
| `tools/download_models.py` | STTモデルの一括ダウンロード |
| `tools/gen_test_audio.py` | テスト用音声ファイル生成 |

## クイックスタート

### 1. モデルのダウンロード

```bash
cd /home/kennypi/work/voicechat

# Kotoba Whisper v2.0 (日本語特化) をダウンロード
uv run python tools/download_models.py --ids kotoba_v2_q5_0

# 全モデルをダウンロード
uv run python tools/download_models.py
```

### 2. テスト用音声の生成

```bash
# テストスイート生成 (7パターンの音声をVOICEVOXで生成)
uv run python tools/gen_test_audio.py --suite

# 単一のテスト音声生成
uv run python tools/gen_test_audio.py --out test_hello.wav --text "こんにちは、テストです。"
```

### 3. ベンチマーク実行

```bash
# 全モデルでベンチマーク
uv run python tools/stt_benchmark.py test_audio_suite/test_short.wav

# 特定のモデルのみ
uv run python tools/stt_benchmark.py test_audio_suite/test_short.wav --ids small kotoba_v2_q5_0

# 参照テキストを指定して類似度スコアも計算
uv run python tools/stt_benchmark.py test_audio_suite/test_short.wav \
  --ref "こんにちは、音声認識のテストです。"

# スレッド数変更
uv run python tools/stt_benchmark.py test_audio_suite/test_short.wav --threads 4
```

### 4. VAD ベンチマーク実行

```bash
# 以前の STT ベンチで使った音声データを既定で比較
uv run --extra wake python tools/vad_benchmark.py

# 旧 test_audio_suite を使う場合
uv run --extra wake python tools/vad_benchmark.py --suite test_audio_suite

# 特定ファイルだけ比較
uv run --extra wake python tools/vad_benchmark.py test_audio_suite/seg_01.wav test_audio_suite/seg_02.wav

# 期待値つきで比較
uv run --extra wake python tools/vad_benchmark.py \
  test_audio_suite/seg_01.wav \
  --expect seg_01.wav=yes
```

`silero-vad` は `wake` extra に入っているため、`uv run --extra wake ...` を使います。
既定では、以前の STT ベンチで使った `/home/kennypi/data/audio/music/テイキョウヘイセイダイガク.mp3` と `/home/kennypi/data/audio/voice_recognition/ItsOnlyNewYork.mp3` を対象にします。`test_audio_suite` を使いたい場合は `--suite test_audio_suite` を指定します。

## ベンチマーク結果の例

```
========================================================================
  STT Benchmark
========================================================================
  入力: test_short.wav
  言語: ja
  スレッド: 4
  参照テキスト: こんにちは、音声認識のテストです。

+----------+------------------+------------+------------+---------+---------+--------------------+
| ID       | モデル           | 状態       | 時間(s)    | 文字数  | 類似度  | 認識結果           |
+----------+------------------+------------+------------+---------+---------+--------------------+
| tiny     | OpenAI Whisper.. | ✅ OK      |       1.23 |      15 |   0.850 | こんにちは 音声認識のテストです |
| base     | OpenAI Whisper.. | ✅ OK      |       2.45 |      15 |   0.920 | こんにちは、音声認識のテストです。 |
| small    | OpenAI Whisper.. | ✅ OK      |       4.67 |      15 |   0.950 | こんにちは、音声認識のテストです。 |
| kotoba.. | Kotoba Whisper.. | ✅ OK      |       5.12 |      15 |   0.980 | こんにちは、音声認識のテストです。 |
+----------+------------------+------------+------------+---------+---------+--------------------+

🏆 推奨モデル (類似度優先): kotoba_v2_q5_0 — Kotoba Whisper v2.0 q5_0
   類似度: 0.980 | 処理時間: 5.12s

── 速度ランキング ──
  1. tiny                       1.23s  (OpenAI Whisper tiny)
  2. base                       2.45s  (OpenAI Whisper base)
  3. small                      4.67s  (OpenAI Whisper small)
  4. kotoba_v2_q5_0             5.12s  (Kotoba Whisper v2.0 q5_0)

💾 結果保存: /home/kennypi/work/voicechat/benchmark_results/benchmark_20260414_123456.json
```

## 利用可能なモデル

`config/whisper_models.yaml` で定義されています。

| ID | ファミリー | 説明 | サイズ |
|---|---|---|---|
| `tiny` | whisper.cpp | OpenAI Whisper tiny | ~75 MB |
| `base` | whisper.cpp | OpenAI Whisper base | ~142 MB |
| `small` | whisper.cpp | OpenAI Whisper small | ~466 MB |
| `medium` | whisper.cpp | OpenAI Whisper medium | ~1.5 GB |
| `large_v3_turbo_q5_0` | whisper.cpp | OpenAI Whisper large-v3-turbo (q5_0) | ~1.6 GB |
| `kotoba_v2_q5_0` | whisper.cpp | Kotoba Whisper v2.0 (日本語特化, q5_0) | ~1.0 GB |
| `moonshine_tiny_ja` | moonshine_onnx | Moonshine tiny-ja | - |

## テストスイートの内容

`gen_test_audio.py --suite` で生成されるテストケース:

| ファイル | 内容 | 目的 |
|---|---|---|
| `test_short.wav` | 短い挨拶 | 基本認識テスト |
| `test_greeting.wav` | 丁寧な挨拶 | 敬語・丁寧詞認識 |
| `test_numbers.wav` | 数字・数量 | 数詞認識 |
| `test_mixed.wav` | 日本語+英語+数字 | 混在テキスト |
| `test_long.wav` | 長文 | 長文認識・一貫性 |
| `test_tech.wav` | 技術用語 | 専門用語認識 |
| `test_tongue.wav` | 早口言葉 | 類似音の識別 |

## ベンチマークのカスタマイズ

### スレッド数

Raspberry Pi のCPUコア数に合わせて調整:

```bash
# ラズパイ4 (4コア)
uv run python tools/stt_benchmark.py audio.wav --threads 4

# ラズパイ5 (4コア)
uv run python tools/stt_benchmark.py audio.wav --threads 4
```

### 参照テキスト付きで精度測定

```bash
uv run python tools/stt_benchmark.py test_audio.wav \
  --ref "正確に認識させたいテキスト"
```

### テキストファイルから参照テキストを読み込み

```bash
echo "期待するテキスト" > reference.txt
uv run python tools/stt_benchmark.py test_audio.wav \
  --ref-file reference.txt
```

### 結果の比較

```bash
# 前回の結果と比較
ls -la benchmark_results/
cat benchmark_results/benchmark_20260414_*.json | python -m json.tool
```

## 自分の音声でテストする場合

```bash
# 1. 音声ファイルを準備 (wav/mp3/flac/m4a等)
#    例: スマフォで録音 → PCに転送

# 2. ベンチマーク実行
uv run python tools/stt_benchmark.py my_voice.wav

# 3. 参照テキストを指定して精度測定
uv run python tools/stt_benchmark.py my_voice.wav \
  --ref "実際に話した内容のテキスト"
```

## トラブルシューティング

### `whisper-cli` が見つからない

```bash
# whisper.cpp をビルド
cd /home/kennypi/work/voicechat/whisper.cpp
cmake -B build
cmake --build build -j4
```

### モデルファイルが見つからない

```bash
# ダウンロード
uv run python tools/download_models.py --ids small kotoba_v2_q5_0
```

### VOICEVOX に接続できない

```bash
# VOICEVOX を起動
voicevox &
# または Docker
docker run --rm -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu22.04-latest
```

### Moonshine が動かない

```bash
# venv 作成 + インストール
python3 -m venv .moonshine-venv
source .moonshine-venv/bin/activate
pip install moonshine-onnx
```

### Silero VAD が動かない

```bash
uv sync --extra wake
uv run --extra wake python tools/vad_benchmark.py
```

# voicechat 基本設計書

最終更新: 2026-03-13

## 1. アーキテクチャ概要

本システムは単一プロセス中心の統合ランナー構成を採用する。

処理の基本流れ:

1. 設定読込
2. 音声入力待受
3. 録音
4. 音声認識
5. 必要なら AI 補正
6. モード別処理
7. 音声応答
8. ログ保存

## 2. コンポーネント構成

### 2.1 Runtime Controller

責務:

- `runtime.run_mode` に応じた処理分岐
- ローカル/リモート認識 backend 切替
- AI 補正有無の切替

主実装:

- [tools/wake_vad_record.py](tools/wake_vad_record.py)

### 2.2 Audio Capture

責務:

- `arecord` による録音
- `webrtcvad` による発話区間検出

### 2.3 STT Layer

責務:

- `local`: `whisper.cpp`
- `ssh_remote`: `scp + ssh + faster-whisper`

### 2.4 Correction Layer

責務:

- 生の認識結果に対する自然文補正
- phrase test の評価補助
- `Ollama` / `Gemini` / `OpenAI` / `Anthropic` 切替
- Ollama Web Search による補助コンテキスト注入

### 2.5 TTS Layer

責務:

- `VOICEVOX` 呼び出し
- ずんだもん固定での音声応答

### 2.6 Command Router

責務:

- 認識結果を音声コマンドへ割当
- 音量操作
- 音楽再生/停止

### 2.7 Persistence

責務:

- `JSONL` 記録
- `SQLite` 記録
- 長期会話検索の検索元

## 3. 外部依存

- `VOICEVOX engine`
- `whisper.cpp`
- `Ollama`
- `Gemini`
- `OpenAI`
- `Anthropic`
- `arecord`
- `aplay`
- `ssh`
- `scp`

## 4. 主要設定

主要設定は [config/config.yaml](config/config.yaml) に集約する。

設定カテゴリ:

- `runtime`
- `storage`
- `timed_record`
- `remote`
- `audio`
- `wake`
- `vad`
- `whisper`
- `ollama`
- `assistant`
- `tts`
- `command_router`

## 5. 運用モード設計

### 5.1 phrase_test

- 指定語をずんだもんで読み上げ
- 復唱を録音
- 生認識と補正結果を記録
- expected / recognized / elapsed を保存

### 5.2 debug_stt

- 連続音声認識
- 認識結果をそのまま読み返す

### 5.3 always_on

- `ベリベリ` をウェイクワードにして待受
- 後続発話を認識
- コマンドまたは会話に分岐

### 5.4 timed_record

- 開始音声
- 固定時間録音
- 終了音声
- 文字起こし
- 保存

### 5.5 ai_duo

- AI 同士の会話生成
- TTS 読み上げ

## 6. 非機能方針

- 複雑なオーケストレーションは避ける
- 設定だけで切替可能にする
- SSD 上へ重いモデルを逃がす
- 常時運用は軽量 backend を前提にする

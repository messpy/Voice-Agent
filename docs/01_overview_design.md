# voicechat 概要設計書

最終更新: 2026-03-13

## 1. 目的

`voicechat` は Raspberry Pi 上で動作する常時待受型の音声インターフェース基盤である。

本システムは以下を目的とする。

- ずんだもん音声での自然な音声入出力
- 復唱テストによる音声認識精度の計測
- `ベリベリ` を起点とした音声リモコン
- 固定時間録音と文字起こし
- ローカル処理と SSH オフロード処理の切替
- 実験結果と運用ログの永続記録

## 2. システムの位置づけ

本システムは音声 I/O の実験基盤であり、単一機能アプリではなく、複数の実行モードを切り替えながら使う統合ランナーである。

中心コンポーネント:

- [tools/wake_vad_record.py](tools/wake_vad_record.py)

補助コンポーネント:

- [tools/timed_record_transcribe.py](tools/timed_record_transcribe.py)
- [tools/zundamon_style_check.py](tools/zundamon_style_check.py)
- [tools/remote_faster_whisper_transcribe.py](tools/remote_faster_whisper_transcribe.py)
- [tools/remote_faster_whisper_bench.py](tools/remote_faster_whisper_bench.py)

## 3. 提供機能

### 3.1 音声認識

- `whisper.cpp` によるローカル認識
- `faster-whisper` による別 PC 認識
- AI 補正あり/なしの切替

### 3.2 音声合成

- `VOICEVOX` によるずんだもん固定の読み上げ
- スタイル切替による復唱テスト

### 3.3 実行モード

- `phrase_test`
- `debug_stt`
- `always_on`
- `timed_record`
- `ai_duo`

### 3.4 記録

- `JSONL`
- `SQLite`

## 4. 対象外

現時点では次は対象外とする。

- GUI 管理画面
- 本格的なユーザー管理
- 複数同時端末制御
- 高度な音源分離
- systemd ユニットの自動生成

## 5. 運用前提

- Raspberry Pi 側にマイクとスピーカーが接続されていること
- `VOICEVOX engine` が起動していること
- ローカル補正時は `Ollama` が起動していること
- SSH オフロード時は公開鍵認証が通ること

## 6. 成功条件

- 実行モードを設定だけで切り替えられること
- 文字起こし backend を `local` / `ssh_remote` で切り替えられること
- AI 補正を設定で切り替えられること
- ログを `JSONL` と `SQLite` の両方に残せること
- 将来の常時運用に耐える単純な構成であること

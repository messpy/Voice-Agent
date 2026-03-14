# voicechat 設計メモ

最終更新: 2026-03-14

## 目的

このプロジェクトは Raspberry Pi 上で常時待受できる音声インターフェースを作りつつ、必要に応じて別 PC へ文字起こし処理を逃がせる構成を目指す。

主な要求は次のとおり。

- ずんだもん固定の音声合成
- 復唱テスト
- `ベリベリ` 起点の音声リモコン
- 5分など固定時間の録音と文字起こし
- 生認識結果と AI 補正結果の両保存
- ローカル処理と SSH オフロード処理の切替
- 記録をあとから追えること
- 実行中状態を CLI から確認できること
- コマンド誤認識を DB へ蓄積して次回認識に活かせること

## 現在の実装方針

本線は `tools/wake_vad_record.py` に集約する。

役割:

- `runtime.run_mode` に応じて挙動を切り替える
- `phrase_test`
- `debug_stt`
- `always_on`
- `timed_record`
- `ai_duo`

補助スクリプト:

- `tools/timed_record_transcribe.py`
  単独実行向けの固定時間録音
- `tools/remote/faster_whisper_transcribe.py`
  別 PC 側の単発文字起こし
- `tools/remote/faster_whisper_bench.py`
  別 PC 側のモデル比較

## 構成

### 音声入力

- `arecord` で 16kHz / mono 録音
- 待受中は短い区間で録音
- `webrtcvad` で発話区間を検出

### 音声認識

2 系統ある。

- `local`
  `whisper.cpp`
- `ssh_remote`
  Raspberry Pi で録音した WAV を別 PC へ `scp` し、`faster-whisper` で認識

`always_on` のローカル本線では、さらに 2 段に分ける。

- `wake.whisper_model`
  ウェイクワード専用
- `wake.backend`
  `whisper` または `porcupine`
- `wake.pre_vad_backend`
  wake 前段の発話検出。`webrtc` または `silero`
- `whisper.realtime_model`
  コマンド判定などリアルタイム応答用
- `whisper.model`
  保存用、あとから見る文字起こし用

このため、体感速度は軽量モデル、ログ品質は重めモデルに寄せる。

追加の wake pipeline として、設定次第で次も使える。

- `audio_pipeline.rnnoise`
  外部コマンドでのノイズ低減
- `Silero VAD`
  wake 窓に発話があるかの前段チェック
- `Porcupine`
  wake word 専用検出

### AI 補正

- ローカル処理時は Pi 上の `Ollama`
- SSH オフロード時は別 PC 上の `Ollama`

`runtime.ai_correction` で有効化する。

## 記録

同じイベントを 2 つの形式で保存する。

- `JSONL`
  人がすぐ読むため
- `SQLite`
  後で抽出しやすくするため

`events` テーブルの主要カラム:

- `ts`
- `date`
- `event_type`
- `mode`
- `backend`
- `model`
- `expected`
- `recognized`
- `elapsed_sec`
- `payload_json`

補足情報は `payload_json` に丸ごと残す。

追加の学習用テーブル:

- `recognition_aliases`
  コマンドやウェイク語の alias 学習

ランタイム状態:

- `/tmp/voicechat/runtime_state.json`
  現在の待機状態、使用モデル、最新認識の要約

## 音声コマンド

`command_router.commands` は現在、外部 action IF を前提にする。

主な項目:

- `id`
- `phrases`
- `reply`
- `action_type`
- `action_name`
- `args`

数値付きコマンドは `command_router.dynamic_patterns` で吸収する。

例:

- `音量を30%にして`
- `音量を3にして`
- `5分タイマー`
- `7時30分にアラームセットして`

動作:

1. 認識結果を受ける
2. 必要なら AI 補正する
3. `dynamic_patterns` または `phrases` と照合する
4. 一致したら `action_type/action_name/args` を組み立てる
5. `command_router.action_runners` で runner を選ぶ
6. 実行結果をイベントとして保存する

`external_cli` runner では JSON を stdin に渡し、JSON を stdout で受ける。

IF:

- 入力
  `action_type`, `action_name`, `args`
- 出力
  `success`, `message`, `data`

未知コマンドは DB からあとで alias として再学習できる。

## 外部 action backend

推奨構成では、音声認識本体と機能実装を分ける。

- `voicechat`
  聞き取り、ウェイク、コマンド判定、状態管理
- `voicechat-actions`
  `timer`, `alarm`, `light`, `music`, `audio` などの実処理

この分離により、音声側は orchestrator に寄せられる。

## 起動時依存サービス

起動前に次を確認する。

- `VOICEVOX`
- `Ollama`

`config/config.yaml` の `services.*` に `auto_start: true` があれば、ヘルスチェック失敗時に `start_command` で自動起動する。

## CLI 運用補助

追加した補助スクリプト:

- `tools/admin/show_status.py`
  実行中状態、最新認識、現在設定の表示
- `tools/admin/recognition_alias_manager.py`
  未知コマンド一覧確認、manual alias 追加

## ドキュメント方針

入口はルートの `README.md` と `voicechat.sh` に集約する。
技術メモは本ファイルに集約し、古い分割設計書や英語版の重複資料は持たない。

## 固定時間録音

`timed_record` では次を行う。

1. ずんだもんで `スタート`
2. 指定秒数録音
3. ずんだもんで `{minutes}分終了`
4. 文字起こし
5. AI 補正
6. 保存

## 今後の拡張候補

- 音声区間キューと非同期ワーカー分離
- `SQLite` からの簡易ダッシュボード表示
- コマンド実行結果の再試行制御
- 音楽再生 API の正式抽象化
- realtime backend と final backend の完全分離
- 手動補正 UI から `recognition_aliases` を育てる運用

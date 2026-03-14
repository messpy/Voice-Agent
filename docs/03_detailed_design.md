# voicechat 詳細設計書

最終更新: 2026-03-13

## 1. 対象

本書は [tools/wake_vad_record.py](tools/wake_vad_record.py) を中心とした主要処理の詳細設計を示す。

## 2. 主要関数設計

### 2.1 設定・共通

#### `load_cfg()`

責務:

- 設定ファイルの読込
- `VOICECHAT_CONFIG` による上書き

入力:

- なし

出力:

- 設定 dict

#### `sanity(cfg)`

責務:

- 実行前依存確認
- TTS, STT, Ollama, SSH 鍵の事前確認

### 2.2 STT

#### `whisper_transcribe_txt(...)`

責務:

- `whisper.cpp` でのローカル認識
- `.txt` 出力を正として抽出

#### `transcribe_local(...)`

責務:

- ローカル認識の計測ラッパ

返却:

- `raw_text`
- `elapsed_sec`
- `model_name`

#### `transcribe_remote(...)`

責務:

- WAV を別 PC へ転送
- リモートスクリプト実行
- 結果ファイルの回収

返却:

- `raw_text`
- `corrected_text`
- `elapsed_sec`
- `model_name`

#### `transcribe_audio(...)`

責務:

- `runtime.transcription_backend` に応じた backend 分岐

### 2.3 補正・評価

#### `correct_transcript(...)`

責務:

- `Ollama` による自然文補正

#### `judge_phrase_result(...)`

責務:

- phrase test の expected / recognized 評価

### 2.4 TTS

#### `speak_with_voicevox(...)`

責務:

- VOICEVOX `/audio_query`
- VOICEVOX `/synthesis`
- 生成 wav の `aplay`

### 2.5 コマンド実行

#### `match_command(text, command_router_cfg)`

責務:

- 認識文を `command_router.commands` に照合

#### `execute_command_action(item, command_execution_enabled)`

責務:

- 設定済み shell command を実行
- 実行抑止時は `skipped` を返す

### 2.6 永続化

#### `init_event_db(path)`

責務:

- `events` テーブル作成

#### `append_db_event(path, event_type, data)`

責務:

- SQLite へイベントを保存

#### `append_event_logs(...)`

責務:

- `JSONL` と `SQLite` への二重保存

## 3. モード別シーケンス

### 3.1 always_on

1. wake 音声を一定秒数録音
2. wake 認識
3. `ベリベリ` を含むか判定
4. 続きの発話を録音
5. 認識
6. 必要なら AI 補正
7. command match
8. コマンドまたは会話処理
9. TTS 応答
10. ログ保存

### 3.2 phrase_test

1. フレーズ一覧から次文を取得
2. ずんだもんスタイルを決定
3. 出題音声生成
4. 復唱音声録音
5. 認識
6. 必要なら補正
7. expected / recognized 比較
8. ログ保存

### 3.3 timed_record

1. `スタート` 読み上げ
2. 指定秒数録音
3. `{minutes}分終了` 読み上げ
4. backend で認識
5. 必要なら補正
6. テキストとメタ情報保存
7. イベント保存

## 4. エラー処理

- 依存不足は `die()` で即終了
- リモート転送失敗は例外
- 音声未検出はスキップ
- 認識空文字時はリトライ案内

## 5. 技術的負債

- 非同期ワーカー分離が未実装
- `systemd` 化が未実装
- コマンドルータは単純な部分一致
- リモート処理のリトライ制御が未実装

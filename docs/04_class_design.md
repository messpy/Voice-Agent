# voicechat クラス・責務設計書

最終更新: 2026-03-13

## 1. 方針

現状の実装は関数中心であり、明確なクラス層はまだ導入していない。

ただし今後の拡張を見越して、責務単位の論理クラス設計を定義する。

## 2. 論理クラス一覧

### 2.1 `RuntimeConfig`

責務:

- `config.yaml` から必要設定を取り出す
- 実行モードに応じた設定解決

主属性:

- `run_mode`
- `assistant_mode`
- `ai_correction`
- `transcription_backend`
- `command_execution`

### 2.2 `AudioRecorder`

責務:

- `arecord` 実行
- chunk 単位録音
- wav 保存

主メソッド候補:

- `record_chunk()`
- `record_fixed_duration()`
- `write_wav()`

### 2.3 `VadDetector`

責務:

- `webrtcvad` による発話判定

主メソッド候補:

- `is_speech_frame()`
- `has_ended()`

### 2.4 `TranscriptionService`

責務:

- 認識 backend 抽象化

実装候補:

- `LocalWhisperTranscriptionService`
- `RemoteWhisperTranscriptionService`

主メソッド候補:

- `transcribe(wav_path)`

### 2.5 `CorrectionService`

責務:

- `Ollama` 補正
- phrase test 判定

主メソッド候補:

- `correct(text)`
- `judge(expected, recognized)`

### 2.6 `VoicevoxTtsService`

責務:

- 音声クエリ作成
- 音声合成
- wav 再生

主メソッド候補:

- `speak(text, speaker)`

### 2.7 `CommandRouter`

責務:

- 発話からコマンド判定
- shell command 実行

主メソッド候補:

- `match(text)`
- `execute(command_id)`

### 2.8 `EventRepository`

責務:

- JSONL 保存
- SQLite 保存

主メソッド候補:

- `save_event(event_type, payload)`

### 2.9 `PhraseTestRunner`

責務:

- phrase list 読込
- style cycle 管理
- expected / recognized 記録

### 2.10 `AlwaysOnRunner`

責務:

- wake word 監視
- command / chat 分岐

### 2.11 `TimedRecordRunner`

責務:

- 固定秒数録音
- 開始終了アナウンス
- 保存

## 3. 依存関係

- `AlwaysOnRunner` -> `AudioRecorder`, `VadDetector`, `TranscriptionService`, `CorrectionService`, `VoicevoxTtsService`, `CommandRouter`, `EventRepository`
- `PhraseTestRunner` -> `AudioRecorder`, `TranscriptionService`, `CorrectionService`, `VoicevoxTtsService`, `EventRepository`
- `TimedRecordRunner` -> `AudioRecorder`, `TranscriptionService`, `CorrectionService`, `VoicevoxTtsService`, `EventRepository`

## 4. 実装方針

当面は関数ベース実装を維持する。

将来次の条件が揃ったらクラス分割を行う。

- モードがさらに増える
- 非同期ワーカー導入
- systemd 常駐と再試行制御を強化する
- テストコードを本格導入する

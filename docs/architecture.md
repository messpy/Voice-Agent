# voicechat 設計メモ

最終更新: 2026-03-29

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
- `tools/cohere_transcribe_file.py`
  単発の音声ファイル文字起こし
- `tools/cohere_transcribe_mic.py`
  単発のマイク録音→文字起こし
- `tools/cohere_phrase_test.py`
  ずんだもん読み上げ付きの復唱テスト
- `tools/remote/faster_whisper_transcribe.py`
  別 PC 側の単発文字起こし
- `tools/remote/whisperx_transcribe.py`
  WhisperX を使った日本語＋話者分離の単発文字起こし
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
  Raspberry Pi で録音した WAV を別 PC へ `scp` し、`faster-whisper` または WhisperX で認識
- `google`
  Google Speech-to-Text API へ 16kHz モノラル PCM を送り、`tools/google_stt.py` 中の設定で wake/command を判断
- `cohere`
  Raspberry Pi 側で録音と `ffmpeg` 正規化を行い、Cohere API へ multipart 送信する

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

### ローカル Whisper の既定値

2026-03-29 時点の既定運用は次のとおり。

- `wake.whisper_model`
  `ggml-small.bin`
- `whisper.realtime_model`
  `ggml-small.bin`
- `whisper.model`
  `ggml-small.bin`

理由:

- Raspberry Pi 上での体感速度を優先するため
- `small` は `base` より聞き取りが少し安定しやすい
- `medium` の方が精度は高い場面があるが、待受用途では遅延コストが大きい

必要に応じて、単発検証や高精度バッチ用途では `medium` に切り替えて使う。

### ローカル Whisper 前処理

ローカル `whisper.cpp` の前には `ffmpeg` ベースの前処理を挟める。

設定位置:

- `audio_pipeline.whisper_preprocess`

既定値:

- `enabled: true`
- `preset_name: compand_denoise`
- `af: afftdn=nr=14:nf=-30,highpass=f=120,lowpass=f=3800,compand=...`

目的:

- 音楽や環境音が混じる音源で、声の帯域を強めに残す
- `small` で聞き取りやすい波形に寄せる

試験結果の要点:

- `small` は前処理なしより `compand_denoise` の方が良かった
- `medium` では `voice_focus` も有効だった
- ただし総合精度は前処理込みでも `medium` の方が高い場面がある

このため現状は次の方針にしている。

- 常用既定: `small + compand_denoise`
- 高精度検証: `medium + voice_focus` または比較実行で都度選定

比較用ツール:

- `tools/lab/audio_preprocess_compare.py`
  1 つの音声ファイルに複数前処理を掛けて比較する
- `tools/lab/whisper_model_compare.py`
  同じ音声を複数モデルで比較する

追加の wake pipeline として、設定次第で次も使える。

- `audio_pipeline.rnnoise`
  外部コマンドでのノイズ低減
- `Silero VAD`
  wake 窓に発話があるかの前段チェック
- `Porcupine`
  wake word 専用検出

### Cohere 復唱テスト

`tools/cohere_phrase_test.py` は本線から独立した単発テスターとして扱う。

目的:

- ずんだもん音声で課題文を読む
- ユーザーに復唱させる
- 文字起こし結果をその場で確認する
- 誤認識 alias を後から減らせる形で記録する

処理順:

1. `assistant.modes.phrase_test` から課題文を選ぶ
2. `VOICEVOX` で `〇〇と言ってください` を再生する
3. `effects.ready` の効果音を鳴らす
4. `arecord` で録音する
5. `effects.recorded` の効果音を鳴らす
6. まず Cohere API へ送る
7. Cohere が失敗した場合はローカル `whisper.cpp` にフォールバックする
8. 認識結果をずんだもん音声で読み上げる
9. `JSONL` と `SQLite` に結果を保存する

この系統は常時待受ではなく、録音・認識・評価を 1 セッションずつ明示的に回す。

### 効果音

復唱テストと本線は同じ `effects` 設定を共有する。

- `effects.ready`
  ユーザーが話し始める前の開始音
- `effects.recorded`
  録音完了直後の終了音

効果音ファイルが存在すればそれを優先し、無ければ設定値から短いトーンを合成する。

### AI 補正

- ローカル処理時は Pi 上の `Ollama`
- SSH オフロード時は別 PC 上の `Ollama`

`runtime.ai_correction` で有効化する。

単発文字起こしでも同じ補正器を使う。

- `src/transcript_correction.py`
  補正用の共通モジュール
- `tools/transcribe_file_local.py`
  単発ファイルのローカル文字起こし + AI 補正

補正時は次の 3 層を組み合わせる。

1. 生文字起こし
   `whisper.cpp` や Cohere の生結果
2. alias
   既知の誤認識を正解候補に寄せる
3. AI 補正
   `Ollama` で自然な日本語へ整形する

RAG は AI 補正の補助として使う。
主用途は固有名詞、設定名、プロジェクト固有用語の判断補助であり、音の取り違えそのものを直す主役ではない。
たとえば `ペリペリ -> ベリベリ` のような誤認識は、RAG より alias 学習の方が効く。

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

保存先は運用モードごとに異なる。

- 本線運用
  `storage.sqlite.path` で指定した `voicechat.db`
- `cohere_phrase_test`
  `/tmp/voicechat/cohere_transcribe/phrase_test_results.db`

単発テスターでも `SQLite` を使う理由は次のとおり。

- hits を持てる
- 最終更新時刻で並べ替えできる
- 有効/無効を切り替えられる
- 管理 CLI で後から編集できる

`cohere_phrase_test` でも同じ `recognition_aliases` テーブルを使う。

- `alias_type = phrase`
  復唱テスト用の alias
- `target`
  正解文
- `alias`
  誤認識として実際に出た文

復唱テストでは、保存済み alias があれば評価前に正規化する。
たとえば `ペリペリ -> ベリベリ` を登録すると、次回から評価時は `ベリベリ` として扱える。

### alias 学習の流れ

考え方:

- RAG は知識補助
- alias は音の誤認識補正

そのため短いウェイク語、固有名詞、短文コマンドでは alias を優先する。

学習フロー:

1. 認識結果を得る
2. `recognition_aliases` を読む
3. 一致する alias があれば `target` に正規化する
4. 条件を満たした誤認識は `recognition_aliases` に保存する
5. 次回以降は保存済み alias を先に使う

主なカラム:

- `alias_type`
  `wake`, `command`, `phrase` など用途別の分類
- `target`
  正解側の文言
- `alias`
  誤認識側の文言
- `source`
  どこで学習したか
- `hits`
  何回観測されたか
- `last_seen_ts`
  最終更新時刻
- `enabled`
  無効化フラグ

実装位置:

- `tools/wake_vad_record.py`
  本線の alias 読み書き
- `tools/cohere_phrase_test.py`
  復唱テスト用 alias 読み書き
- `tools/admin/recognition_alias_manager.py`
  手動メンテナンス

現状は `SQLite` ベースの学習が主で、固定辞書ファイルは主系統では使っていない。
必要なら将来 `config/transcript_aliases.yaml` のような初期辞書を併用できる余地は残す。

### 補正時の RAG

補正器は `assistant.rag` の設定を流用する。

既定では次を読む。

- `README.md`
- `config/**/*.yaml`
- `logs/**/*.txt`

使い方:

- 補正対象の文と RAG コーパスを n-gram 類似で照合する
- 上位チャンクだけを system prompt に差し込む
- さらに `recognition_aliases` から近い alias 候補も補助文脈として渡す

これにより、補正器は単に文をきれいにするだけでなく、
既存設定にある用語や、過去に学習した読み違いも参考にできる。

ランタイム状態:

- `/tmp/voicechat/runtime_state.json`
  現在の待機状態、使用モデル、最新認識の要約

## 音声コマンド

`command_router.commands` は現在、外部 action IF を前提にする。
コマンドの定義と `help` の読み上げ内容は、この設定を単一ソースにする。

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
- `アラーム一覧`

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

`alarm` の現在仕様:

- `alarm.set`
  1 回限りの pending job を保存する
- `voicechat-actions.scheduler`
  `due_at` を監視し、時刻到来で job を実行する
- アラーム再生
  `music.play` と同じ `music_runner` を `--max-seconds` 付きで起動する
- アラーム停止
  再生停止は `music.stop` と同じ停止経路を使う
- `alarm.status`
  次の 1 件ではなく、保留中アラーム一覧を返せる

この分離により、音声側は orchestrator に寄せられる。
`music.play` は `voicechat-actions` 側で `yt-dlp` を使って候補を展開し、各曲のタイトルを `VOICEVOX` で読み上げてから再生する。

## 起動時依存サービス

起動前に次を確認する。

- `VOICEVOX`
- `Ollama`

`config/config.yaml` の `services.*` に `auto_start: true` があれば、ヘルスチェック失敗時に `start_command` で自動起動する。

現状の Raspberry Pi 実機では `VOICEVOX` を Docker コンテナとして `127.0.0.1:50121` に公開して使う構成も想定している。
この場合、`tts.engine_host` と `services.voicevox.health_url` は同じポートを向く必要がある。

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

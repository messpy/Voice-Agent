# voicechat

Raspberry Pi 上で動かす音声認識・音声合成・音声テスト用の実験環境です。

現在は次の 2 系統を持っています。

- ローカル処理
  Raspberry Pi 上で録音、Whisper 系認識、Ollama 補正まで完結させる
- SSH オフロード処理
  Raspberry Pi で録音した WAV を別 PC に送り、別 PC 側で重いモデルを使って文字起こしと AI 補正を行う

完全ローカル運用を基本にしていますが、重いモデルを回したい場合は SSH オフロードも選べます。

## 現在の主な機能

- `VOICEVOX` のずんだもんによる音声合成
- 会話音声とは別に ready / recorded 効果音を鳴らすコマンドモード
- `whisper.cpp` によるローカル音声認識
- リアルタイム判定用と保存用でモデルを分ける 2 段 STT
- `RNNoise -> Silero VAD -> Porcupine -> Whisper.cpp` に切り替え可能な wake pipeline
- `faster-whisper` による別 PC での音声認識
- `Ollama` / `Gemini` / `OpenAI` / `Anthropic` の切替可能な LLM 補正
- `balanced` / `light_chat` / `precise` の認識モード切替
- `phrase_test` による復唱テスト
- 単語テスト JSON の外出し
- 5 分録音と文字起こし
- 生認識結果と AI 補正結果の両方を保存
- `スタックチャン` をウェイクワードにした音声アシスタント実験
- `スタックチャン音量上げて` のような inline command 起動
- 統合ランナーでの運用モード切替
- `SQLite` と `JSONL` の二重記録
- `recognition_aliases` によるコマンド誤認識 alias 学習
- `tools/admin/show_status.py` による CLI 状態確認
- 起動前の `VOICEVOX` / `Ollama` ヘルスチェックと自動起動

## ディレクトリの考え方

主に使う場所は次です。

- `config/config.yaml`
  全体設定
- `config/config.example.yaml`
  公開用サンプル設定
- `config/phrase_test_words.json`
  単語テスト一覧
- `tools/wake_vad_record.py`
  復唱テスト、ウェイクワード待受、固定時間録音、VOICEVOX 読み上げ、音声コマンドの本線
- `tools/timed_record_transcribe.py`
  `スタート` と `5分終了` つき録音と文字起こし
- `tools/admin/`
  状態確認、alias 管理、履歴検索などの運用補助
- `tools/remote/`
  別 PC 用の単発文字起こしとモデル比較
- `tools/lab/`
  音声・Whisper・VOICEVOX の検証やベンチ
- `tools/admin/show_status.py`
  現在状態、最新認識、設定全文の CLI 表示
- `tools/admin/recognition_alias_manager.py`
  未知コマンド一覧確認と alias 手動登録
- `voicechat.sh`
  よく使う入口をまとめたランチャー
- `legacy/`
  旧世代の単発実験スクリプト
- `docs/architecture.md`
  設計メモ

`tools/` は次の 3 つに分けています。

- `tools/admin`
  運用中の確認と管理
- `tools/remote`
  SSH オフロード関連
- `tools/lab`
  実験、検証、ベンチ

## 公開用設定

公開リポジトリでは実運用の `config/config.yaml` は追跡せず、`config/config.example.yaml` をサンプルとして使います。

初回セットアップ時はサンプルをコピーして使ってください。

## 前提環境

### Raspberry Pi 側

最低限必要です。

- Python 3.12 前後
- `uv`
- `arecord`
- `aplay`
- `VOICEVOX engine`
- `whisper.cpp`
- `Ollama`

仮想環境:

```bash
cd <repo-root>
cp config/config.example.yaml config/config.yaml
uv venv
uv sync
```

### systemd user service

`voicechat` をログイン時に自動起動したい場合は `systemd/voicechat.service` を `~/.config/systemd/user/voicechat.service` に置いて有効化します。

```bash
mkdir -p ~/.config/systemd/user
install -m 644 systemd/voicechat.service ~/.config/systemd/user/voicechat.service
systemctl --user daemon-reload
systemctl --user enable --now voicechat.service
```

### VOICEVOX

本プロジェクトは `open_jtalk` や `piper` を削除済みで、現在の音声合成は `VOICEVOX` 前提です。

設定上の話者は `speaker: 3` で、これは `ずんだもん / ノーマル` です。

確認:

```bash
curl -sS http://127.0.0.1:50021/version
curl -sS http://127.0.0.1:50021/speakers
```

### 別 PC へ SSH オフロードする場合

Raspberry Pi 側から別 PC に公開鍵認証で入れる必要があります。
鍵パス、接続先ホスト、リモート作業ディレクトリは `config/config.yaml` の `remote` にローカル環境ごとの値を入れてください。

## 統合設定

現在は `config/config.yaml` の `runtime` を中心に切り替えます。

- `runtime.run_mode`
  `phrase_test` / `debug_stt` / `always_on` / `timed_record` / `ai_duo`
- `runtime.assistant_mode`
  `always_on` 時に使う会話人格
- `runtime.recognition_mode`
  `balanced` / `light_chat` / `precise`
- `runtime.ai_correction`
  認識結果に AI 補正を入れるか
- `runtime.transcription_backend`
  `local` または `ssh_remote`
- `recognition_profiles`
  wake / realtime / final / AI 補正をモード単位で束ねる
- `services.voicevox`
  `VOICEVOX` の自動起動設定
- `services.ollama`
  `Ollama` の自動起動設定
- `audio_pipeline.rnnoise`
  RNNoise 前処理
- `wake.backend`
  `whisper` または `porcupine`
- `wake.pre_vad_backend`
  wake 前段の発話検出。`webrtc` または `silero`
- `wake.porcupine`
  Porcupine の access key と keyword `.ppn`
- `effects.ready`
  `はい、どうぞ。` の直後に鳴らす入力開始音
- `effects.recorded`
  ユーザー録音直後に鳴らす入力終了音
- `whisper.realtime_model`
  コマンド即応用の軽量モデル
- `llm.provider`
  `ollama` / `gemini` / `openai` / `anthropic`
- `runtime.command_execution`
  音声コマンドを実行するか
- `command_router.action_runners`
  外部リポジトリとの実行 IF
- `command_router.normalization.llm_enabled`
  Ollama を使って、意図が近いコマンドに補正するか
- `alarm.set`
  `7時に起こして` / `7時30分に起こして` / `7時半に起こして` / `明日7時に起こして` を受ける
- `timed_record.seconds`
  固定録音秒数
- `storage.jsonl.path`
  JSONL の保存先
- `storage.sqlite.path`
  SQLite の保存先
- `assistant.memory_search`
  長期会話検索

基本の起動コマンドはこれです。

```bash
cd <repo-root>
./voicechat.sh run
```

起動時は先に `services.voicevox` と `services.ollama` を確認し、未起動なら `start_command` で自動起動してから本体に入ります。

`voicechat.sh` は `.env` / `.env.local` を読んだあと、使える Python 実行環境を自動解決します。既定は `auto` で、必要なら次を使えます。

- `VOICECHAT_PYTHON`
  明示的に使う Python 実体
- `VOICECHAT_RUNNER`
  `auto` / `python` / `uv`
- `VOICECHAT_REQUIREMENTS_CHECK`
  起動前 import チェックをするか

`.env.example` にサンプルを置いてあります。

## 追加の wake pipeline

既定値は従来どおり `whisper` wake です。Porcupine を使う場合は `uv sync --extra wake` で追加依存を入れたうえで、`config/config.yaml` をこう切り替えます。

```yaml
audio_pipeline:
  rnnoise:
    enabled: true
    command:
      - /path/to/rnnoise_wrapper
      - "{input}"
      - "{output}"

wake:
  backend: porcupine
  pre_vad_backend: silero
  porcupine:
    access_key_env: PORCUPINE_ACCESS_KEY
    keyword_paths:
      スタックチャン: /path/to/stackchan_ja_raspberry-pi.ppn

vad:
  silero:
    threshold: 0.5
```

このときの流れは次です。

- RNNoise で wake 窓と本認識音声を前処理
- Silero VAD で wake 窓に発話があるか確認
- Porcupine で wake word 判定
- wake 後の本文は `whisper.cpp`

最低限必要なのは次です。

```bash
cd <repo-root>
cp .env.example .env
```

`.env` に `PORCUPINE_ACCESS_KEY` を入れてください。
同梱の `assets/porcupine/alexa_raspberry-pi.ppn` があるので、key が入れば `アレクサ` は先に Porcupine で試せます。
`スタックチャン` と `ずんだもん` は専用 `.ppn` を追加で置く必要があります。

## コマンド IF

外部リポジトリに機能を分離したい場合は、音声側は次の IF で action を渡します。

- 入力
  `action_type`, `action_name`, `args`
- 出力
  `success`, `message`, `data`

`command_router.action_runners` に runner を定義し、各コマンドは `action_type`, `action_name`, `args` だけを持ちます。
`external_cli` では runner に JSON を stdin で渡し、JSON を stdout で返す前提です。

推奨構成として、外部 backend は sibling project の `voicechat-actions` に分離できます。
`timer`, `alarm`, `light`, `music`, `audio` などの実装をそちらへ寄せると、`voicechat` 本体は聞き取りと判定に集中できます。
`music.play` は `voicechat-actions` 側で `yt-dlp` による再生前 probe を行い、曲ごとにタイトルを `VOICEVOX` で読み上げてから再生します。
`music.next` を command に追加すると、再生中の曲を止めて次の候補へ進められます。
`alarm` は scheduler が時刻到来を監視し、到来時は `music.play` と同じ `music_runner` を `--max-seconds` 付きで起動します。
そのため、アラーム再生の停止も通常の YouTube 再生停止と同じ経路に寄せられます。
現在の `alarm.set` は 1 回限りで、時刻到来後は `done` になり、自動で翌日に再登録はしません。
`alarm.status` は次の 1 件だけでなく、保留中アラーム一覧を返します。
親ディレクトリからは `voicechat` / `voicechat-actions` / `natureremo` を sibling に置く前提です。

## STT の使い分け

現在の `always_on` 系では、ローカル認識時にモードごとに使い分けます。

- `wake.whisper_model`
  ウェイクワード専用
- `whisper.realtime_model`
  コマンド判定や即応用
- `whisper.model`
  保存用、あとで見返す文字起こし用

現在は `recognition_profiles` で各モードをまとめています。

- `balanced`
  `small -> small`、AI 補正なし
- `light_chat`
  `tiny -> base`、AI 補正あり
- `precise`
  `small -> medium`、AI 補正あり

たとえば現在は次の使い分けです。

```yaml
runtime:
  recognition_mode: balanced

recognition_profiles:
  balanced:
    realtime_model: /tmp/voicechat-whisper/ggml-small.bin
    final_model: /tmp/voicechat-whisper/ggml-small.bin
    wake_model: /tmp/voicechat-whisper/ggml-small.bin
  light_chat:
    realtime_model: /tmp/voicechat-whisper/ggml-tiny.bin
    final_model: /tmp/voicechat-whisper/ggml-base.bin
    wake_model: /tmp/voicechat-whisper/ggml-base.bin
  precise:
    realtime_model: /tmp/voicechat-whisper/ggml-small.bin
    final_model: /tmp/voicechat-whisper/ggml-medium.bin
    wake_model: /tmp/voicechat-whisper/ggml-medium.bin
```

イベントログには次も残します。

- `recognized_fast`
- `recognized_final`
- `model_fast`
- `model_final`

## 状態確認

CLI から現在状態を確認できます。

```bash
cd <repo-root>
./voicechat.sh status
```

例:

```bash
./voicechat.sh status runtime
./voicechat.sh status config wake.backend
./voicechat.sh status events --limit 20 --types wake_check command command_unknown
./voicechat.sh status conversation --limit 10
./voicechat.sh commands --limit 20
./voicechat.sh failures --limit 20
```

`status events` では、`command` の行に `fast=...` `final=...` `cmd=...` `ok=...` が出ます。
`commands` は実行されたコマンドだけを `発話 -> command_id -> ok` で見られます。
`failures` は wake の未ヒットと、失敗した command / unknown をまとめて見られます。

表示内容:

- 起動中かどうか
- 現在の状態
  `wake_wait` / `command_retry_wait` / `recording` / `transcribing_realtime` / `transcribing_final` など
- 最新認識イベント
- 現在の設定全文

ランタイム状態は `paths.workdir/runtime_state.json` に保存します。現在の既定は `/home/kennypi/work/voicechat/.runtime/runtime_state.json` です。

## 誤認識 alias 学習

コマンド誤認識は SQLite の `recognition_aliases` テーブルに学習できます。

未知コマンド一覧:

```bash
cd <repo-root>
./voicechat.sh aliases unknowns --limit 20
```

登録済みコマンド一覧:

```bash
cd <repo-root>
./voicechat.sh aliases commands
```

未知コマンドのイベント ID から alias 登録:

```bash
cd <repo-root>
./voicechat.sh aliases add-command-alias --event-id 712 --command-id help
```

文字列を直接 alias 登録:

```bash
cd <repo-root>
./voicechat.sh aliases add-command-alias --alias "気分がもちろん" --command-id help
```

ウェイク語 alias の手動登録:

```bash
cd <repo-root>
./voicechat.sh aliases add-wake-alias --target アレクサ --alias "あれくさい"
```

## LLM プロバイダ

現在の補正・会話系 API は `Ollama` 固定ではなく、`Ollama` / `Gemini` / `OpenAI` / `Anthropic` を切り替えられます。

### Ollama

```yaml
llm:
  provider: ollama
  model: qwen2.5:7b
  host: http://127.0.0.1:11434
  timeout_sec: 120
  api_key_env: OLLAMA_API_KEY
  web_search:
    enabled: false
    max_results: 5
```

`host` を `https://ollama.com` に寄せて `OLLAMA_API_KEY` を入れれば、Ollama Cloud 側の利用もできます。`web_search.enabled: true` のときは、Ollama の Web 検索結果を会話前に補助コンテキストとして注入します。

### Gemini

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
  timeout_sec: 120
  api_key_env: GEMINI_API_KEY
  api_base: https://generativelanguage.googleapis.com/v1beta
```

実行前に環境変数を入れます。

```bash
export GEMINI_API_KEY=your_api_key_here
```

### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-5-mini
  timeout_sec: 120
  api_key_env: OPENAI_API_KEY
  api_base: https://api.openai.com/v1
```

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Anthropic

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  timeout_sec: 120
  api_key_env: ANTHROPIC_API_KEY
  api_base: https://api.anthropic.com/v1
  anthropic_version: 2023-06-01
```

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## ローカル処理と SSH オフロード処理の違い

### ローカル処理

向いている用途:

- 常時待受
- 軽いモデルでの 24 時間運用
- スタックチャン起点の音声リモコン
- 復唱テスト

利点:

- 配線が単純
- ネットワーク転送が不要
- 常時運用しやすい

欠点:

- 重いモデルは遅い
- 長い録音の認識に時間がかかる

### SSH オフロード処理

向いている用途:

- 重めのモデルを試す
- モデル比較
- ラズパイの CPU 負荷を下げたい

利点:

- Raspberry Pi で重い推論をしなくてよい
- `faster-whisper` 系モデルを別 PC に集約できる

欠点:

- SSH、Python 環境、モデルキャッシュの準備が必要
- 別 PC が弱いと大差ない
- 24 時間常時運用は構成が少し複雑になる

## 主要な使い方

### 1. ずんだもん全スタイル確認

```bash
cd <repo-root>
./voicechat.sh lab-zundamon
```

読み上げ対象:

- `3 ノーマル`
- `1 あまあま`
- `7 ツンツン`
- `5 セクシー`
- `22 ささやき`
- `38 ヒソヒソ`
- `75 ヘロヘロ`
- `76 なみだめ`

### 2. 復唱テスト

```bash
cd <repo-root>
./voicechat.sh run
```

現在の `phrase_test` は次の設定で動きます。

- ずんだもんが `3、2、1。〇〇、と言ってください` と読む
- 単語一覧は `config/phrase_test_words.json`
- ずんだもんのスタイルを順番に回す
- ログに `phrase_id`、`phrase_note`、`prompt_style`、`expected`、`recognized`、`elapsed_sec` を保存
- 同時に `events.jsonl` と `voicechat.db` にも保存

### 3. 5分録音と文字起こし

統合ランナーでやる場合は `config/config.yaml` をこうします。

```yaml
runtime:
  run_mode: timed_record
  ai_correction: true
  transcription_backend: local

timed_record:
  seconds: 300
```

そのうえで次を実行します。

```bash
cd <repo-root>
./.venv/bin/python -m tools.wake_vad_record
```

流れ:

- ずんだもんで `スタート`
- 指定秒数録音
- ずんだもんで `5分終了`
- 生文字起こし
- 必要なら AI 補正
- `JSONL` と `SQLite` に保存

同期実行:

```bash
cd <repo-root>
./voicechat.sh timed-record --seconds 300
```

流れ:

- ずんだもんで `スタート`
- 300 秒録音
- ずんだもんで `5分終了`
- 生文字起こし
- AI 補正

バックグラウンド実行:

```bash
cd <repo-root>
./voicechat.sh timed-record --seconds 300 --background
```

出力先:

- `paths.workdir/timed_record_<tag>.wav`
- `paths.workdir/timed_record_<tag>_raw.txt`
- `paths.workdir/timed_record_<tag>_corrected.txt`
- `paths.workdir/timed_record_<tag>.json`
- `paths.workdir/timed_record_<tag>_bg.log`

### 4. 常時待受と音声リモコン

`always_on` では、`スタックチャン` を待ち受けてから命令文を認識し、登録済みコマンドなら即実行します。
現在の既定 wake 語は `スタックチャン` と `アレクサ` です。

```yaml
runtime:
  run_mode: always_on
  assistant_mode: zundamon
  recognition_mode: balanced
  command_execution: true
```

既定の `balanced` は `small` ベースです。会話寄りに速くしたいときは `light_chat`、精度優先なら `precise` へ切り替えます。

起動:

```bash
cd <repo-root>
./voicechat.sh run
```

実行の流れ:

1. `スタックチャン` と言う
2. ずんだもんが `はい、どうぞ。` と返す
3. そのあとにコマンドを言う

実行例:

- `スタックチャン`
- `音量あげて`

または inline command:

- `スタックチャン音量あげて`
- `スタックチャン 音楽起動して`

または

- `スタックチャン`
- `音楽とめて`

または

- `スタックチャン`
- `この話したのいつだっけ`

お話モードに入る例:

- `スタックチャン お話モード`
- `スタックチャン 会話モード`

使えるコマンド一覧は `config/config.yaml` の `command_router.commands` を単一ソースにしています。
`help` / `ヘルプ` で読まれる内容も、この設定から自動生成されます。

認識モードの切替コマンド:

- `スタックチャン 認識モード`
- `スタックチャン 認識モード標準`
- `スタックチャン 認識モード軽量`
- `スタックチャン 認識モード高精度`

終了したいときの例:

- `スタックチャン`
- `終了`

補足:

- command mode では、既知コマンド候補を前段で検索してから正規化します
- つまり `音量あげて` や `音楽とめて` のような短い命令文は、自由会話よりコマンド候補に寄せて解釈します
- 候補一覧と正規化後の文もログへ保存します

実行内容と `help` の読み上げ内容は `command_router.commands` 側で管理します。

### 5. 長期会話検索

`assistant.memory_search` を有効にすると、`voicechat.db` に保存済みの過去イベントから関連会話を引いて、会話コンテキストへ自動で差し込みます。

これで次のような質問を拾いやすくなります。

- `この話したのいつだっけ`
- `前に電気消してって言ったのいつ`
- `この前の音楽起動の話いつしたっけ`

設定例:

```yaml
assistant:
  memory_search:
    enabled: true
    top_k: 5
    min_score: 0.18
    scan_limit: 2000
    lookback_turns: 3
```

補足:

- `WAKECHK` の結果も `events.jsonl` と `voicechat.db` に保存されます
- つまり `スタックチャン` の誤認識や、何と聞こえていたかも後から追えます

### 6. 別 PC で単発文字起こし

別 PC 側で一度だけ環境構築:

```bash
python3 -m venv ~/sandbox/voicechat_remote/.venv
~/sandbox/voicechat_remote/.venv/bin/pip install --upgrade pip setuptools wheel faster-whisper requests
```

Raspberry Pi からスクリプト転送:

```bash
scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  ./voicechat.sh remote-transcribe \
  your-user@your-remote-host:/home/your-user/voicechat_remote/
```

別 PC 側で実行:

```bash
cd ~/sandbox/voicechat_remote
./.venv/bin/python remote_faster_whisper_transcribe.py \
  --wav /path/to/input.wav \
  --out-prefix /path/to/out/test1 \
  --model large-v3 \
  --device cpu \
  --compute-type int8
```

### 7. 別 PC で複数モデル比較

```bash
cd ~/sandbox/voicechat_remote
./voicechat.sh remote-bench \
  --wav /path/to/input.wav \
  --out-dir /path/to/bench_out \
  --models small medium distil-large-v3 large-v3 \
  --device cpu \
  --compute-type int8 \
  --skip-correct
```

出力:

- `small.json`
- `medium.json`
- `distil-large-v3.json`
- `large-v3.json`
- `summary.json`

### 8. WhisperX で日本語＋話者分離の高精度文字起こし

WhisperX の CLI を別 PC に入れておくと、日本語かつ話者ラベル付きの文字起こしを Raspberry Pi から offload できます。先に環境を作っておき、`config/config.yaml` の `remote.remote_script` を `/home/your-user/voicechat_remote/remote_whisperx_transcribe.py` に置き換えると `ssh_remote` の文字起こしでも WhisperX が使われます。

```bash
/opt/homebrew/bin/python3.11 -m venv ~/venvs/whisperx
source ~/venvs/whisperx/bin/activate
pip install -U pip wheel setuptools
pip install torch==2.2.2 torchaudio==2.2.2
pip install "git+https://github.com/m-bain/whisperX.git" --no-deps
pip install numpy pandas==2.2.3 onnxruntime nltk==3.9.1 transformers==4.48.0 faster-whisper==1.1.1 ctranslate2==4.4.0 pyannote.audio==3.3.2
```

依存が整ったら `voicechat.sh remote-whisperx` で new script を呼び出します。

```bash
cd ~/sandbox/voicechat_remote
./voicechat.sh remote-whisperx \
  --wav /path/to/input.wav \
  --out-prefix /path/to/whisperx/out \
  --model large \
  --device cpu \
  --compute-type int8 \
  --language Japanese \
  --diarize \
  --min-speakers 2 \
  --max-speakers 4
```

`--skip-correct` を付けると AI 補正を飛ばせます。`--diarize` を入れると話者ラベル（`SPEAKER_00` など）が付くので、`--min-speakers`/`--max-speakers` で調整してください。

主要パラメータ:

| パラメータ | 説明 | 推奨値 / 備考 |
| --- | --- | --- |
| `audio_file` | 入力音声パス (`.wav`, `.mp3`, `.flac` 等) | `--wav` で指定 |
| `--model` | 倍率（`tiny`～`large`） | 日本語精度重視は `large` |
| `--compute_type` | 計算精度 | GPU: `float16`、CPU: `int8`/`float32` |
| `--device` | 使用デバイス | `cpu`, `cuda`, `mps` |
| `--output_dir` | 出力先ディレクトリ | `--out-prefix` の親を指定 |
| `--output_format` | 出力形式 | 複数指定可 (`txt,srt`) |
| `--diarize` | 話者分離を有効化 | `--diarize` フラグ |
| `--min_speakers` / `--max_speakers` | 話者数の範囲 | 例: `--min 2 --max 4` |
| `--align_model` | タイムスタンプ補正モデル | 自動選択を上書き |
| `--vad_onset` / `--vad_offset` | 無音判定閾値 | 音声の切れ目がズレるとき |
 | `--batch_size` | 一度に送るチャンク | GPU に余裕があれば大きめに |

このスクリプトは `raw`/`corrected`/`meta` ファイルを `--out-prefix` から生成し、`wake_vad_record` の `ssh_remote` ルートから `raw_text`, `corrected_text`, `elapsed_sec` を返します。

### 9. Google Speech-to-Text について

コードには `google-cloud-speech` 系の補助ツールを残していますが、現在の常用構成は Google 非依存です。

理由:

- 無料枠を超えると課金になる
- 常時待受ではローカルだけで完結した方が運用しやすい
- 現在の `voicechat` 既定設定は `local` + `whisper` wake で調整しています

そのため README の推奨構成は、`Google STT` ではなく `recognition_profiles` とローカル Whisper の組み合わせです。

### 10. Cohere 文字起こしツール

Raspberry Pi を録音と前処理に使い、文字起こし本体は Cohere API に送る最小構成を追加しています。現時点では既存の常駐ランナーとは切り離し、単発ツールとして使う前提です。

準備:

- `.env.example` をコピーして `.env` を作り、`COHERE_API_KEY` を入れる
- `ffmpeg` と `arecord` が使えることを確認する
- 必要なら `config/config.yaml` の `cohere_transcribe` で `api_url`, `model`, `language`, `output_dir` を調整する

ファイル文字起こし:

```bash
cd <repo-root>
./voicechat.sh cohere-file path/to/input.wav --language ja
```

マイク録音して文字起こし:

```bash
cd <repo-root>
./voicechat.sh cohere-mic --seconds 10 --device plughw:CARD=Device,DEV=0 --language ja
```

ずんだもんが課題文を読んで復唱テスト:

```bash
cd <repo-root>
./voicechat.sh cohere-phrase-test --seconds 8 --device plughw:CARD=Device,DEV=0
```

このモードでは `assistant.modes.phrase_test` の課題文を 1 つ選び、VOICEVOX で「〇〇と言ってください」と読み上げてから録音し、Cohere で文字起こしして一致率を出します。

どちらも次を行います。

- `ffmpeg` で 16kHz / mono / PCM WAV に正規化
- Cohere API に multipart で送信
- 文字起こし結果を標準出力へ表示
- `output_dir` に `.txt` と `.json` を保存

主な追加ファイル:

- `tools/cohere_transcribe.py`
  共通処理。録音、`ffmpeg` 変換、API 呼び出し、保存を担当
- `tools/cohere_transcribe_file.py`
  音声ファイル 1 本を送る最小 CLI
- `tools/cohere_transcribe_mic.py`
  USB マイクなどを `arecord` で録音してから送る CLI

## 設定ファイルの見方

主に触るのは `config/config.yaml` です。

よく使う項目:

- `audio.input`
- `audio.output`
- `wake.word`
- `runtime.recognition_mode`
- `recognition_profiles`
- `vad.silence_ms`
- `vad.max_record_sec`
- `whisper.model`
- `whisper.threads`
- `assistant.active_mode`
- `runtime.run_mode`
- `runtime.ai_correction`
- `runtime.transcription_backend`
- `storage.sqlite.path`
- `assistant.modes.phrase_test.phrases_file`
- `assistant.modes.phrase_test.style_cycle`
- `tts.speaker`
- `tts.voicevox_volume_scale`
- `command_router.commands`

## 単語テストの管理

単語テストは `config/phrase_test_words.json` で管理します。

例:

```json
{
  "phrases": [
    {
      "id": "wake_only",
      "text": "スタックチャン",
      "note": "ウェイクワード単体"
    },
    {
      "id": "music_stop",
      "text": "音楽とめて",
      "note": "音量との混線確認"
    }
  ]
}
```

この形式にしておくと、ログを見返したときに何のテストだったか分かります。

## 現時点の運用方針

24 時間 365 日で使うなら、重いモデルのリアルタイム同期処理より、次の方が向いています。

- 録音は軽く
- 区間を短く切る
- `small` で一次認識
- 必要なら後段で再認識
- `raw` と `corrected` を両方保存

つまり、常時運用はローカル軽量、比較や高品質再処理は別 PC という切り分けが現実的です。

## 注意点

- YouTube などの音と自分の声が同じマイクに混ざると、どのモデルでも大きく崩れやすいです
- AI 補正は自然文に寄せるので、記録用途では `raw` を正として扱う方が安全です
- 別 PC オフロードは、オフロード先の CPU が弱いと期待ほど速くなりません

## 現在の主な出力ファイル

`paths.workdir` 配下に次のようなファイルができます。現在の既定は `/home/kennypi/work/voicechat/.runtime/` です。

- `*_raw.txt`
- `*_corrected.txt`
- `*.json`
- `*.wav`
- `*_bg.log`
- `events.jsonl`
- `voicechat.db`

文字起こしの保存形式は、最低限次を持つようにしています。

- `date`
- `model`
- `expected`
- `recognized`
- `elapsed_sec`

## 作者メモ

- `whisper.cpp` の出力は `stdout` ではなく `-otxt` のファイルを正とする
- `VOICEVOX` は Raspberry Pi 側で常駐起動して使う
- SSH オフロードは構成が増えるが、重いモデルの比較には便利

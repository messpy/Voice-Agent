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
- `whisper.cpp` によるローカル音声認識
- リアルタイム判定用と保存用でモデルを分ける 2 段 STT
- `faster-whisper` による別 PC での音声認識
- `Ollama` / `Gemini` / `OpenAI` / `Anthropic` の切替可能な LLM 補正
- `phrase_test` による復唱テスト
- 単語テスト JSON の外出し
- 5 分録音と文字起こし
- 生認識結果と AI 補正結果の両方を保存
- `ベリベリ` をウェイクワードにした音声アシスタント実験
- 統合ランナーでの運用モード切替
- `SQLite` と `JSONL` の二重記録
- `recognition_aliases` によるコマンド誤認識 alias 学習
- `tools/show_status.py` による CLI 状態確認
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
- `tools/zundamon_style_check.py`
  ずんだもん全スタイル読み上げ確認
- `tools/remote_faster_whisper_transcribe.py`
  別 PC 用の単発文字起こし
- `tools/remote_faster_whisper_bench.py`
  別 PC 用の複数モデル比較
- `tools/show_status.py`
  現在状態、最新認識、設定全文の CLI 表示
- `tools/recognition_alias_manager.py`
  未知コマンド一覧確認と alias 手動登録
- `docs/architecture.md`
  設計メモ

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
- `runtime.ai_correction`
  認識結果に AI 補正を入れるか
- `runtime.transcription_backend`
  `local` または `ssh_remote`
- `services.voicevox`
  `VOICEVOX` の自動起動設定
- `services.ollama`
  `Ollama` の自動起動設定
- `whisper.realtime_model`
  コマンド即応用の軽量モデル
- `llm.provider`
  `ollama` / `gemini` / `openai` / `anthropic`
- `runtime.command_execution`
  音声コマンドを実行するか
- `command_router.action_runners`
  外部リポジトリとの実行 IF
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
./.venv/bin/python -m tools.wake_vad_record
```

起動時は先に `services.voicevox` と `services.ollama` を確認し、未起動なら `start_command` で自動起動してから本体に入ります。

## コマンド IF

外部リポジトリに機能を分離したい場合は、音声側は次の IF で action を渡します。

- 入力
  `action_type`, `action_name`, `args`
- 出力
  `success`, `message`, `data`

`command_router.action_runners` に runner を定義し、各コマンドは `action_type`, `action_name`, `args` だけを持ちます。
`external_cli` では runner に JSON を stdin で渡し、JSON を stdout で返す前提です。

## STT の使い分け

現在の `always_on` 系では、ローカル認識時に 2 段で使い分けます。

- `wake.whisper_model`
  ウェイクワード専用
- `whisper.realtime_model`
  コマンド判定や即応用
- `whisper.model`
  保存用、あとで見返す文字起こし用

たとえば現在は次の使い分けです。

```yaml
wake:
  whisper_model: ./whisper.cpp/models/ggml-base.bin

whisper:
  realtime_model: ./whisper.cpp/models/ggml-base.bin
  model: ./whisper.cpp/models/ggml-medium.bin
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
./.venv/bin/python tools/show_status.py
```

表示内容:

- 起動中かどうか
- 現在の状態
  `wake_wait` / `command_retry_wait` / `recording` / `transcribing_realtime` / `transcribing_final` など
- 最新認識イベント
- 現在の設定全文

ランタイム状態は `/tmp/voicechat/runtime_state.json` に保存します。

## 誤認識 alias 学習

コマンド誤認識は SQLite の `recognition_aliases` テーブルに学習できます。

未知コマンド一覧:

```bash
cd <repo-root>
./.venv/bin/python tools/recognition_alias_manager.py unknowns --limit 20
```

登録済みコマンド一覧:

```bash
cd <repo-root>
./.venv/bin/python tools/recognition_alias_manager.py commands
```

未知コマンドのイベント ID から alias 登録:

```bash
cd <repo-root>
./.venv/bin/python tools/recognition_alias_manager.py add-command-alias --event-id 712 --command-id help
```

文字列を直接 alias 登録:

```bash
cd <repo-root>
./.venv/bin/python tools/recognition_alias_manager.py add-command-alias --alias "気分がもちろん" --command-id help
```

ウェイク語 alias の手動登録:

```bash
cd <repo-root>
./.venv/bin/python tools/recognition_alias_manager.py add-wake-alias --target アレクサ --alias "あれくさい"
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
- ベリベリ起点の音声リモコン
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
./.venv/bin/python tools/zundamon_style_check.py
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
./.venv/bin/python -m tools.wake_vad_record
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
./.venv/bin/python tools/timed_record_transcribe.py --seconds 300
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
./.venv/bin/python tools/timed_record_transcribe.py --seconds 300 --background
```

出力先:

- `/tmp/voicechat/timed_record_<tag>.wav`
- `/tmp/voicechat/timed_record_<tag>_raw.txt`
- `/tmp/voicechat/timed_record_<tag>_corrected.txt`
- `/tmp/voicechat/timed_record_<tag>.json`
- `/tmp/voicechat/timed_record_<tag>_bg.log`

### 4. 常時待受と音声リモコン

`always_on` では、`ベリベリ` を待ち受けてから命令文を認識し、登録済みコマンドなら即実行します。
いまは `ずんだもん` と `アレクサ` でも起動できます。

```yaml
runtime:
  run_mode: always_on
  assistant_mode: zundamon
  ai_correction: true
  transcription_backend: local
  command_execution: true
```

現在の既定値では、ウェイクワード検出は速度と精度のバランスのため `ggml-base.bin` と `window_sec: 2.0` を使います。起動後の本認識は別モデルのままです。

起動:

```bash
cd <repo-root>
./.venv/bin/python -m tools.wake_vad_record
```

実行の流れ:

1. `ベリベリ` と言う
2. ずんだもんが `はい、どうぞ。` と返す
3. そのあとにコマンドを言う

または

1. `ずんだもん` と言う
2. ずんだもんが `はい、どうぞ。` と返す
3. そのあとにコマンドを言う

または

1. `アレクサ` と言う
2. ずんだもんが `はい、どうぞ。` と返す
3. そのあとにコマンドを言う

実行例:

- `ベリベリ`
- `音量あげて`

または

- `ベリベリ`
- `音楽とめて`

または

- `ベリベリ`
- `この話したのいつだっけ`

初期実装で入っているコマンド:

- `音量あげて`
- `音量下げて`
- `音楽とめて`
- `音楽起動して`
- `ベリベリ音楽起動して`
- `help`
- `ヘルプ`
- `使い方`
- `できること`
- `終了`
- `停止`
- `終わり`
- `直近の文字起こし`
- `今日の文字起こし`
- `aiをollamaにして`
- `aiをgeminiにして`
- `aiをopenaiにして`
- `aiをanthropicにして`
- `ollama pull qwen`

終了したいときの例:

- `ベリベリ`
- `終了`

補足:

- command mode では、既知コマンド候補を前段で検索してから正規化します
- つまり `音量あげて` や `音楽とめて` のような短い命令文は、自由会話よりコマンド候補に寄せて解釈します
- 候補一覧と正規化後の文もログへ保存します

実行内容は `command_router.commands` の `command` 配列で差し替えられます。

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
- つまり `ベリベリ` の誤認識や、何と聞こえていたかも後から追えます

### 6. 別 PC で単発文字起こし

別 PC 側で一度だけ環境構築:

```bash
python3 -m venv ~/sandbox/voicechat_remote/.venv
~/sandbox/voicechat_remote/.venv/bin/pip install --upgrade pip setuptools wheel faster-whisper requests
```

Raspberry Pi からスクリプト転送:

```bash
scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  ./tools/remote_faster_whisper_transcribe.py \
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
./.venv/bin/python remote_faster_whisper_bench.py \
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

## 設定ファイルの見方

主に触るのは `config/config.yaml` です。

よく使う項目:

- `audio.input`
- `audio.output`
- `wake.word`
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
      "text": "ベリベリ",
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

`/tmp/voicechat/` に次のようなファイルができます。

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

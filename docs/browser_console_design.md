# voicechat ブラウザ管理コンソール設計

最終更新: 2026-05-04

## 目的

`voicechat` をブラウザから確認・操作できる管理コンソールを追加する。

欲しい機能は次の 3 本柱とする。

- 設定の閲覧と編集
- コマンド履歴と実行結果の確認
- ブラウザからのコマンド送信

既存の音声待受本線は崩さず、`runtime_state.json`、`events`、`recognition_aliases`、既存コマンド実行経路を再利用する。

## スコープ

初期スコープ:

- 現在状態の表示
- 直近コマンド履歴の表示
- `config/config.yaml` の閲覧と編集
- ブラウザからのテキストコマンド送信
- 実行結果の表示
- 簡易な alias 管理の下地

初期スコープ外:

- ブラウザからのマイク録音送信
- 音声ストリーミング
- 複数端末の同時制御
- 本格的な認証基盤
- リアルタイム双方向同期の作り込み

## 要求

機能要求:

- `voicechat` の現在状態を見たい
- 最近の command / wake / failure を見たい
- 実際の `config/config.yaml` を編集したい
- ブラウザから `音楽とめて` や `次の曲` のようなコマンドを送りたい
- 音声本線と同じ command router / action runner を使いたい
- 後から alias 管理や簡易ダッシュボードを足せる構造にしたい

非機能要求:

- Raspberry Pi 上で無理なく動く
- 既存の `always_on` 常駐プロセスと疎結合
- 既存ログ形式を壊さない
- UI が止まっても音声本線には影響しない
- 追加依存は最小限に抑える

## 前提

既存 `voicechat` には次がすでにある。

- 実行状態:
  `paths.workdir/runtime_state.json`
- イベントログ:
  `storage.sqlite.path` の `events`
- alias 学習:
  `recognition_aliases`
- コマンド定義:
  `command_router.commands`
- コマンド実行:
  `execute_command_action()` -> `execute_action_runner()`

つまり、ブラウザ版が新しく持つべき責務は「HTTP 入口」と「画面」だけでよい。

## 推奨構成

### 方針

別プロセスの `voicechat-console` を追加する。

- 音声本線:
  `tools/wake_vad_record.py`
- 管理コンソール:
  `voicechat_console` または `tools/web_console.py`

ブラウザ版は既存 DB と state file を読む。
コマンド送信時だけ既存 command 実行ロジックを呼ぶ。

### 理由

- 既存の常駐ループに HTTP サーバを混ぜない方が安全
- UI 側の再起動や改修が音声本線に波及しにくい
- systemd 的にも別 service にしやすい

## 構成案

### 1. Backend

Python の軽量 HTTP サーバを 1 つ追加する。

推奨:

- `FastAPI`
- `uvicorn`

理由:

- JSON API を素直に書ける
- 将来 SSE を足しやすい
- 型付き request/response を整理しやすい

依存をさらに減らしたい場合の代替:

- `Flask`
- 標準ライブラリ + 自前 HTTP

ただし、このリポジトリでは API と静的 UI を分けて保守する都合上、`FastAPI` が最も扱いやすい。

### 2. Frontend

初期版は build step を重くしない。

推奨:

- サーバ配信の静的 HTML
- 素の TypeScript なしの JavaScript
- 小さな CSS

初期版で不要:

- Next.js
- Vite + React の重い構成

理由:

- まず必要なのは管理 UI であり、大規模 SPA ではない
- Pi 上での依存と起動時間を増やしたくない
- API 設計が固まる前にフロント基盤を重くしない方がよい

ただし API は SPA 化しやすい形で設計しておく。

### 3. データソース

- `runtime_state.json`
  現在状態の表示
- `voicechat.db.events`
  履歴と失敗一覧
- `voicechat.db.recognition_aliases`
  alias 一覧と編集
- `config/config.yaml`
  設定表示・保存

## ディレクトリ案

```text
voicechat/
  docs/
    browser_console_design.md
  src/
    voicechat_console/
      __init__.py
      app.py
      config_io.py
      runtime_reader.py
      event_store.py
      command_bridge.py
      models.py
      static/
        index.html
        app.js
        app.css
  systemd/
    voicechat-console.service
```

## 責務分離

### `runtime_reader.py`

- `runtime_state.json` を読む
- state file が壊れていても UI 全体は落とさない
- `show_status.py` と近い整形を返す

### `event_store.py`

- `events` テーブルの参照
- `recognition_aliases` の参照と更新
- 一覧 API 用の query を持つ

### `config_io.py`

- `config/config.yaml` の読み込み
- 保存前バックアップ
- バリデーション
- 部分更新か全文更新かの制御

### `command_bridge.py`

- ブラウザ入力テキストを command に解決
- 既存の `match_command()`、`resolve_playback_context_command()`、`execute_command_action()` を再利用
- command event を `events` に保存
- 返答文を UI に返す

この部分が最重要で、音声本線とブラウザ版で command 実行経路をなるべく共通化する。

## 画面設計

### 1. ダッシュボード

表示項目:

- 実行中か
- PID
- `state`
- `active_mode`
- `transcription_backend`
- `recognition_mode`
- `wake_word`
- `llm_provider / llm_model`
- `recognized_fast`
- `recognized_final`
- `last_reply`
- `last_command`
- `updated_at`

操作:

- 手動更新
- 自動更新 on/off

### 2. コマンドコンソール

表示項目:

- 入力欄
- 送信ボタン
- 直近の送信結果
- 実行ログ一覧

送れるもの:

- `次の曲`
- `音楽とめて`
- `音量上げて`
- `ニュース読んで`

動作:

1. ブラウザでテキスト入力
2. backend が command 解決
3. 既存 action runner を実行
4. `command` event を保存
5. reply / stdout / stderr / success を返す

### 3. 履歴ビュー

タブ:

- `Commands`
- `Wake`
- `Failures`
- `Unknown`

表示項目:

- date
- recognized_fast
- recognized_final
- command_id
- ok
- reply
- stderr

### 4. 設定ビュー

初期版は 2 段構成にする。

- フォーム編集
  よく使う主要設定だけ
- YAML 直接編集
  上級者向け

フォームで触る対象:

- `runtime.run_mode`
- `runtime.assistant_mode`
- `runtime.recognition_mode`
- `runtime.ai_correction`
- `runtime.transcription_backend`
- `runtime.command_execution`
- `wake.backend`
- `wake.pre_vad_backend`
- `llm.provider`
- `llm.model`

YAML 編集では:

- 現在値の表示
- 保存
- バックアップ作成
- 差分表示

### 5. Alias 管理

初期版では read-only でもよいが、設計上は入れておく。

表示項目:

- alias_type
- target
- alias
- source
- hits
- enabled
- last_seen_ts

操作:

- 有効/無効
- alias 追加
- unknown command から昇格登録

## API 設計

### `GET /api/runtime`

現在の runtime 状態を返す。

例:

```json
{
  "running": true,
  "pid": 12345,
  "state": "wake_wait",
  "active_mode": "always_on",
  "transcription_backend": "local",
  "recognition_mode": "balanced",
  "wake_word": "アレクサ",
  "recognized_fast": "次の曲",
  "recognized_final": "次の曲",
  "last_reply": "次の曲にするのだ。",
  "last_command": "music_next",
  "updated_at": "2026-05-04 14:20:10"
}
```

### `GET /api/events?type=command&limit=50`

イベント一覧を返す。

対象:

- `command`
- `command_unknown`
- `wake_check`
- failure 用の複合 view

### `POST /api/commands/execute`

ブラウザから command を送る。

request:

```json
{
  "text": "次の曲",
  "source": "web_console"
}
```

response:

```json
{
  "ok": true,
  "command_id": "music_next",
  "reply": "次の曲にするのだ。",
  "stdout": "",
  "stderr": "",
  "payload": {
    "action_type": "external_cli",
    "action_name": "music.next",
    "args": {}
  }
}
```

### `GET /api/config`

現在の設定を返す。

レスポンスは 2 形式を持てるようにする。

- `yaml_text`
- `parsed`

### `PUT /api/config`

設定保存。

request:

```json
{
  "yaml_text": "runtime:\n  run_mode: always_on\n"
}
```

保存時の処理:

1. YAML parse
2. 必須キー確認
3. `config.yaml.bak.YYYYMMDD_HHMMSS` を保存
4. 本体保存

### `GET /api/aliases`

alias 一覧。

### `POST /api/aliases`

alias 追加。

### `PATCH /api/aliases/{id}`

enabled 切替など。

## コマンド実行設計

### 方針

ブラウザ版専用の command 実行ロジックは増やさない。

理想の流れ:

1. テキストを受ける
2. `match_command()` と `resolve_playback_context_command()` で候補解決
3. `execute_command_action()` を呼ぶ
4. `append_event_logs()` 相当で記録
5. reply を返す

### 追加したい整理

現状 `tools/wake_vad_record.py` に command 周辺の関数が集まりすぎている。

ブラウザ版着手時に、次の共通モジュール切り出しを行うのがよい。

- `src/voicechat/command_router.py`
  command match / normalize / resolve
- `src/voicechat/command_executor.py`
  action 実行
- `src/voicechat/event_log.py`
  DB / JSONL 記録

ブラウザ版 backend と音声本線の両方がこれを使う構成にすると、実装の重複が減る。

## 更新通知

初期版:

- 3 秒おき polling

後で追加:

- `GET /api/stream` の SSE

理由:

- Pi 上ではまず polling で十分
- 実装が単純
- 問題がなければ SSE に進める

## セキュリティ

この UI は実質ローカル管理画面なので、最初から外部公開前提にはしない。

最低限:

- `127.0.0.1` bind を既定にする
- LAN 公開時は明示設定
- 簡易 token 認証を持てるようにする
- 設定保存と command 実行 API には認証をかける
- CORS は閉じる

推奨:

- `systemd` で localhost bind
- 必要なら `nginx` や Tailscale 越しにだけ出す

## systemd 案

`voicechat-console.service`

役割:

- ブラウザ管理 API と静的 UI を提供
- `voicechat.service` と独立して起動

依存:

- `After=voicechat.service` は必須ではない
- 単独起動でも state file / DB が無ければ空表示でよい

## 実装フェーズ

### Phase 1

- `GET /api/runtime`
- `GET /api/events`
- ダッシュボード画面
- コマンド履歴画面

この段階では read-only。

### Phase 2

- `POST /api/commands/execute`
- コマンド入力 UI
- 実行結果表示

ここで「ブラウザから次の曲」が可能になる。

### Phase 3

- `GET /api/config`
- `PUT /api/config`
- 主要設定フォーム
- YAML 直接編集

### Phase 4

- alias 一覧
- alias 追加・有効無効
- unknown command 支援

### Phase 5

- SSE
- フィルタ付き履歴
- 失敗分析 UI

## リスク

### 1. `wake_vad_record.py` 依存が強い

command 解決と実行が 1 ファイルに寄っているため、HTTP から再利用しにくい。

対策:

- 先に共通モジュールへ分割する

### 2. 設定保存で壊す危険

YAML をブラウザから直接触ると、保存ミスで本線起動不能になり得る。

対策:

- 保存前バックアップ
- 最低限の必須キー validation
- 主要項目はフォーム編集を優先

### 3. 実行中本線との整合

設定を書き換えても実行中プロセスには即時反映されない項目がある。

対策:

- UI 上で「保存済み / 実行中未反映」を明示
- 必要なら再起動ボタンは後で追加

### 4. コマンド送信元の混在

音声からの command とブラウザからの command が同じ `events` に入る。

対策:

- `payload_json.source = web_console` を入れる
- UI で source filter を持つ

## 推奨する最初の実装順

1. command / event / state 読み出しの共通モジュール化
2. read-only backend
3. read-only Web UI
4. browser command execute
5. config edit

この順なら、まず観測系を安定させてから操作系へ進める。

## 結論

ブラウザ版は新しい「音声アシスタント本体」ではなく、既存 `voicechat` の管理コンソールとして作るのが正しい。

中核は次の再利用にある。

- `runtime_state.json`
- `events`
- `recognition_aliases`
- 既存 command router
- 既存 action runner

最初は read-only ダッシュボードと履歴から始め、その次に command 送信、最後に設定編集へ進む構成が最も安全で実装コストも低い。

# voicechat 運用設計書

最終更新: 2026-03-14

## 1. 目的

本書は常時運用と実験運用の切替を安全に行うための運用指針を示す。

## 2. 実行モード別運用

### 2.1 phrase_test

用途:

- 認識精度比較
- ずんだもんスタイル比較

### 2.2 always_on

用途:

- 24時間365日待受
- ウェイクワード起点の音声リモコン

推奨:

- `runtime.transcription_backend: local`
- `whisper.realtime_model` は軽量
- `whisper.model` は保存品質優先
- `command_execution: true`

### 2.3 timed_record

用途:

- 会議、作業、独り言の一定時間記録

推奨:

- 長時間時はバックグラウンド文字起こし
- 必要に応じて SSH オフロード

## 3. 障害時運用

- `VOICEVOX` 停止時は TTS 不可
- `Ollama` 停止時は AI 補正不可
- SSH 接続不可時は remote backend 利用不可
- マイク未接続時は録音不可

現在は起動前に `services.voicevox` と `services.ollama` を確認し、`auto_start: true` なら自動起動を試みる。

## 4. ログ運用

記録先:

- `/tmp/voicechat/events.jsonl`
- `/tmp/voicechat/voicechat.db`

最低限確認する項目:

- `date`
- `event_type`
- `mode`
- `recognized`
- `elapsed_sec`

CLI からの状態確認:

```bash
cd <repo-root>
./.venv/bin/python tools/show_status.py
```

確認できる内容:

- 起動中かどうか
- 現在どこで待っているか
- 最新認識イベント
- 現在設定

実行中状態は `/tmp/voicechat/runtime_state.json` に保存される。

誤認識 alias の手動学習:

```bash
cd <repo-root>
./.venv/bin/python tools/recognition_alias_manager.py unknowns --limit 20
./.venv/bin/python tools/recognition_alias_manager.py commands
./.venv/bin/python tools/recognition_alias_manager.py add-command-alias --event-id 712 --command-id help
```

## 5. 今後の運用改善

- `systemd` サービス化
- 自動再起動
- DB ローテーション
- 録音ファイル整理
- 失敗イベント監視
- `recognition_aliases` の棚卸し

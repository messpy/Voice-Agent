# voicechat データ設計書

最終更新: 2026-03-14

## 1. 設定ファイル

対象:

- [config/config.yaml](config/config.yaml)
- [config/phrase_test_words.json](config/phrase_test_words.json)

## 2. config.yaml 主体

### 2.1 `runtime`

- `run_mode`
- `assistant_mode`
- `ai_correction`
- `transcription_backend`
- `command_execution`
- `announce_startup`

### 2.2 `storage`

- `jsonl.enabled`
- `jsonl.path`
- `sqlite.enabled`
- `sqlite.path`

### 2.3 `services`

- `voicevox.enabled`
- `voicevox.auto_start`
- `voicevox.health_url`
- `voicevox.start_command`
- `voicevox.cwd`
- `voicevox.startup_wait_sec`
- `ollama.enabled`
- `ollama.auto_start`
- `ollama.start_command`
- `ollama.startup_wait_sec`

### 2.4 `timed_record`

- `seconds`
- `announce_start`
- `announce_end_template`
- `background`
- `save_wav`

### 2.5 `remote`

- `host`
- `user`
- `port`
- `ssh_key`
- `remote_workdir`
- `remote_python`
- `remote_script`
- `whisper_model`
- `device`
- `compute_type`
- `skip_correction`

### 2.6 `command_router`

- `enabled`
- `fallback_reply`
- `ready_prompt`
- `unknown_command_reply`
- `short_input_reset_chars`
- `short_input_reset_reply`
- `no_speech_timeout_sec`
- `no_speech_reset_reply`
- `commands[].id`
- `commands[].phrases`
- `commands[].reply`
- `commands[].command`

### 2.7 `whisper`

- `bin`
- `realtime_model`
- `model`
- `lang`
- `beam`
- `best`
- `temp`
- `threads`

### 2.8 `llm`

- `provider`
- `model`
- `timeout_sec`
- `host`
- `api_key_env`
- `api_base`
- `anthropic_version`
- `web_search.enabled`
- `web_search.max_results`

### 2.9 `assistant.memory_search`

- `enabled`
- `top_k`
- `min_score`
- `scan_limit`
- `lookback_turns`

## 3. phrase test JSON

形式:

```json
{
  "phrases": [
    {
      "id": "wake_only",
      "text": "ベリベリ",
      "note": "ウェイクワード単体"
    }
  ]
}
```

## 4. SQLite 設計

DB パス:

- `/tmp/voicechat/voicechat.db`

テーブル:

### `events`

- `id integer primary key autoincrement`
- `ts integer not null`
- `date text not null`
- `event_type text not null`
- `mode text`
- `backend text`
- `model text`
- `expected text`
- `recognized text`
- `elapsed_sec real`
- `payload_json text not null`

`payload_json` には次も入る。

- `recognized_fast`
- `recognized_final`
- `model_fast`
- `model_final`
- `command_candidates`
- `command_reply`

### `recognition_aliases`

- `id integer primary key autoincrement`
- `alias_type text not null`
- `target text not null`
- `alias text not null`
- `source text not null default 'auto'`
- `hits integer not null default 1`
- `last_seen_ts integer not null`
- `enabled integer not null default 1`
- `unique(alias_type, target, alias)`

インデックス:

- `idx_events_ts`
- `idx_events_type`
- `idx_recognition_aliases_type_target`
- `idx_recognition_aliases_alias`

## 5. JSONL 記録

保存先:

- `/tmp/voicechat/events.jsonl`

イベント例:

- `phrase_test`
- `command`
- `command_unknown`
- `command_reset_short_input`
- `command_reset_no_speech`
- `assistant_turn`
- `timed_record`
- `wake_check`

## 6. 出力ファイル

主な出力:

- `*_raw.txt`
- `*_corrected.txt`
- `*.json`
- `*.wav`
- `events.jsonl`
- `voicechat.db`
- `runtime_state.json`

## 7. 運用補助スクリプト

- `tools/show_status.py`
  現在状態、最新認識、設定全文を CLI 表示
- `tools/recognition_alias_manager.py`
  未知コマンド一覧確認、manual alias 追加

# voicechat

This repository is an experimental voice stack for Raspberry Pi.

It combines:

- speech recognition
- Zundamon voice synthesis via `VOICEVOX`
- phrase repetition tests
- wake-word based voice control
- fixed-duration recording and transcription
- optional AI-based transcript correction
- optional SSH offloading to another machine
- multiple LLM providers for correction and assistant replies

The current main entrypoint is [tools/wake_vad_record.py](tools/wake_vad_record.py).

## Main capabilities

- Local speech recognition with `whisper.cpp`
- Remote speech recognition with `faster-whisper` over SSH
- Zundamon-only TTS with `VOICEVOX`
- Phrase testing with external JSON phrase lists
- Always-on wake-word mode using `ベリベリ`
- Timed recording with spoken `Start` and `5 minutes finished`
- Dual logging to `JSONL` and `SQLite`

## Main files

- [config/config.yaml](config/config.yaml)
  Local runtime configuration copied from the example
- [config/config.example.yaml](config/config.example.yaml)
  Public example configuration
- [config/phrase_test_words.json](config/phrase_test_words.json)
  Phrase test word list
- [tools/wake_vad_record.py](tools/wake_vad_record.py)
  Unified runtime entrypoint
- [tools/timed_record_transcribe.py](tools/timed_record_transcribe.py)
  Standalone timed recording helper
- [tools/remote_faster_whisper_transcribe.py](tools/remote_faster_whisper_transcribe.py)
  Remote transcription helper
- [tools/remote_faster_whisper_bench.py](tools/remote_faster_whisper_bench.py)
  Remote model benchmark helper
- [docs/architecture.md](docs/architecture.md)
  Japanese architecture note
- [docs/architecture.en.md](docs/architecture.en.md)
  English architecture note

## Public configuration

The tracked repository keeps `config/config.example.yaml` as the public sample.
Create your local `config/config.yaml` from it before running the app.

## Requirements

### On Raspberry Pi

- Python 3.12+
- `uv`
- `arecord`
- `aplay`
- `VOICEVOX engine`
- `whisper.cpp`
- `Ollama`

Setup:

```bash
cd <repo-root>
cp config/config.example.yaml config/config.yaml
uv venv
uv sync
```

### VOICEVOX

This project is now VOICEVOX-only for TTS.

Current default speaker:

- `speaker: 3`
- `Zundamon / Normal`

Health checks:

```bash
curl -sS http://127.0.0.1:50021/version
curl -sS http://127.0.0.1:50021/speakers
```

### SSH offload

If you want remote transcription, Raspberry Pi must be able to SSH into the remote host with public key auth.
Set the SSH key path, remote host, and remote working directory in your local `config/config.yaml`.

## Runtime configuration

The unified runtime is controlled mainly by `runtime` in [config/config.yaml](config/config.yaml).

Key flags:

- `runtime.run_mode`
  `phrase_test` / `debug_stt` / `always_on` / `timed_record` / `ai_duo`
- `runtime.assistant_mode`
  Assistant persona used during `always_on`
- `runtime.ai_correction`
  Enable or disable transcript correction
- `runtime.transcription_backend`
  `local` or `ssh_remote`
- `runtime.command_execution`
  Enable or disable voice command execution
- `timed_record.seconds`
  Fixed recording duration
- `storage.jsonl.path`
  JSONL log destination
- `storage.sqlite.path`
  SQLite database path
- `assistant.memory_search`
  Long-term conversation retrieval

Main run command:

```bash
cd <repo-root>
./.venv/bin/python -m tools.wake_vad_record
```

## LLM providers

The project can switch between multiple providers for transcript correction and assistant replies.

Supported providers:

- `ollama`
- `gemini`
- `openai`
- `anthropic`

Example configurations:

```yaml
llm:
  provider: ollama
  model: qwen2.5:7b
  host: http://127.0.0.1:11434
  api_key_env: OLLAMA_API_KEY
  web_search:
    enabled: false
    max_results: 5
```

If you point `host` at `https://ollama.com` and provide `OLLAMA_API_KEY`, the same provider block can be used against Ollama Cloud. When `web_search.enabled: true`, Ollama web search results are injected as auxiliary context before generation.

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
  api_key_env: GEMINI_API_KEY
  api_base: https://generativelanguage.googleapis.com/v1beta
```

```yaml
llm:
  provider: openai
  model: gpt-5-mini
  api_key_env: OPENAI_API_KEY
  api_base: https://api.openai.com/v1
```

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY
  api_base: https://api.anthropic.com/v1
  anthropic_version: 2023-06-01
```

## Modes

### Phrase test

Use:

```yaml
runtime:
  run_mode: phrase_test
```

Behavior:

- Zundamon prompts the user
- phrase list is loaded from JSON
- speaker style rotates across Zundamon styles
- expected text, recognized text, and elapsed time are logged

### Always-on mode

Use:

```yaml
runtime:
  run_mode: always_on
  assistant_mode: zundamon
  ai_correction: true
  transcription_backend: local
  command_execution: true
```

Behavior:

- listens for the wake word `ベリベリ`
- records the following utterance
- optionally corrects the transcript
- executes known voice commands if matched
- otherwise routes into the assistant conversation flow

Built-in command examples:

- `音量あげて`
- `音量下げて`
- `音楽とめて`
- `音楽起動して`
- `ベリベリ音楽起動して`

Command mappings are defined in `command_router.commands`.

### Long-term memory retrieval

If `assistant.memory_search` is enabled, the runtime scans past events from `voicechat.db` and injects date-stamped memory hits into the conversation context.

This is intended to help with queries like:

- `When did we talk about this?`
- `When did I ask to turn off the light?`
- `When did we talk about starting the music?`

Example:

```yaml
assistant:
  memory_search:
    enabled: true
    top_k: 5
    min_score: 0.18
    scan_limit: 2000
    lookback_turns: 3
```

### Timed record mode

Use:

```yaml
runtime:
  run_mode: timed_record
  ai_correction: true
  transcription_backend: local

timed_record:
  seconds: 300
```

Behavior:

- Zundamon says `スタート`
- records for the configured duration
- Zundamon says `5分終了`
- performs transcription
- optionally applies AI correction
- stores both text outputs and event logs

## Local vs remote transcription

### Local

Best for:

- always-on usage
- low operational complexity
- light models
- wake-word voice control

Configured with:

```yaml
runtime:
  transcription_backend: local
```

### SSH remote

Best for:

- trying heavier models
- model comparisons
- offloading transcription from Raspberry Pi

Configured with:

```yaml
runtime:
  transcription_backend: ssh_remote
```

Remote host settings are defined in `remote`.

## Logging

Events are stored in two places:

- `/tmp/voicechat/events.jsonl`
- `/tmp/voicechat/voicechat.db`

The SQLite database contains an `events` table with fields such as:

- `date`
- `event_type`
- `mode`
- `backend`
- `model`
- `expected`
- `recognized`
- `elapsed_sec`
- `payload_json`

## Notes

- If YouTube audio and your own voice are mixed into one microphone input, recognition quality drops sharply for all models.
- AI correction is useful for readability, but raw transcripts are safer for audit or record-keeping.
- SSH offload only helps if the remote machine is actually stronger than the Pi for the chosen model.

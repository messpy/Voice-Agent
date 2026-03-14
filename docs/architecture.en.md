# voicechat Architecture Note

Last updated: 2026-03-13

## Goal

This project provides a Raspberry Pi based voice runtime with the following capabilities:

- Zundamon TTS through VOICEVOX
- local and remote speech recognition
- wake-word interaction using `ベリベリ`
- timed recording and transcription
- AI correction with selectable LLM providers
- persistent event logging

## Main runtime

The current main runtime is:

- [tools/wake_vad_record.py](tools/wake_vad_record.py)

It acts as a unified runner and switches behavior by configuration.

## Runtime modes

- `phrase_test`
- `debug_stt`
- `always_on`
- `timed_record`
- `ai_duo`

## Speech recognition backends

- `local`
  uses `whisper.cpp`
- `ssh_remote`
  sends WAV files to another machine and runs `faster-whisper`

## LLM providers

The project no longer assumes Ollama only.

Supported providers:

- `ollama`
- `gemini`
- `openai`
- `anthropic`

Typical uses:

- transcript correction
- phrase test judgement
- assistant replies
- AI duo mode

## Persistence

Events are stored in:

- `JSONL`
- `SQLite`

SQLite table:

- `events`

Main columns:

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

## Future work

- systemd service definition
- async worker split for long-running deployments
- stronger command routing
- retry policy for remote offload

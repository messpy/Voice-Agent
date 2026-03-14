#!/usr/bin/env bash
set -euo pipefail

MIC_CARD="${MIC_CARD:-2}"
MIC_DEV="plughw:${MIC_CARD},0"
REC_SEC="${REC_SEC:-5}"

WHISPER_BIN="$HOME/voicechat/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL="$HOME/voicechat/whisper.cpp/models/ggml-base.bin"

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:latest}"

IN_WAV="$HOME/voicechat/in.wav"
OUT_PREFIX="$HOME/voicechat/out"
OUT_TXT="$HOME/voicechat/out.txt"

say() {
  espeak-ng -v ja "$1" --stdout | aplay -q || true
}

ollama_chat() {
  local prompt="$1"
  local tmp_body
  tmp_body="$(mktemp)"

  local payload
  payload="$(python3 - <<'PY' "$OLLAMA_MODEL" "$prompt"
import json, sys
model=sys.argv[1]
prompt=sys.argv[2]
print(json.dumps({
  "model": model,
  "messages": [
    {"role":"system","content":"あなたは日本語で簡潔に返答する。1〜2文で答える。"},
    {"role":"user","content": prompt}
  ],
  "stream": False
}, ensure_ascii=False))
PY
)"

  local http
  http="$(curl -sS -o "$tmp_body" -w '%{http_code}' \
    http://127.0.0.1:11434/api/chat \
    -H 'Content-Type: application/json' \
    -d "$payload" || true)"

  if [[ "$http" != "200" ]]; then
    echo "[OLLAMA] HTTP=$http" >&2
    echo "[OLLAMA] BODY:" >&2
    head -c 2000 "$tmp_body" >&2 || true
    rm -f "$tmp_body"
    return 1
  fi

  python3 - <<'PY' "$tmp_body"
import json, sys, pathlib
data=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore"))
print(data["message"]["content"].strip())
PY

  rm -f "$tmp_body"
}

while true; do
  echo "[REC] ${REC_SEC}s 話して"
  arecord -D "$MIC_DEV" -f S16_LE -r 16000 -c 1 -d "$REC_SEC" -t wav "$IN_WAV" >/dev/null 2>&1 || true

  "$WHISPER_BIN" \
    -m "$WHISPER_MODEL" \
    -f "$IN_WAV" \
    -l ja \
    -nt \
    -otxt \
    -of "$OUT_PREFIX" >/dev/null 2>&1 || true

  USER_TEXT="$(cat "$OUT_TXT" 2>/dev/null || true | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if [[ -z "$USER_TEXT" ]]; then
    echo "[STT] empty"
    continue
  fi

  echo "[YOU] $USER_TEXT"

  if RESP="$(ollama_chat "$USER_TEXT")"; then
    echo "[AI ] $RESP"
    say "$RESP"
  else
    say "オラマに接続できませんでした。"
  fi
done

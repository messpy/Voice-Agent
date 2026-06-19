#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/diagnostics"
AUDIO_TEST_DIR="$PROJECT_DIR/logs/audio-tests"
mkdir -p "$LOG_DIR" "$AUDIO_TEST_DIR"
LOG_FILE="$LOG_DIR/voicechat_audio_owner_check_$(date +%Y%m%d_%H%M%S).log"

stop_error() {
  echo "[ERROR] STEP=$1 $2" | tee -a "$LOG_FILE"
  echo "[ACTION] Ctrl+Cで停止してください。Enterで終了します。" | tee -a "$LOG_FILE"
  read -r _
  exit 1
}

echo "[CHECK] STEP=1 状態確認 voicechatプロセス" | tee "$LOG_FILE"
ps aux | grep -Ei 'tools.wake_vad_record|voicechat|arecord|python' | grep -v grep | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=2 想定状態判定 ALSAデバイス" | tee -a "$LOG_FILE"
arecord -l 2>&1 | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=3 マイクを掴んでいるプロセス確認" | tee -a "$LOG_FILE"
if command -v fuser >/dev/null 2>&1; then
  fuser -v /dev/snd/* 2>&1 | tee -a "$LOG_FILE" || true
else
  echo "[ERROR] STEP=3 fuser がありません" | tee -a "$LOG_FILE"
fi

echo "[CHECK] STEP=4 lsof確認" | tee -a "$LOG_FILE"
if command -v lsof >/dev/null 2>&1; then
  lsof /dev/snd/* 2>&1 | tee -a "$LOG_FILE" || true
else
  echo "[ERROR] STEP=4 lsof がありません" | tee -a "$LOG_FILE"
fi

echo "[ACTION] STEP=5 voicechatを一時停止して録音テストしますか？ yes/no" | tee -a "$LOG_FILE"
echo "[ACTION] yesの場合: voicechat.serviceを停止 -> 3秒録音 -> voicechat.serviceを再起動します" | tee -a "$LOG_FILE"
read -r ANSWER

if [ "$ANSWER" = "yes" ]; then
  echo "[ACTION] voicechat.service 停止" | tee -a "$LOG_FILE"
  systemctl --user stop voicechat.service || stop_error "5" "voicechat.service停止に失敗"

  echo "[CHECK] STEP=6 停止後のALSA確認" | tee -a "$LOG_FILE"
  arecord -l 2>&1 | tee -a "$LOG_FILE" || true

  echo "[ACTION] STEP=7 card 3 device 0 で3秒録音テスト" | tee -a "$LOG_FILE"
  TEST_WAV="$AUDIO_TEST_DIR/voicechat_mic_test_card3.wav"
  arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -d 3 "$TEST_WAV" 2>&1 | tee -a "$LOG_FILE" || echo "[ERROR] STEP=7 card3録音失敗" | tee -a "$LOG_FILE"

  echo "[ACTION] STEP=8 card 1 device 0 で3秒録音テスト" | tee -a "$LOG_FILE"
  TEST_WAV2="$AUDIO_TEST_DIR/voicechat_mic_test_card1.wav"
  arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 3 "$TEST_WAV2" 2>&1 | tee -a "$LOG_FILE" || echo "[ERROR] STEP=8 card1録音失敗" | tee -a "$LOG_FILE"

  echo "[ACTION] STEP=9 voicechat.service 再起動" | tee -a "$LOG_FILE"
  systemctl --user start voicechat.service || stop_error "9" "voicechat.service再起動に失敗"

elif [ "$ANSWER" = "no" ]; then
  echo "[INFO] 録音テストをスキップしました" | tee -a "$LOG_FILE"
else
  stop_error "5" "yes/no 以外が入力されました"
fi

echo "[CHECK] STEP=10 結果確認" | tee -a "$LOG_FILE"
ls -lh "$AUDIO_TEST_DIR"/voicechat_mic_test_card*.wav 2>/dev/null | tee -a "$LOG_FILE" || true
grep -Ein 'error|fatal|failed|busy|permission|card|device|arecord|voicechat|python|snd|recording|録音' "$LOG_FILE" | tail -160 || true

echo "[INFO] 調査完了"
echo "[INFO] ログ: $LOG_FILE"

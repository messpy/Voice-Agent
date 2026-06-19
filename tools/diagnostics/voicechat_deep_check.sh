#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/diagnostics"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/voicechat_deep_check_$(date +%Y%m%d_%H%M%S).log"

stop_error() {
  echo "[ERROR] STEP=$1 $2" | tee -a "$LOG_FILE"
  echo "[ACTION] Ctrl+Cで停止してください。Enterで終了します。" | tee -a "$LOG_FILE"
  read -r _
  exit 1
}

echo "[CHECK] STEP=1 voicechat.service定義確認" | tee "$LOG_FILE"
systemctl --user cat voicechat.service | tee -a "$LOG_FILE" || stop_error "1" "voicechat.service を読めません"

echo "[CHECK] STEP=2 想定状態判定" | tee -a "$LOG_FILE"
systemctl --user status voicechat.service --no-pager | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=3 voicechat.serviceログ 生表示" | tee -a "$LOG_FILE"
journalctl --user -u voicechat.service -n 120 --no-pager | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=4 実行中プロセス確認" | tee -a "$LOG_FILE"
ps -fp "$(pgrep -f 'tools.wake_vad_record' | head -1)" | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=5 音声デバイス確認" | tee -a "$LOG_FILE"
arecord -l 2>&1 | tee -a "$LOG_FILE" || true
pactl info 2>&1 | tee -a "$LOG_FILE" || true
pactl list short sources 2>&1 | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=6 voicechat設定確認 envのキー名だけ表示" | tee -a "$LOG_FILE"
if [ -f "$PROJECT_DIR/.env" ]; then
  sed -n 's/^\([A-Za-z0-9_][A-Za-z0-9_]*\)=.*/\1=***MASKED***/p' "$PROJECT_DIR/.env" | tee -a "$LOG_FILE"
else
  echo "[ERROR] STEP=6 .env がありません" | tee -a "$LOG_FILE"
fi

echo "[ACTION] STEP=7 直接起動テストを実行しますか？ yes/no" | tee -a "$LOG_FILE"
read -r ANSWER
if [ "$ANSWER" = "yes" ]; then
  cd "$PROJECT_DIR" || stop_error "7" "voicechatに移動できません"
  timeout 15s "$PROJECT_DIR/.venv/bin/python" -m tools.wake_vad_record 2>&1 | tee -a "$LOG_FILE" || true
elif [ "$ANSWER" = "no" ]; then
  echo "[INFO] 直接起動テストをスキップしました" | tee -a "$LOG_FILE"
else
  stop_error "7" "yes/no 以外が入力されました"
fi

echo "[CHECK] STEP=8 重要行抽出" | tee -a "$LOG_FILE"
grep -Ein 'error|fatal|exception|traceback|failed|cannot|not found|permission|timeout|refused|wake|vad|mic|audio|alsa|pulse|pipewire|device|input|source|record|listen|started|ready' "$LOG_FILE" | tail -160 || true

echo "[INFO] 調査完了"
echo "[INFO] ログ: $LOG_FILE"

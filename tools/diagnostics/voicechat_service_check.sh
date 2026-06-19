#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/diagnostics"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/voicechat_service_check_$(date +%Y%m%d_%H%M%S).log"

stop_error() {
  echo "[ERROR] STEP=$1 $2" | tee -a "$LOG_FILE"
  echo "[ACTION] Ctrl+Cで停止してください。Enterで終了します。" | tee -a "$LOG_FILE"
  read -r _
  exit 1
}

echo "[CHECK] STEP=1 systemd user service状態確認" | tee "$LOG_FILE"
systemctl --user status voicechat.service --no-pager | tee -a "$LOG_FILE" || true
systemctl --user status voicechat-web.service --no-pager | tee -a "$LOG_FILE" || true
systemctl --user status voicechat-actions-scheduler.service --no-pager | tee -a "$LOG_FILE" || true
systemctl --user status shared-voicevox.service --no-pager | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=2 想定状態判定" | tee -a "$LOG_FILE"
VOICECHAT_STATE="$(systemctl --user is-active voicechat.service || true)"
WEB_STATE="$(systemctl --user is-active voicechat-web.service || true)"
VOICEVOX_STATE="$(systemctl --user is-active shared-voicevox.service || true)"

echo "[INFO] voicechat.service=$VOICECHAT_STATE" | tee -a "$LOG_FILE"
echo "[INFO] voicechat-web.service=$WEB_STATE" | tee -a "$LOG_FILE"
echo "[INFO] shared-voicevox.service=$VOICEVOX_STATE" | tee -a "$LOG_FILE"

if [ "$VOICECHAT_STATE" != "active" ]; then
  stop_error "2" "voicechat.service が active ではありません"
fi

if [ "$WEB_STATE" != "active" ]; then
  stop_error "2" "voicechat-web.service が active ではありません"
fi

if [ "$VOICEVOX_STATE" != "active" ]; then
  echo "[ERROR] STEP=2 shared-voicevox.service が active ではありません。音声合成失敗の原因候補です" | tee -a "$LOG_FILE"
fi

echo "[ACTION] STEP=3 ログ収集" | tee -a "$LOG_FILE"
journalctl --user -u voicechat.service -n 120 --no-pager | tee -a "$LOG_FILE"
journalctl --user -u voicechat-web.service -n 80 --no-pager | tee -a "$LOG_FILE"
journalctl --user -u voicechat-actions-scheduler.service -n 80 --no-pager | tee -a "$LOG_FILE"
journalctl --user -u shared-voicevox.service -n 120 --no-pager | tee -a "$LOG_FILE"

echo "[CHECK] STEP=4 ポート疎通確認" | tee -a "$LOG_FILE"
curl -sS --max-time 3 http://127.0.0.1:8787/ 2>&1 | head -20 | tee -a "$LOG_FILE" || true
curl -sS --max-time 3 http://127.0.0.1:50021/version 2>&1 | tee -a "$LOG_FILE" || true
curl -sS --max-time 3 http://127.0.0.1:11434/api/tags 2>&1 | head -20 | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=5 重要行抽出" | tee -a "$LOG_FILE"
grep -Ein 'error|fatal|exception|traceback|failed|cannot|not found|permission denied|address already in use|timeout|refused|missing|enoent|eaddrinuse|inactive|dead' "$LOG_FILE" | tail -120 || true

echo "[INFO] 調査完了"
echo "[INFO] ログ: $LOG_FILE"

#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/diagnostics"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/check_voicevox_condition_$(date +%Y%m%d_%H%M%S).log"

stop_error() {
  echo "[ERROR] STEP=$1 $2" | tee -a "$LOG_FILE"
  echo "[ACTION] Ctrl+Cで停止してください。Enterで終了します。" | tee -a "$LOG_FILE"
  read -r _
  exit 1
}

echo "[CHECK] STEP=1 shared-voicevox.service 定義確認" | tee "$LOG_FILE"
systemctl --user cat shared-voicevox.service | tee -a "$LOG_FILE" || stop_error "1" "service定義を読めません"

echo "[CHECK] STEP=2 ExecCondition 抽出" | tee -a "$LOG_FILE"
systemctl --user cat shared-voicevox.service | grep -nE 'ExecCondition|Condition|Environment|ExecStart|WorkingDirectory' | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=3 podman/docker確認" | tee -a "$LOG_FILE"
command -v docker >/dev/null 2>&1 && docker ps -a | grep -Ei 'voicevox|voice' | tee -a "$LOG_FILE" || true
command -v podman >/dev/null 2>&1 && podman ps -a | grep -Ei 'voicevox|voice' | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=4 VOICEVOXポート確認" | tee -a "$LOG_FILE"
ss -ltnup 2>/dev/null | grep -E ':50021\b' | tee -a "$LOG_FILE" || echo "[ERROR] STEP=4 50021 がLISTENしていません" | tee -a "$LOG_FILE"

echo "[CHECK] STEP=5 重要行確認" | tee -a "$LOG_FILE"
grep -Ein 'ExecCondition|Condition|ExecStart|voicevox|docker|podman|50021|error|failed|skipped' "$LOG_FILE" | tail -120 || true

echo "[INFO] 調査完了"
echo "[INFO] ログ: $LOG_FILE"

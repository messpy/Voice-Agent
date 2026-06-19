#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/diagnostics"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/voicechat_diag_$(date +%Y%m%d_%H%M%S).log"
TARGET_DIR="$PROJECT_DIR"

pause_error() {
  echo "[ERROR] STEP=$1 $2" | tee -a "$LOG_FILE"
  echo "[ACTION] Ctrl+Cで停止してください。Enterで終了します。" | tee -a "$LOG_FILE"
  read -r _
  exit 1
}

echo "[CHECK] STEP=1 voicechatディレクトリ確認" | tee "$LOG_FILE"
if [ ! -d "$TARGET_DIR" ]; then
  pause_error "1" "$TARGET_DIR が存在しません"
fi

echo "[CHECK] STEP=2 kennybot稼働確認" | tee -a "$LOG_FILE"
ps aux | grep -E 'Kenny-bot|kennybot|bin/run.py|src/kennybot' | grep -v grep | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=3 voicechat関連プロセス確認" | tee -a "$LOG_FILE"
ps aux | grep -Ei 'voicechat|voice-chat|shared_voice_ai|uvicorn|fastapi|node|python' | grep -v grep | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=4 想定状態判定 systemd service確認" | tee -a "$LOG_FILE"
systemctl --user list-units --type=service --all | grep -Ei 'voice|chat|kenny' | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=5 voicechat構成確認" | tee -a "$LOG_FILE"
cd "$TARGET_DIR" || pause_error "5" "$TARGET_DIR に移動できません"
find . -maxdepth 2 -type f \( -name 'package.json' -o -name 'pyproject.toml' -o -name 'requirements.txt' -o -name '.env' -o -name '*.service' -o -name 'README*' \) -print | sort | tee -a "$LOG_FILE"

echo "[ACTION] STEP=6 起動方式の判定" | tee -a "$LOG_FILE"
if [ -f "package.json" ]; then
  echo "[INFO] Node.js系として判定" | tee -a "$LOG_FILE"
  node -v | tee -a "$LOG_FILE" || pause_error "6" "node がありません"
  npm -v | tee -a "$LOG_FILE" || pause_error "6" "npm がありません"
  sed -n '/"scripts"[[:space:]]*:/,/}/p' package.json | tee -a "$LOG_FILE"
elif [ -f "pyproject.toml" ] || [ -f "requirements.txt" ]; then
  echo "[INFO] Python系として判定" | tee -a "$LOG_FILE"
  python3 --version | tee -a "$LOG_FILE" || pause_error "6" "python3 がありません"
  command -v uv >/dev/null 2>&1 && uv --version | tee -a "$LOG_FILE" || true
else
  pause_error "6" "起動方式を判定できません"
fi

echo "[CHECK] STEP=7 ポート確認" | tee -a "$LOG_FILE"
ss -ltnup 2>/dev/null | grep -Ei ':(3000|5000|5001|8000|8080|11434|5173|8765|7860)\b|voice|python|node|uvicorn' | tee -a "$LOG_FILE" || true

echo "[CHECK] STEP=8 ログ確認" | tee -a "$LOG_FILE"
find "$TARGET_DIR" "$HOME/work/log" "$HOME/work/runtime" -maxdepth 3 -type f \( -name '*.log' -o -name 'nohup.out' -o -name '*.err' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -20 | tee -a "$LOG_FILE"

echo "[ACTION] STEP=9 起動テストを実行しますか？ yes/no" | tee -a "$LOG_FILE"
read -r ANSWER
if [ "$ANSWER" = "yes" ]; then
  if [ -f "package.json" ]; then
    if grep -q '"dev"' package.json; then
      timeout 10s npm run dev >>"$LOG_FILE" 2>&1 || true
    elif grep -q '"start"' package.json; then
      timeout 10s npm start >>"$LOG_FILE" 2>&1 || true
    else
      pause_error "9" "npm script に dev/start がありません"
    fi
  elif [ -f "main.py" ]; then
    timeout 10s python3 main.py >>"$LOG_FILE" 2>&1 || true
  elif [ -f "app.py" ]; then
    timeout 10s python3 app.py >>"$LOG_FILE" 2>&1 || true
  else
    pause_error "9" "起動候補がありません"
  fi
elif [ "$ANSWER" = "no" ]; then
  echo "[INFO] 起動テストをスキップしました" | tee -a "$LOG_FILE"
else
  pause_error "9" "yes/no 以外が入力されました"
fi

echo "[CHECK] STEP=10 結果確認" | tee -a "$LOG_FILE"
grep -Ein 'error|fatal|exception|traceback|failed|cannot|not found|permission denied|address already in use|timeout|refused|missing|enoent|eaddrinuse' "$LOG_FILE" | tail -80 || true

echo "[INFO] 調査完了"
echo "[INFO] ログ: $LOG_FILE"

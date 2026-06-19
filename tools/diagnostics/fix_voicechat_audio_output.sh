#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/diagnostics"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fix_voicechat_audio_output_$(date +%Y%m%d_%H%M%S).log"
ASOUNDRC="$HOME/.asoundrc"
BACKUP="$HOME/.asoundrc.backup.$(date +%Y%m%d_%H%M%S)"

stop_error() {
  echo "[ERROR] STEP=$1 $2" | tee -a "$LOG_FILE"
  echo "[ACTION] Ctrl+Cで停止してください。Enterで終了します。" | tee -a "$LOG_FILE"
  read -r _
  exit 1
}

echo "[CHECK] STEP=1 状態確認 再生デバイス一覧" | tee "$LOG_FILE"
aplay -l | tee -a "$LOG_FILE"

echo "[CHECK] STEP=2 想定状態判定 USB音声デバイス確認" | tee -a "$LOG_FILE"
if ! aplay -l | grep -q 'card 1: B01'; then
  stop_error "2" "card 1: B01 が見つかりません。USB音声デバイス番号が変わっています"
fi

echo "[ACTION] STEP=3 .asoundrc をUSB出力固定に変更しますか？ yes/no" | tee -a "$LOG_FILE"
read -r ANSWER

if [ "$ANSWER" = "yes" ]; then
  if [ -f "$ASOUNDRC" ]; then
    cp "$ASOUNDRC" "$BACKUP" || stop_error "3" ".asoundrc のバックアップに失敗"
    echo "[INFO] backup=$BACKUP" | tee -a "$LOG_FILE"
  fi

  cat > "$ASOUNDRC" <<'ASOUND'
pcm.!default {
    type plug
    slave.pcm "hw:1,0"
}

ctl.!default {
    type hw
    card 1
}
ASOUND

  echo "[INFO] .asoundrc を更新しました" | tee -a "$LOG_FILE"

elif [ "$ANSWER" = "no" ]; then
  echo "[INFO] 変更をスキップしました" | tee -a "$LOG_FILE"
else
  stop_error "3" "yes/no 以外が入力されました"
fi

echo "[ACTION] STEP=4 voicechat.service を再起動しますか？ yes/no" | tee -a "$LOG_FILE"
read -r ANSWER2

if [ "$ANSWER2" = "yes" ]; then
  systemctl --user restart voicechat.service || stop_error "4" "voicechat.service 再起動に失敗"
elif [ "$ANSWER2" = "no" ]; then
  echo "[INFO] 再起動をスキップしました" | tee -a "$LOG_FILE"
else
  stop_error "4" "yes/no 以外が入力されました"
fi

echo "[CHECK] STEP=5 結果確認 default再生テスト" | tee -a "$LOG_FILE"
aplay "$PROJECT_DIR/.runtime/reply_command.wav" 2>&1 | tee -a "$LOG_FILE" || stop_error "5" "default再生に失敗"

echo "[CHECK] STEP=6 voicechat状態確認" | tee -a "$LOG_FILE"
systemctl --user status voicechat.service --no-pager | tee -a "$LOG_FILE" || true

echo "[INFO] 完了"
echo "[INFO] ログ: $LOG_FILE"

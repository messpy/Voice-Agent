#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env.local"
  set +a
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/voicechat-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/voicechat-uv-cache}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:pkg_resources is deprecated as an API:UserWarning}"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR"

VOICECHAT_RUNNER="${VOICECHAT_RUNNER:-auto}"
VOICECHAT_PYTHON="${VOICECHAT_PYTHON:-${PYTHON:-}}"
VOICECHAT_REQUIREMENTS_CHECK="${VOICECHAT_REQUIREMENTS_CHECK:-1}"

PYTHON_REQUIREMENTS_CHECK='import importlib.util, sys
required = ("yaml", "requests", "soundfile", "webrtcvad")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write("missing Python packages: " + ", ".join(missing) + "\n")
    raise SystemExit(1)
'

RUNNER_KIND=""
RUNNER_CMD=()

log_info() {
  echo "INFO: $*" >&2
}

can_run_python() {
  local candidate="$1"
  [[ -n "$candidate" ]] || return 1
  [[ -x "$candidate" ]] || return 1
  "$candidate" -c 'import sys; sys.exit(0)' >/dev/null 2>&1 || return 1
  if [[ "$VOICECHAT_REQUIREMENTS_CHECK" == "0" ]]; then
    return 0
  fi
  "$candidate" -c "$PYTHON_REQUIREMENTS_CHECK" >/dev/null 2>&1
}

can_run_uv() {
  command -v uv >/dev/null 2>&1 || return 1
  if [[ "$VOICECHAT_REQUIREMENTS_CHECK" == "0" ]]; then
    return 0
  fi
  (
    cd "$ROOT" &&
    uv run --no-sync python -c "$PYTHON_REQUIREMENTS_CHECK"
  ) >/dev/null 2>&1
}

can_run_uv_sync() {
  command -v uv >/dev/null 2>&1 || return 1
  if [[ "$VOICECHAT_REQUIREMENTS_CHECK" == "0" ]]; then
    return 0
  fi
  (
    cd "$ROOT" &&
    uv run python -c "$PYTHON_REQUIREMENTS_CHECK"
  ) >/dev/null 2>&1
}

resolve_runner() {
  local -a candidates=()
  local candidate=""

  if [[ -n "$VOICECHAT_PYTHON" ]]; then
    candidates+=("$VOICECHAT_PYTHON")
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("$VIRTUAL_ENV/bin/python")
  fi
  candidates+=(
    "$ROOT/.venv/bin/python"
    "$ROOT/.venv/bin/python3"
  )
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi

  if [[ "$VOICECHAT_RUNNER" == "uv" ]]; then
    if can_run_uv; then
      RUNNER_KIND="uv"
      RUNNER_CMD=(uv run --no-sync python)
      return 0
    fi
    echo "NG: VOICECHAT_RUNNER=uv だが uv run を使えません。" >&2
    return 1
  fi

  if [[ "$VOICECHAT_RUNNER" == "python" ]]; then
    for candidate in "${candidates[@]}"; do
      if can_run_python "$candidate"; then
        RUNNER_KIND="python"
        RUNNER_CMD=("$candidate")
        return 0
      fi
    done
    echo "NG: VOICECHAT_RUNNER=python だが利用可能な Python が見つかりません。" >&2
    return 1
  fi

  for candidate in "${candidates[@]}"; do
    if can_run_python "$candidate"; then
      RUNNER_KIND="python"
      RUNNER_CMD=("$candidate")
      return 0
    fi
  done

  if can_run_uv; then
    RUNNER_KIND="uv"
    RUNNER_CMD=(uv run --no-sync python)
    return 0
  fi

  if can_run_uv_sync; then
    RUNNER_KIND="uv"
    RUNNER_CMD=(uv run python)
    return 0
  fi

  echo "NG: 利用可能な Python 実行環境が見つかりません。" >&2
  echo "   .env に VOICECHAT_PYTHON=/path/to/python を指定するか、uv 実行環境を用意してください。" >&2
  return 1
}

usage() {
  cat <<'EOF'
usage: ./voicechat.sh <command> [args...]

main:
  run                 start the main voicechat runtime
  timed-record        run timed record/transcribe helper
  transcribe-file     transcribe a local audio file with optional AI correction
  status              show runtime status and recent recognition
  web-console         start the browser console
  failures            show recent wake failures and command failures
  phrase-report       rank phrase_test results and optionally announce them
  aliases             manage learned aliases
  memory-search       search recent stored conversation memory
  vad-bench           benchmark webrtc/silero VAD on audio files

env:
  setup-env           copy .env.example -> .env if missing

runner overrides:
  VOICECHAT_PYTHON    explicit Python path
  VOICECHAT_RUNNER    auto | python | uv
  VOICECHAT_REQUIREMENTS_CHECK 1 | 0

remote:
  remote-transcribe   run remote faster-whisper transcription helper
  remote-whisperx     run remote WhisperX transcription helper
  remote-bench        benchmark multiple remote faster-whisper models

cohere:
  cohere-file         transcribe an audio file with Cohere
  cohere-mic          record from microphone and transcribe with Cohere
  cohere-phrase-test  zundamon phrase-repeat test with Cohere

lab:
  lab-voice           launch the interactive voice lab
  lab-zundamon        run the VOICEVOX zundamon style check
  lab-record          record until silence using VAD
  lab-quickcheck      quick whisper smoke test
  lab-quickcheck-cfg  quick whisper smoke test using config.yaml
  lab-bench           benchmark whisper.cpp parameters
  lab-bench-rate      benchmark recognition rate presets
  lab-model-compare   benchmark multiple local STT models on one file
  lab-preprocess-compare compare ffmpeg preprocessing presets on one file
  lab-wake-matrix     compare microphone and model combinations for wake testing

benchmark:
  bench-pi            run full STT benchmark with Raspberry Pi temperature monitoring
EOF
}

run_module() {
  local module="$1"
  shift
  cd "$ROOT"
  resolve_runner
  log_info "runner=$RUNNER_KIND command=${RUNNER_CMD[*]}"
  exec "${RUNNER_CMD[@]}" -m "$module" "$@"
}

run_script() {
  local script="$1"
  shift
  cd "$ROOT"
  resolve_runner
  log_info "runner=$RUNNER_KIND command=${RUNNER_CMD[*]}"
  exec "${RUNNER_CMD[@]}" "$script" "$@"
}

run_module_light() {
  local module="$1"
  shift
  cd "$ROOT"
  local old_check="${VOICECHAT_REQUIREMENTS_CHECK:-1}"
  VOICECHAT_REQUIREMENTS_CHECK=0
  resolve_runner
  VOICECHAT_REQUIREMENTS_CHECK="$old_check"
  log_info "runner=$RUNNER_KIND command=${RUNNER_CMD[*]}"
  exec "${RUNNER_CMD[@]}" -m "$module" "$@"
}

cmd="${1:-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  run)
    run_module tools.wake_vad_record "$@"
    ;;
  timed-record)
    run_script tools/timed_record_transcribe.py "$@"
    ;;
  transcribe-file)
    run_module tools.transcribe_file_local "$@"
    ;;
  status)
    run_module tools.admin.show_status "$@"
    ;;
  web-console)
    run_module_light tools.web_console "$@"
    ;;
  commands)
    run_module tools.admin.show_status commands "$@"
    ;;
  failures)
    run_module tools.admin.show_status failures "$@"
    ;;
  phrase-report)
    run_module tools.admin.phrase_test_report "$@"
    ;;
  aliases)
    run_module tools.admin.recognition_alias_manager "$@"
    ;;
  memory-search)
    run_module tools.admin.search_memory "$@"
    ;;
  vad-bench)
    run_module tools.vad_benchmark "$@"
    ;;
  setup-env)
    run_script tools/setup_env.py "$@"
    ;;
  remote-transcribe)
    run_module tools.remote.faster_whisper_transcribe "$@"
    ;;
  remote-whisperx)
    run_module tools.remote.whisperx_transcribe "$@"
    ;;
  remote-bench)
    run_module tools.remote.faster_whisper_bench "$@"
    ;;
  cohere-file)
    run_module tools.cohere_transcribe_file "$@"
    ;;
  cohere-mic)
    run_module tools.cohere_transcribe_mic "$@"
    ;;
  cohere-phrase-test)
    run_module tools.cohere_phrase_test "$@"
    ;;
  lab-voice)
    run_script tools/lab/voice_lab.py "$@"
    ;;
  lab-zundamon)
    run_module tools.lab.zundamon_style_check "$@"
    ;;
  lab-record)
    run_script tools/lab/record_until_silence.py "$@"
    ;;
  lab-quickcheck)
    run_script tools/lab/whisper_quickcheck.py "$@"
    ;;
  lab-quickcheck-cfg)
    run_module tools.lab.whisper_quickcheck_cfg "$@"
    ;;
  lab-bench)
    run_script tools/lab/whisper_bench.py "$@"
    ;;
  lab-bench-rate)
    run_script tools/lab/whisper_bench_rate.py "$@"
    ;;
  lab-model-compare)
    run_module tools.lab.whisper_model_compare "$@"
    ;;
  lab-preprocess-compare)
    run_module tools.lab.audio_preprocess_compare "$@"
    ;;
  lab-wake-matrix)
    run_script tools/lab/wake_matrix.py "$@"
    ;;
  bench-pi)
    run_script tools/stt_bench_pi.py "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "NG: unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac

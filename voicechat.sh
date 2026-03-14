#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "NG: python not found: $PYTHON" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
usage: ./voicechat.sh <command> [args...]

main:
  run                 start the main voicechat runtime
  timed-record        run timed record/transcribe helper
  status              show runtime status and recent recognition
  aliases             manage learned aliases
  memory-search       search recent stored conversation memory

remote:
  remote-transcribe   run remote faster-whisper transcription helper
  remote-bench        benchmark multiple remote faster-whisper models

lab:
  lab-voice           launch the interactive voice lab
  lab-zundamon        run the VOICEVOX zundamon style check
  lab-record          record until silence using VAD
  lab-quickcheck      quick whisper smoke test
  lab-quickcheck-cfg  quick whisper smoke test using config.yaml
  lab-bench           benchmark whisper.cpp parameters
  lab-bench-rate      benchmark recognition rate presets
EOF
}

run_module() {
  local module="$1"
  shift
  cd "$ROOT"
  exec "$PYTHON" -m "$module" "$@"
}

run_script() {
  local script="$1"
  shift
  cd "$ROOT"
  exec "$PYTHON" "$script" "$@"
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
  status)
    run_module tools.admin.show_status "$@"
    ;;
  aliases)
    run_module tools.admin.recognition_alias_manager "$@"
    ;;
  memory-search)
    run_module tools.admin.search_memory "$@"
    ;;
  remote-transcribe)
    run_module tools.remote.faster_whisper_transcribe "$@"
    ;;
  remote-bench)
    run_module tools.remote.faster_whisper_bench "$@"
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
  -h|--help|help)
    usage
    ;;
  *)
    echo "NG: unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac

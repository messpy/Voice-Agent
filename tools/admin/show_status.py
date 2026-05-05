import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_cfg, resolve_config_path
from src.runtime_files import read_json_file

DEFAULT_CONFIG = resolve_config_path()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    proc_path = Path(f"/proc/{pid}")
    if proc_path.exists():
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def resolve_paths(cfg: dict) -> tuple[Path, Path, Path, Path]:
    workdir = Path(str(cfg["paths"]["workdir"]))
    sqlite_path = workdir / str(cfg.get("storage", {}).get("sqlite", {}).get("path", "voicechat.db"))
    state_path = workdir / "runtime_state.json"
    log_cfg = cfg.get("assistant", {}).get("conversation_log", {})
    conversation_path = workdir / str(log_cfg.get("path", "conversation.jsonl"))
    return workdir, sqlite_path, state_path, conversation_path


def read_state(path: Path) -> dict:
    return read_json_file(path)


def latest_events(
    sqlite_path: Path, limit: int = 10, event_types: list[str] | None = None
) -> list[tuple]:
    if not sqlite_path.exists():
        return []
    query = """
        select date, event_type, recognized, payload_json
        from events
    """
    params: list[Any] = []
    if event_types:
        placeholders = ",".join("?" for _ in event_types)
        query += f" where event_type in ({placeholders})"
        params.extend(event_types)
    else:
        query += " where recognized is not null and recognized != ''"
    query += " order by ts desc limit ?"
    params.append(limit)
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(query, params)
        return cursor.fetchall()


def latest_conversation(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


def print_section(title: str):
    print(f"\n[{title}]")


def print_runtime(cfg: dict, state: dict, workdir: Path, state_path: Path):
    pid = int(state.get("pid", 0) or 0)
    running = pid_alive(pid) if pid else False
    print_section("Runtime")
    print(f"running: {'yes' if running else 'no'}")
    print(f"workdir: {workdir}")
    print(f"state_file: {state_path}")
    if not state:
        print("state: (no runtime state file)")
        return
    print(f"pid: {pid}")
    print(f"state: {state.get('state', '(unknown)')}")
    print(f"updated_at: {state.get('updated_at', '')}")
    print(f"mode: {state.get('active_mode', '')}")
    print(f"backend: {state.get('transcription_backend', '')}")
    if state.get("recognition_mode"):
        label = state.get("recognition_mode_label", state.get("recognition_mode"))
        print(f"recognition_mode: {state.get('recognition_mode')} / {label}")
    print(f"wake_word: {state.get('wake_word', '')}")
    print(f"llm: {state.get('llm_provider', '')} / {state.get('llm_model', '')}")
    print(f"stt_realtime: {state.get('whisper_realtime_model', '')}")
    print(f"stt_final: {state.get('whisper_model', '')}")
    if state.get("recognized_fast"):
        print(f"recognized_fast: {state.get('recognized_fast')}")
    if state.get("recognized_final"):
        print(f"recognized_final: {state.get('recognized_final')}")
    if state.get("last_reply"):
        print(f"last_reply: {state.get('last_reply')}")
    if state.get("last_command"):
        print(f"last_command: {state.get('last_command')}")
    if state.get("last_reset"):
        print(f"last_reset: {state.get('last_reset')}")


def resolve_config_key(cfg: dict, key: str) -> Any:
    value: Any = cfg
    for part in key.split("."):
        if isinstance(value, dict):
            if part not in value:
                raise KeyError(part)
            value = value[part]
            continue
        if isinstance(value, list):
            idx = int(part)
            value = value[idx]
            continue
        raise KeyError(part)
    return value


def print_config(cfg: dict, key: str = ""):
    print_section("Config")
    if not key:
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).rstrip())
        return
    value = resolve_config_key(cfg, key)
    if isinstance(value, (dict, list)):
        print(yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip())
    else:
        print(value)


def print_events(sqlite_path: Path, limit: int, event_types: list[str] | None):
    print_section("Events")
    rows = latest_events(sqlite_path, limit=limit, event_types=event_types)
    if not rows:
        print("(no events)")
        return
    for date, event_type, recognized, payload_json in rows:
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            payload = {}
        if event_type == "command":
            recognized_fast = payload.get("recognized_fast") or recognized or ""
            recognized_final = payload.get("recognized_final") or ""
            command_id = payload.get("command_id") or ""
            command_ok = payload.get("command_ok")
            command_reply = payload.get("command_reply") or ""
            command_input_text = payload.get("command_input_text") or ""
            print(
                f"{date} | command                  | "
                f"fast={recognized_fast} | final={recognized_final} | "
                f"cmd={command_id} | ok={command_ok} | input={command_input_text}"
            )
            if command_reply:
                print(f"{'':19}   reply={command_reply}")
            continue
        if event_type == "command_unknown":
            recognized_fast = payload.get("recognized_fast") or recognized or ""
            command_candidates = payload.get("command_candidates") or []
            print(
                f"{date} | command_unknown          | "
                f"fast={recognized_fast} | final={payload.get('recognized_final', '')} | "
                f"candidates={len(command_candidates)}"
            )
            continue
        if event_type == "wake_check":
            wake_word = payload.get("wake_word") or ""
            matched = payload.get("matched") or False
            matched_wake_word = payload.get("matched_wake_word") or ""
            print(
                f"{date} | wake_check               | "
                f"recognized={recognized or ''} | matched={matched} | "
                f"wake_word={wake_word} | hit={matched_wake_word}"
            )
            continue
        if event_type == "command_reset_no_speech":
            print(f"{date} | command_reset_no_speech   | {recognized or ''}")
            continue
        if event_type == "command_reset_short_input":
            print(f"{date} | command_reset_short_input | {recognized or ''}")
            continue
        if event_type == "command_reset_to_wake":
            print(f"{date} | command_reset_to_wake    | {recognized or ''}")
            continue
        if event_type == "command_candidate_confirmation":
            candidate_id = payload.get("candidate_command_id") or ""
            print(
                f"{date} | command_candidate_confirm | "
                f"recognized={recognized or ''} | candidate={candidate_id}"
            )
            continue
        print(f"{date} | {event_type:24} | {recognized or ''}")


def print_commands(sqlite_path: Path, limit: int):
    print_section("Commands")
    rows = latest_events(sqlite_path, limit=limit, event_types=["command"])
    if not rows:
        print("(no command events)")
        return
    for date, _event_type, recognized, payload_json in rows:
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            payload = {}
        fast = payload.get("recognized_fast") or recognized or ""
        final = payload.get("recognized_final") or ""
        command_id = payload.get("command_id") or ""
        command_ok = payload.get("command_ok")
        command_reply = payload.get("command_reply") or ""
        command_stderr = payload.get("command_stderr") or ""
        print(f"{date} | {fast} -> {command_id} | ok={command_ok}")
        if final and final != fast:
            print(f"{'':19}   final={final}")
        if command_reply:
            print(f"{'':19}   reply={command_reply}")
        if command_stderr:
            print(f"{'':19}   stderr={command_stderr}")


def print_failures(sqlite_path: Path, limit: int):
    print_section("Wake Failures")
    wake_rows = latest_events(sqlite_path, limit=limit * 4, event_types=["wake_check"])
    wake_count = 0
    if not wake_rows:
        print("(no wake events)")
    for date, _event_type, recognized, payload_json in wake_rows:
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            payload = {}
        matched = bool(payload.get("matched"))
        speech_detected = bool(payload.get("speech_detected"))
        recognized_text = payload.get("recognized") or recognized or ""
        if matched or not speech_detected:
            continue
        wake_word = payload.get("wake_word") or ""
        matched_wake_word = payload.get("matched_wake_word") or ""
        print(
            f"{date} | recognized={recognized_text} | wake_word={wake_word} | hit={matched_wake_word}"
        )
        wake_count += 1
        if wake_count >= limit:
            break
    if wake_count == 0 and wake_rows:
        print("(no failed wake detections)")

    print_section("Command Failures")
    command_rows = latest_events(
        sqlite_path, limit=limit * 4, event_types=["command", "command_unknown"]
    )
    command_count = 0
    if not command_rows:
        print("(no command events)")
        return
    for date, event_type, recognized, payload_json in command_rows:
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            payload = {}
        if event_type == "command":
            if payload.get("command_ok") is True:
                continue
            fast = payload.get("recognized_fast") or recognized or ""
            final = payload.get("recognized_final") or ""
            command_id = payload.get("command_id") or ""
            command_reply = payload.get("command_reply") or ""
            command_stderr = payload.get("command_stderr") or ""
            print(f"{date} | {fast} -> {command_id} | ok={payload.get('command_ok')}")
            if final and final != fast:
                print(f"{'':19}   final={final}")
            if command_reply:
                print(f"{'':19}   reply={command_reply}")
            if command_stderr:
                print(f"{'':19}   stderr={command_stderr}")
            command_count += 1
        elif event_type == "command_unknown":
            fast = payload.get("recognized_fast") or recognized or ""
            final = payload.get("recognized_final") or ""
            candidates = payload.get("command_candidates") or []
            print(f"{date} | {fast} | unknown | candidates={len(candidates)}")
            if final and final != fast:
                print(f"{'':19}   final={final}")
            command_count += 1
        if command_count >= limit:
            break
    if command_count == 0:
        print("(no failed command events)")


def print_conversation(conversation_path: Path, limit: int):
    print_section("Conversation")
    rows = latest_conversation(conversation_path, limit)
    if not rows:
        print("(no conversation log)")
        return
    for row in rows:
        date = row.get("date") or row.get("ts") or ""
        mode = row.get("mode", "")
        raw_user = row.get("raw_user", "")
        corrected_user = row.get("corrected_user", "")
        recognized = row.get("recognized", "")
        text = corrected_user or recognized or raw_user
        print(f"{date} | {mode:12} | raw={raw_user} | corrected={text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show voicechat runtime status and recent data.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="view")

    runtime = subparsers.add_parser("runtime", help="Show current runtime state.")
    runtime.set_defaults(view="runtime")

    config = subparsers.add_parser("config", help="Show config or a specific config key.")
    config.add_argument("key", nargs="?", default="")
    config.set_defaults(view="config")

    events = subparsers.add_parser("events", help="Show recent events.")
    events.add_argument("--limit", type=int, default=15)
    events.add_argument("--types", nargs="*", default=[])
    events.set_defaults(view="events")

    conversation = subparsers.add_parser("conversation", help="Show recent conversation log rows.")
    conversation.add_argument("--limit", type=int, default=10)
    conversation.set_defaults(view="conversation")

    commands = subparsers.add_parser("commands", help="Show recent executed commands.")
    commands.add_argument("--limit", type=int, default=15)
    commands.set_defaults(view="commands")

    failures = subparsers.add_parser("failures", help="Show recent wake and command failures.")
    failures.add_argument("--limit", type=int, default=15)
    failures.set_defaults(view="failures")

    all_view = subparsers.add_parser("all", help="Show runtime, recent events, and full config.")
    all_view.add_argument("--limit", type=int, default=15)
    all_view.set_defaults(view="all")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    view = args.view or "all"
    cfg = load_cfg(args.config_path)
    workdir, sqlite_path, state_path, conversation_path = resolve_paths(cfg)
    state = read_state(state_path)

    if view == "runtime":
        print_runtime(cfg, state, workdir, state_path)
        return 0
    if view == "config":
        print_config(cfg, getattr(args, "key", ""))
        return 0
    if view == "events":
        event_types = getattr(args, "types", []) or None
        print_events(sqlite_path, getattr(args, "limit", 15), event_types)
        return 0
    if view == "conversation":
        print_conversation(conversation_path, getattr(args, "limit", 10))
        return 0
    if view == "commands":
        print_commands(sqlite_path, getattr(args, "limit", 15))
        return 0
    if view == "failures":
        print_failures(sqlite_path, getattr(args, "limit", 15))
        return 0

    print_runtime(cfg, state, workdir, state_path)
    print_events(sqlite_path, getattr(args, "limit", 15), None)
    print_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

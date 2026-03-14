import json
import os
import signal
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    workdir = Path(str(cfg["paths"]["workdir"]))
    sqlite_path = workdir / str(cfg.get("storage", {}).get("sqlite", {}).get("path", "voicechat.db"))
    state_path = workdir / "runtime_state.json"
    return workdir, sqlite_path, state_path


def read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_events(sqlite_path: Path, limit: int = 10) -> list[tuple]:
    if not sqlite_path.exists():
        return []
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select date, event_type, recognized
            from events
            where recognized is not null and recognized != ''
            order by ts desc
            limit ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def print_section(title: str):
    print(f"\n[{title}]")


def main() -> int:
    cfg = load_cfg(DEFAULT_CONFIG)
    workdir, sqlite_path, state_path = resolve_paths(cfg)
    state = read_state(state_path)
    pid = int(state.get("pid", 0) or 0)
    running = pid_alive(pid) if pid else False

    print_section("Runtime")
    print(f"running: {'yes' if running else 'no'}")
    print(f"workdir: {workdir}")
    print(f"state_file: {state_path}")
    if state:
        print(f"pid: {pid}")
        print(f"state: {state.get('state', '(unknown)')}")
        print(f"updated_at: {state.get('updated_at', '')}")
        print(f"mode: {state.get('active_mode', '')}")
        print(f"backend: {state.get('transcription_backend', '')}")
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
    else:
        print("state: (no runtime state file)")

    print_section("Latest Events")
    for date, event_type, recognized in latest_events(sqlite_path, limit=15):
        print(f"{date} | {event_type:24} | {recognized}")

    print_section("Config")
    print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

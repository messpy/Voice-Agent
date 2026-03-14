import argparse
import json
import sqlite3
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def load_cfg(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def resolve_sqlite_path(cfg: dict) -> Path:
    workdir = Path(str(cfg["paths"]["workdir"]))
    sqlite_cfg = cfg.get("storage", {}).get("sqlite", {})
    return workdir / str(sqlite_cfg.get("path", "voicechat.db"))


def fetch_unknown_events(sqlite_path: Path, limit: int) -> list[tuple]:
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select id, date, recognized, payload_json
            from events
            where event_type = 'command_unknown'
            order by ts desc
            limit ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def fetch_event_recognized(sqlite_path: Path, event_id: int, expected_type: str) -> str:
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select event_type, recognized
            from events
            where id = ?
            """,
            (event_id,),
        )
        row = cursor.fetchone()
    if not row:
        raise SystemExit(f"event not found: {event_id}")
    event_type, recognized = row
    if event_type != expected_type:
        raise SystemExit(f"event {event_id} is {event_type}, expected {expected_type}")
    text = normalize_text(str(recognized or ""))
    if not text:
        raise SystemExit(f"event {event_id} has empty recognized text")
    return text


def upsert_alias(sqlite_path: Path, *, alias_type: str, target: str, alias: str, source: str) -> None:
    now_ts = int(time.time())
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            insert into recognition_aliases(alias_type, target, alias, source, hits, last_seen_ts, enabled)
            values (?, ?, ?, ?, 1, ?, 1)
            on conflict(alias_type, target, alias)
            do update set
                hits = recognition_aliases.hits + 1,
                last_seen_ts = excluded.last_seen_ts,
                enabled = 1
            """,
            (alias_type, normalize_text(target), normalize_text(alias), source, now_ts),
        )


def list_command_defs(cfg: dict) -> list[tuple[str, list[str]]]:
    rows = []
    for item in cfg.get("command_router", {}).get("commands", []):
        command_id = normalize_text(str(item.get("id", "")))
        phrases = [normalize_text(str(p)) for p in item.get("phrases", []) if normalize_text(str(p))]
        if command_id:
            rows.append((command_id, phrases))
    return rows


def cmd_unknowns(args: argparse.Namespace) -> int:
    cfg = load_cfg(args.config)
    sqlite_path = resolve_sqlite_path(cfg)
    rows = fetch_unknown_events(sqlite_path, args.limit)
    for event_id, date, recognized, payload_json in rows:
        payload = json.loads(payload_json or "{}")
        print(f"{event_id}\t{date}\t{recognized}\t{payload.get('command_candidates', [])}")
    return 0


def cmd_commands(args: argparse.Namespace) -> int:
    cfg = load_cfg(args.config)
    for command_id, phrases in list_command_defs(cfg):
        joined = ", ".join(phrases)
        print(f"{command_id}\t{joined}")
    return 0


def cmd_add_command_alias(args: argparse.Namespace) -> int:
    cfg = load_cfg(args.config)
    sqlite_path = resolve_sqlite_path(cfg)
    valid_ids = {command_id for command_id, _ in list_command_defs(cfg)}
    if args.command_id not in valid_ids:
        raise SystemExit(f"unknown command_id: {args.command_id}")
    alias = normalize_text(args.alias) if args.alias else fetch_event_recognized(sqlite_path, args.event_id, "command_unknown")
    upsert_alias(
        sqlite_path,
        alias_type="command",
        target=args.command_id,
        alias=alias,
        source="manual",
    )
    print(f"saved command alias: {alias} -> {args.command_id}")
    return 0


def cmd_add_wake_alias(args: argparse.Namespace) -> int:
    cfg = load_cfg(args.config)
    sqlite_path = resolve_sqlite_path(cfg)
    wake_words = {normalize_text(str(item)) for item in cfg.get("wake", {}).get("words", []) if normalize_text(str(item))}
    wake_words.add(normalize_text(str(cfg.get("wake", {}).get("word", ""))))
    if normalize_text(args.target) not in wake_words:
        raise SystemExit(f"unknown wake target: {args.target}")
    alias = normalize_text(args.alias)
    upsert_alias(
        sqlite_path,
        alias_type="wake",
        target=args.target,
        alias=alias,
        source="manual",
    )
    print(f"saved wake alias: {alias} -> {args.target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage learned recognition aliases for voicechat.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    unknowns = subparsers.add_parser("unknowns", help="List recent unknown command events.")
    unknowns.add_argument("--limit", type=int, default=20)
    unknowns.set_defaults(func=cmd_unknowns)

    commands = subparsers.add_parser("commands", help="List configured command IDs.")
    commands.set_defaults(func=cmd_commands)

    add_command_alias = subparsers.add_parser("add-command-alias", help="Register a manual alias for a command.")
    add_command_alias.add_argument("--command-id", required=True)
    add_command_alias.add_argument("--event-id", type=int)
    add_command_alias.add_argument("--alias")
    add_command_alias.set_defaults(func=cmd_add_command_alias)

    add_wake_alias = subparsers.add_parser("add-wake-alias", help="Register a manual alias for a wake word.")
    add_wake_alias.add_argument("--target", required=True)
    add_wake_alias.add_argument("--alias", required=True)
    add_wake_alias.set_defaults(func=cmd_add_wake_alias)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "event_id", None) is None and getattr(args, "alias", None) is None and args.command == "add-command-alias":
        parser.error("add-command-alias requires either --event-id or --alias")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

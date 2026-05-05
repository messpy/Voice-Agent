from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import command_runtime
from src.config_loader import dump_cfg, load_cfg, resolve_config_path
from src.runtime_files import atomic_write_text, read_json_file


STATIC_DIR = ROOT / "src" / "voicechat_console" / "static"


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


def resolve_paths(cfg: dict) -> dict[str, Path]:
    workdir = Path(str(cfg["paths"]["workdir"])).expanduser()
    sqlite_path = workdir / str(
        cfg.get("storage", {}).get("sqlite", {}).get("path", "voicechat.db")
    )
    jsonl_path = workdir / str(
        cfg.get("storage", {}).get("jsonl", {}).get("path", "events.jsonl")
    )
    return {
        "workdir": workdir,
        "state_path": workdir / "runtime_state.json",
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
    }


def runtime_payload() -> dict:
    cfg = load_cfg()
    paths = resolve_paths(cfg)
    state = read_json_file(paths["state_path"])
    pid = int(state.get("pid", 0) or 0)
    return {
        "config_path": str(resolve_config_path()),
        "workdir": str(paths["workdir"]),
        "running": pid_alive(pid) if pid else False,
        "state": state,
    }


def latest_events(sqlite_path: Path, limit: int, event_type: str = "") -> list[dict]:
    if not sqlite_path.exists():
        return []
    query = """
        select id, ts, date, event_type, recognized, payload_json
        from events
    """
    params: list[object] = []
    if event_type:
        query += " where event_type = ?"
        params.append(event_type)
    query += " order by ts desc limit ?"
    params.append(limit)
    rows: list[dict] = []
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(query, params)
        for event_id, ts, date, row_type, recognized, payload_json in cursor.fetchall():
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            rows.append(
                {
                    "id": event_id,
                    "ts": ts,
                    "date": date,
                    "event_type": row_type,
                    "recognized": recognized or "",
                    "payload": payload,
                }
            )
    return rows


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False) + "\n")


def append_event_logs(
    *,
    payload: dict,
    event_type: str,
    jsonl_enabled: bool,
    jsonl_path: Path,
    sqlite_enabled: bool,
    sqlite_path: Path,
) -> None:
    if jsonl_enabled:
        append_jsonl(jsonl_path, payload)
    if sqlite_enabled:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as conn:
            conn.execute(
                """
                insert into events (
                    ts, date, event_type, mode, backend, model,
                    expected, recognized, elapsed_sec, payload_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(payload.get("ts") or time.time()),
                    str(payload.get("date") or time.strftime("%Y-%m-%d %H:%M:%S")),
                    event_type,
                    payload.get("mode"),
                    payload.get("backend"),
                    payload.get("model"),
                    payload.get("expected"),
                    payload.get("recognized"),
                    payload.get("elapsed_sec"),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )


def validate_config(cfg: object) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    for key in ["paths", "runtime", "storage", "command_router"]:
        if key not in cfg:
            raise ValueError(f"missing top-level key: {key}")
    return cfg


def save_config_yaml(yaml_text: str) -> dict:
    parsed = yaml.safe_load(yaml_text) or {}
    cfg = validate_config(parsed)
    config_path = resolve_config_path()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_name(f"{config_path.name}.bak.{stamp}")
    backup_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    if backup_text:
        atomic_write_text(backup_path, backup_text)
    atomic_write_text(config_path, dump_cfg(cfg))
    return {
        "ok": True,
        "config_path": str(config_path),
        "backup_path": str(backup_path) if backup_text else "",
    }


def build_ai_alias_map(ai_control_cfg: dict) -> dict[str, dict]:
    items = {}
    for item in ai_control_cfg.get("aliases", []):
        alias = command_runtime.normalize_text(str(item.get("alias", "")))
        if alias:
            items[alias] = item
    return items


def save_config_object(cfg: dict) -> dict:
    return save_config_yaml(dump_cfg(cfg))


def command_catalog(cfg: dict) -> list[dict]:
    router = cfg.get("command_router", {})
    rows: list[dict] = []
    for item in router.get("commands", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "id": str(item.get("id", "")).strip(),
                "help_label": str(item.get("help_label", "")).strip(),
                "phrases": [
                    str(phrase).strip()
                    for phrase in item.get("phrases", [])
                    if str(phrase).strip()
                ],
                "reply": str(item.get("reply", "")).strip(),
                "action_type": str(item.get("action_type", "")).strip(),
                "action_name": str(item.get("action_name", "")).strip(),
                "args": item.get("args", {}) if isinstance(item.get("args", {}), dict) else {},
            }
        )
    return rows


def add_command_to_config(payload: dict) -> dict:
    cfg = load_cfg()
    router = cfg.setdefault("command_router", {})
    commands = router.setdefault("commands", [])
    if not isinstance(commands, list):
        raise ValueError("command_router.commands must be a list")

    command_id = command_runtime.normalize_text(str(payload.get("id", "")))
    help_label = command_runtime.normalize_text(str(payload.get("help_label", "")))
    reply = command_runtime.normalize_text(str(payload.get("reply", "")))
    action_type = command_runtime.normalize_text(
        str(payload.get("action_type", "external_cli"))
    ) or "external_cli"
    action_name = command_runtime.normalize_text(str(payload.get("action_name", "")))
    phrases_raw = payload.get("phrases", [])
    args = payload.get("args", {})

    if isinstance(phrases_raw, str):
        phrases = [
            command_runtime.normalize_text(part)
            for part in re.split(r"[\n,、]+", phrases_raw)
            if command_runtime.normalize_text(part)
        ]
    else:
        phrases = [
            command_runtime.normalize_text(str(part))
            for part in phrases_raw
            if command_runtime.normalize_text(str(part))
        ]

    if not command_id:
        raise ValueError("id is required")
    if not phrases:
        raise ValueError("phrases is required")
    if not action_name:
        raise ValueError("action_name is required")
    if not isinstance(args, dict):
        raise ValueError("args must be an object")

    for item in commands:
        if command_runtime.normalize_text(str(item.get("id", ""))) == command_id:
            raise ValueError(f"command id already exists: {command_id}")

    new_item = {
        "id": command_id,
        "help_label": help_label or phrases[0],
        "phrases": phrases,
        "reply": reply,
        "action_type": action_type,
        "action_name": action_name,
        "args": args,
    }
    commands.append(new_item)
    save_info = save_config_object(cfg)
    return {"ok": True, "command": new_item, "save_info": save_info}


def handle_internal_command(command_hit: dict, cfg: dict, state: dict) -> dict:
    action_type = command_runtime.normalize_text(str(command_hit.get("action_type", "")))
    ai_alias_map = build_ai_alias_map(cfg.get("ai_control", {}))
    recognition_profiles = cfg.get("recognition_profiles", {})
    runtime_cfg = cfg.setdefault("runtime", {})
    llm_cfg = cfg.setdefault("llm", {})
    running = pid_alive(int(state.get("pid", 0) or 0))

    if action_type == "internal_show_recognition_mode":
        current_mode = command_runtime.normalize_text(
            str(state.get("recognition_mode") or runtime_cfg.get("recognition_mode", ""))
        )
        profile = recognition_profiles.get(current_mode, {})
        label = (
            command_runtime.normalize_text(str(profile.get("label", current_mode)))
            or current_mode
        )
        description = command_runtime.normalize_text(str(profile.get("description", "")))
        reply = f"今の認識モードは {label} なのだ。"
        if description:
            reply += f" {description}"
        return {"ok": True, "command_id": command_hit.get("id"), "reply": reply}

    if action_type == "internal_set_recognition_mode":
        target_mode = command_runtime.normalize_text(str(command_hit.get("recognition_mode", "")))
        if not target_mode or target_mode not in recognition_profiles:
            return {
                "ok": False,
                "error": "unknown_recognition_mode",
                "command_id": command_hit.get("id"),
                "reply": "その認識モードは見つからないのだ。",
            }
        updated = copy.deepcopy(cfg)
        updated.setdefault("runtime", {})["recognition_mode"] = target_mode
        save_info = save_config_object(updated)
        label = (
            command_runtime.normalize_text(
                str(recognition_profiles[target_mode].get("label", target_mode))
            )
            or target_mode
        )
        reply = f"認識モードを {label} に保存したのだ。"
        if running:
            reply += " 実行中の voicechat にはまだ反映されないので、再起動後に有効になるのだ。"
        return {
            "ok": True,
            "command_id": command_hit.get("id"),
            "reply": reply,
            "config_saved": True,
            "save_info": save_info,
        }

    if action_type == "internal_set_model_alias":
        alias_name = command_runtime.normalize_text(str(command_hit.get("ai_alias", "")))
        alias_cfg = ai_alias_map.get(alias_name)
        if not alias_cfg:
            return {
                "ok": False,
                "error": "unknown_ai_alias",
                "command_id": command_hit.get("id"),
                "reply": "そのAI設定は見つからないのだ。",
            }
        updated = copy.deepcopy(cfg)
        updated.setdefault("llm", {})["provider"] = str(alias_cfg["provider"])
        updated.setdefault("llm", {})["model"] = str(alias_cfg["model"])
        save_info = save_config_object(updated)
        reply = f"AIを {alias_cfg['provider']} の {alias_cfg['model']} に保存したのだ。"
        if running:
            reply += " 実行中の voicechat にはまだ反映されないので、再起動後に有効になるのだ。"
        return {
            "ok": True,
            "command_id": command_hit.get("id"),
            "reply": reply,
            "config_saved": True,
            "save_info": save_info,
        }

    if action_type == "internal_pull_model_alias":
        alias_name = command_runtime.normalize_text(str(command_hit.get("ai_alias", "")))
        alias_cfg = ai_alias_map.get(alias_name)
        if not alias_cfg:
            return {
                "ok": False,
                "error": "unknown_ai_alias",
                "command_id": command_hit.get("id"),
                "reply": "そのモデル設定は見つからないのだ。",
            }
        pull_cmd = alias_cfg.get("pull_command", [])
        if not isinstance(pull_cmd, list) or not pull_cmd:
            return {
                "ok": False,
                "error": "missing_pull_command",
                "command_id": command_hit.get("id"),
                "reply": "pull コマンドが未設定なのだ。",
            }
        proc = subprocess.run(
            [str(part) for part in pull_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            detail = command_runtime.normalize_text(stderr or stdout or f"rc={proc.returncode}")
            return {
                "ok": False,
                "error": "pull_failed",
                "command_id": command_hit.get("id"),
                "reply": f"pull に失敗したのだ。{detail}",
                "stdout": stdout,
                "stderr": stderr,
            }
        return {
            "ok": True,
            "command_id": command_hit.get("id"),
            "reply": f"{alias_cfg.get('model', alias_name)} を pull したのだ。",
            "stdout": stdout,
            "stderr": stderr,
        }

    return {
        "ok": False,
        "error": "unsupported_internal_command",
        "command_id": command_hit.get("id"),
        "reply": "その内部コマンドはブラウザ版ではまだ未対応なのだ。",
    }


def execute_web_command(text: str) -> dict:
    cfg = load_cfg()
    command_router_cfg = cfg.get("command_router", {})
    runtime_cfg = cfg.get("runtime", {})
    llm_cfg = cfg.get("llm", {})
    paths = resolve_paths(cfg)
    state = read_json_file(paths["state_path"])
    workdir = paths["workdir"]
    sqlite_cfg = cfg.get("storage", {}).get("sqlite", {})
    jsonl_cfg = cfg.get("storage", {}).get("jsonl", {})
    command_execution_enabled = bool(runtime_cfg.get("command_execution", True))

    command_hit = command_runtime.resolve_playback_context_command(text, command_router_cfg)
    if not command_hit:
        command_hit = command_runtime.match_command(text, command_router_cfg)
    if not command_hit:
        return {
            "ok": False,
            "error": "unknown_command",
            "reply": "その操作はまだ分からないのだ。",
        }

    action_type = command_runtime.normalize_text(str(command_hit.get("action_type", "")))
    if action_type.startswith("internal_"):
        result = handle_internal_command(command_hit, cfg, state)
        payload = {
            "ts": int(time.time()),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "command",
            "mode": "web_console",
            "backend": "web_console",
            "model": "",
            "recognized": text,
            "raw_user": text,
            "corrected_user": text,
            "recognized_fast": text,
            "recognized_final": text,
            "model_fast": "",
            "model_final": "",
            "command_input_text": text,
            "command_candidates": [],
            "elapsed_sec": 0.0,
            "command_id": command_hit.get("id"),
            "command_reply": result.get("reply", ""),
            "command_ok": result.get("ok"),
            "command_returncode": 0 if result.get("ok") else -1,
            "command_stdout": command_runtime.normalize_text(str(result.get("stdout", ""))),
            "command_stderr": command_runtime.normalize_text(str(result.get("stderr", ""))),
            "command_skipped": False,
            "llm_provider_after": llm_cfg.get("provider"),
            "llm_model_after": llm_cfg.get("model"),
            "source": "web_console",
        }
        append_event_logs(
            payload=payload,
            event_type="command",
            jsonl_enabled=bool(jsonl_cfg.get("enabled", True)),
            jsonl_path=workdir / str(jsonl_cfg.get("path", "events.jsonl")),
            sqlite_enabled=bool(sqlite_cfg.get("enabled", True)),
            sqlite_path=workdir / str(sqlite_cfg.get("path", "voicechat.db")),
        )
        return result

    command_result = command_runtime.execute_command_action(
        command_hit,
        command_router_cfg,
        command_execution_enabled,
    )
    reply_text = command_runtime.render_command_reply(
        str(command_hit.get("reply", "")),
        command_hit,
    ) or command_runtime.normalize_text(
        str(command_router_cfg.get("fallback_reply", ""))
    )
    if command_runtime.normalize_text(str(command_result.get("message", ""))):
        reply_text = command_runtime.normalize_text(str(command_result.get("message", "")))

    payload = {
        "ts": int(time.time()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": "command",
        "mode": "web_console",
        "backend": "web_console",
        "model": "",
        "recognized": text,
        "raw_user": text,
        "corrected_user": text,
        "recognized_fast": text,
        "recognized_final": text,
        "model_fast": "",
        "model_final": "",
        "command_input_text": text,
        "command_candidates": [],
        "elapsed_sec": 0.0,
        "command_id": command_hit.get("id"),
        "command_reply": reply_text,
        "command_ok": command_result.get("ok"),
        "command_returncode": command_result.get("returncode"),
        "command_stdout": command_runtime.normalize_text(
            str(command_result.get("stdout", ""))
        ),
        "command_stderr": command_runtime.normalize_text(
            str(command_result.get("stderr", ""))
        ),
        "command_skipped": bool(command_result.get("skipped", False)),
        "llm_provider_after": llm_cfg.get("provider"),
        "llm_model_after": llm_cfg.get("model"),
        "source": "web_console",
    }
    append_event_logs(
        payload=payload,
        event_type="command",
        jsonl_enabled=bool(jsonl_cfg.get("enabled", True)),
        jsonl_path=workdir / str(jsonl_cfg.get("path", "events.jsonl")),
        sqlite_enabled=bool(sqlite_cfg.get("enabled", True)),
        sqlite_path=workdir / str(sqlite_cfg.get("path", "voicechat.db")),
    )

    return {
        "ok": bool(command_result.get("ok")),
        "command_id": command_hit.get("id"),
        "reply": reply_text,
        "result": command_result,
    }


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "voicechat-console/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runtime":
            return self._send_json(runtime_payload())
        if parsed.path == "/api/events":
            params = parse_qs(parsed.query)
            limit = max(1, min(200, int(params.get("limit", ["50"])[0])))
            event_type = params.get("type", [""])[0]
            cfg = load_cfg()
            paths = resolve_paths(cfg)
            return self._send_json(
                {"items": latest_events(paths["sqlite_path"], limit, event_type)}
            )
        if parsed.path == "/api/config":
            cfg = load_cfg()
            return self._send_json(
                {
                    "config_path": str(resolve_config_path()),
                    "yaml_text": dump_cfg(cfg),
                }
            )
        if parsed.path == "/api/commands/catalog":
            cfg = load_cfg()
            return self._send_json({"items": command_catalog(cfg)})
        return self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json_body()
        if parsed.path == "/api/commands/execute":
            text = str(body.get("text", "")).strip()
            if not text:
                return self._send_json(
                    {"ok": False, "error": "missing_text"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            result = execute_web_command(text)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            return self._send_json(result, status=status)
        if parsed.path == "/api/commands/catalog":
            try:
                result = add_command_to_config(body)
            except Exception as exc:
                return self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return self._send_json(result)
        return self._send_json(
            {"ok": False, "error": "not_found"},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json_body()
        if parsed.path == "/api/config":
            yaml_text = str(body.get("yaml_text", ""))
            if not yaml_text.strip():
                return self._send_json(
                    {"ok": False, "error": "missing_yaml_text"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            try:
                result = save_config_yaml(yaml_text)
            except Exception as exc:
                return self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return self._send_json(result)
        return self._send_json(
            {"ok": False, "error": "not_found"},
            status=HTTPStatus.NOT_FOUND,
        )

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = STATIC_DIR / rel
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/plain; charset=utf-8"
        if target.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the voicechat web console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    print(f"INFO: voicechat web console listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

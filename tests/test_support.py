from __future__ import annotations

import base64
import json
import os
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0fcAAAAASUVORK5CYII="
)


class FakeLlmHandler(BaseHTTPRequestHandler):
    routes: dict[tuple[str, str], dict[str, Any]] = {}
    requests_log: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/version":
            self._send_json(200, {"version": "test"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length else b""
        body_text = raw_body.decode("utf-8", errors="replace")
        try:
            payload = json.loads(body_text) if body_text else {}
        except json.JSONDecodeError:
            payload = {"_raw": body_text}
        self.requests_log.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers.items()),
                "payload": payload,
            }
        )
        response = self.routes.get((self.command, self.path))
        if response is None:
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(int(response.get("status", 200)), response.get("body", {}))

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        blob = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


class FakeLlmServer:
    def __init__(self) -> None:
        FakeLlmHandler.routes = {}
        FakeLlmHandler.requests_log = []
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            host, port = probe.getsockname()
        self._server = ThreadingHTTPServer((host, port), FakeLlmHandler)
        self.base_url = f"http://{host}:{port}"
        self.routes = FakeLlmHandler.routes
        self.requests = FakeLlmHandler.requests_log
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)


class ProdLikeEnv:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workdir = self.tmp_path / "runtime"
        self.config_dir = self.tmp_path / "config"
        self.bin_dir = self.tmp_path / "bin"
        self.workdir.mkdir()
        self.config_dir.mkdir()
        self.bin_dir.mkdir()

        self.image_path = self.tmp_path / "sample.png"
        self.image_path.write_bytes(base64.b64decode(PNG_1X1_BASE64))

        self.codex_log = self.tmp_path / "codex_invocations.jsonl"
        self.gemini_log = self.tmp_path / "gemini_invocations.jsonl"

        self._write_fake_binaries()
        self.config_path = self.config_dir / "config.yaml"
        self.write_config(
            {
                "paths": {"workdir": str(self.workdir)},
                "runtime": {"command_execution": False, "recognition_mode": "balanced"},
                "storage": {"sqlite": {"path": "voicechat.db"}, "jsonl": {"path": "events.jsonl"}},
                "command_router": {"enabled": True, "commands": []},
                "llm": {
                    "provider": "codex",
                    "model": "gpt-5",
                    "timeout_sec": 30,
                    "command": "codex",
                    "sandbox": "read-only",
                    "skip_git_repo_check": True,
                    "workdir": str(self.tmp_path),
                },
            }
        )

        self._old_env = {
            "VOICECHAT_CONFIG": os.environ.get("VOICECHAT_CONFIG"),
            "VOICECHAT_TEST_CODEX_LOG": os.environ.get("VOICECHAT_TEST_CODEX_LOG"),
            "VOICECHAT_TEST_GEMINI_LOG": os.environ.get("VOICECHAT_TEST_GEMINI_LOG"),
            "PATH": os.environ.get("PATH"),
        }
        os.environ["VOICECHAT_CONFIG"] = str(self.config_path)
        os.environ["VOICECHAT_TEST_CODEX_LOG"] = str(self.codex_log)
        os.environ["VOICECHAT_TEST_GEMINI_LOG"] = str(self.gemini_log)
        os.environ["PATH"] = str(self.bin_dir) + os.pathsep + (os.environ.get("PATH", ""))

    def _write_fake_binaries(self) -> None:
        codex_script = self.bin_dir / "codex"
        codex_script.write_text(
            """#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ["VOICECHAT_TEST_CODEX_LOG"])
entry = {"argv": sys.argv[1:]}
with log_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry, ensure_ascii=False) + "\\n")
args = sys.argv[1:]
out_path = ""
for idx, part in enumerate(args):
    if part == "--output-last-message" and idx + 1 < len(args):
        out_path = args[idx + 1]
if out_path:
    Path(out_path).write_text("codex image answer\\n", encoding="utf-8")
print("codex stdout fallback")
""",
            encoding="utf-8",
        )
        codex_script.chmod(0o755)

        gemini_script = self.bin_dir / "gemini"
        gemini_script.write_text(
            """#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ["VOICECHAT_TEST_GEMINI_LOG"])
entry = {"argv": sys.argv[1:]}
with log_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry, ensure_ascii=False) + "\\n")
args = sys.argv[1:]
if "--output-format" in args:
    idx = args.index("--output-format")
    fmt = args[idx + 1]
else:
    fmt = "text"
if fmt == "json":
    print(json.dumps({"text": "gemini cli json answer"}, ensure_ascii=False))
else:
    print("gemini cli text answer")
""",
            encoding="utf-8",
        )
        gemini_script.chmod(0o755)

    def write_config(self, data: dict[str, Any]) -> None:
        self.config_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def close(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

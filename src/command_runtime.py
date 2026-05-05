from __future__ import annotations

import json
import os
import re
import subprocess
import unicodedata


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


KANJI_DIGIT_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_japanese_number_token(token: str) -> int | None:
    token = unicodedata.normalize("NFKC", normalize_text(token))
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if any(ch not in KANJI_DIGIT_MAP and ch != "十" for ch in token):
        return None
    if "十" in token:
        left, right = token.split("十", 1)
        tens = KANJI_DIGIT_MAP.get(left, 1) if left else 1
        if left and left not in KANJI_DIGIT_MAP:
            return None
        if not right:
            ones = 0
        elif all(ch in KANJI_DIGIT_MAP for ch in right):
            ones = int("".join(str(KANJI_DIGIT_MAP[ch]) for ch in right))
        else:
            return None
        return tens * 10 + ones
    if all(ch in KANJI_DIGIT_MAP for ch in token):
        return int("".join(str(KANJI_DIGIT_MAP[ch]) for ch in token))
    return None


def normalize_japanese_time_text(text: str) -> str:
    body = unicodedata.normalize("NFKC", text)

    def repl(match: re.Match[str]) -> str:
        value = parse_japanese_number_token(match.group(1))
        if value is None:
            return match.group(0)
        return f"{value}{match.group(2)}"

    return re.sub(r"([零〇一二三四五六七八九十0-9]+)(時|分)", repl, body)


def coerce_dynamic_capture(raw_value: str | None, spec: dict | str | None) -> object:
    if isinstance(spec, str):
        spec = {"target": spec}
    spec = spec or {}
    if raw_value in (None, ""):
        return spec.get("default")
    cast = str(spec.get("type", "str")).strip()
    if cast == "int":
        try:
            value = int(raw_value)
        except ValueError:
            value = spec.get("default")
    elif cast == "times_ten":
        try:
            value = int(raw_value) * 10
        except ValueError:
            value = spec.get("default")
    elif cast == "direct_percent":
        try:
            value = int(raw_value)
        except ValueError:
            value = spec.get("default")
    else:
        value = raw_value
    if isinstance(value, int):
        if "min" in spec:
            value = max(int(spec["min"]), value)
        if "max" in spec:
            value = min(int(spec["max"]), value)
    return value


def match_dynamic_command(text: str, command_router_cfg: dict) -> dict | None:
    body = compact_text(normalize_japanese_time_text(text)).lower()
    if not body:
        return None
    patterns = command_router_cfg.get("dynamic_patterns", [])
    if not isinstance(patterns, list):
        return None
    for item in patterns:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern", "")).strip()
        if not pattern:
            continue
        matched = re.search(pattern, body)
        if not matched:
            continue
        command = dict(item)
        args = (
            dict(command.get("args", {}))
            if isinstance(command.get("args", {}), dict)
            else {}
        )
        captures = item.get("captures", {})
        if isinstance(captures, dict) and captures:
            for group_name, spec in captures.items():
                target = (
                    spec.get("target", group_name)
                    if isinstance(spec, dict)
                    else str(spec)
                )
                value = coerce_dynamic_capture(
                    matched.groupdict().get(group_name),
                    spec,
                )
                if target and value is not None:
                    args[str(target)] = value
        command["args"] = args
        return command
    return None


def match_command(text: str, command_router_cfg: dict) -> dict | None:
    if not command_router_cfg.get("enabled", False):
        return None
    dynamic_hit = match_dynamic_command(text, command_router_cfg)
    if dynamic_hit:
        return dynamic_hit
    body = compact_text(text).lower()
    if not body:
        return None
    for item in command_router_cfg.get("commands", []):
        phrases = item.get("phrases", [])
        for phrase in phrases:
            if compact_text(str(phrase)).lower() in body:
                return item
    return None


def find_command_by_id(command_id: str, command_router_cfg: dict) -> dict | None:
    target = normalize_text(command_id)
    if not target:
        return None
    for item in command_router_cfg.get("commands", []):
        if normalize_text(str(item.get("id", ""))) == target:
            return item
    return None


def resolve_playback_context_command(text: str, command_router_cfg: dict) -> dict | None:
    direct = match_command(text, command_router_cfg)
    if direct:
        return direct

    body = compact_text(text).lower()
    if not body:
        return None

    stop_aliases = {
        compact_text(item).lower()
        for item in ["止めて", "とめて", "止める", "とめる", "ストップ"]
    }
    next_aliases = {
        compact_text(item).lower()
        for item in ["次", "次の曲", "次の歌", "次にして", "スキップ", "スキップして"]
    }
    prev_aliases = {
        compact_text(item).lower()
        for item in ["前", "前の曲", "前の歌", "前にして", "戻して", "ひとつ前"]
    }
    volume_up_aliases = {
        compact_text(item).lower()
        for item in ["上げて", "あげて", "大きくして", "大きく"]
    }
    volume_down_aliases = {
        compact_text(item).lower()
        for item in ["下げて", "さげて", "小さくして", "小さく"]
    }
    volume_query_aliases = {
        compact_text(item).lower() for item in ["音量", "いくつ", "どれくらい"]
    }

    if body in stop_aliases:
        return find_command_by_id("music_stop", command_router_cfg)
    if body in next_aliases:
        return find_command_by_id("music_next", command_router_cfg)
    if body in prev_aliases:
        return find_command_by_id("music_prev", command_router_cfg)
    if body in volume_up_aliases:
        return find_command_by_id("volume_up", command_router_cfg)
    if body in volume_down_aliases:
        return find_command_by_id("volume_down", command_router_cfg)
    if body in volume_query_aliases:
        return find_command_by_id("volume_query", command_router_cfg)
    return None


def render_command_reply(template: str, command_hit: dict) -> str:
    body = normalize_text(template)
    if not body:
        return ""
    args = command_hit.get("args", {})
    if not isinstance(args, dict):
        return body
    try:
        return normalize_text(body.format(**args))
    except Exception:
        return body


def build_action_payload(item: dict) -> dict:
    args = item.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return {
        "action_type": normalize_text(str(item.get("action_type", ""))),
        "action_name": normalize_text(str(item.get("action_name", ""))),
        "args": args,
    }


def execute_action_runner(
    item: dict,
    command_router_cfg: dict,
    command_execution_enabled: bool,
) -> dict:
    payload = build_action_payload(item)
    action_type = payload["action_type"]
    if not action_type:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "missing action_type",
        }
    runner_cfg = command_router_cfg.get("action_runners", {}).get(action_type, {})
    if not isinstance(runner_cfg, dict):
        runner_cfg = {}
    runner_command = runner_cfg.get("command", [])
    if not isinstance(runner_command, list) or not runner_command:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"missing action runner for {action_type}",
        }
    if not command_execution_enabled:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "skipped": True,
            "payload": payload,
        }
    env = os.environ.copy()
    extra_env = runner_cfg.get("env", {})
    if isinstance(extra_env, dict):
        for key, value in extra_env.items():
            env[str(key)] = str(value)
    proc = subprocess.run(
        [str(part) for part in runner_command],
        input=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(runner_cfg.get("cwd", "")) or None,
        env=env,
        check=False,
    )
    stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "payload": payload,
    }
    if stdout_text.strip():
        try:
            parsed = json.loads(stdout_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            result["action_result"] = parsed
            if "success" in parsed:
                result["ok"] = bool(parsed.get("success"))
            if "message" in parsed:
                result["message"] = normalize_text(str(parsed.get("message", "")))
    return result


def execute_command_action(
    item: dict,
    command_router_cfg: dict,
    command_execution_enabled: bool,
) -> dict:
    action_type = normalize_text(str(item.get("action_type", "")))
    if action_type:
        return execute_action_runner(item, command_router_cfg, command_execution_enabled)
    command = item.get("command", [])
    if not isinstance(command, list) or not command:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "missing command",
        }
    if not command_execution_enabled:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "skipped": True,
        }
    proc = subprocess.run(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or b"").decode("utf-8", errors="replace"),
        "stderr": (proc.stderr or b"").decode("utf-8", errors="replace"),
    }

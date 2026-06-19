from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from base64 import b64encode
from pathlib import Path
from typing import Any

import requests


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def _normalize_image_paths(image_paths: list[str] | None) -> list[Path]:
    items: list[Path] = []
    for raw in image_paths or []:
        body = str(raw).strip()
        if not body:
            continue
        path = Path(body).expanduser()
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"image not found: {path}")
        items.append(path)
    return items


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _image_base64(path: Path) -> str:
    return b64encode(path.read_bytes()).decode("ascii")


def _append_images_to_last_user_message(
    messages: list[dict[str, Any]],
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    if not image_paths:
        return [dict(msg) for msg in messages]
    out = [dict(msg) for msg in messages]
    for msg in reversed(out):
        if str(msg.get("role", "")).strip().lower() == "user":
            images = list(msg.get("images") or [])
            images.extend(str(path) for path in image_paths)
            msg["images"] = images
            return out
    out.append({"role": "user", "content": "", "images": [str(path) for path in image_paths]})
    return out


def _ollama_prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        item = dict(msg)
        images = []
        for raw_image_path in item.get("images") or []:
            image_path = Path(str(raw_image_path))
            images.append(_image_base64(image_path))
        if images:
            item["images"] = images
        out.append(item)
    return out


def resolve_llm_config(cfg: dict) -> dict[str, Any]:
    llm_cfg = dict(cfg.get("llm", {}))
    if llm_cfg:
        provider = str(llm_cfg.get("provider", "ollama")).strip() or "ollama"
        timeout_sec = int(llm_cfg.get("timeout_sec", 120))
        model = str(llm_cfg.get("model", "")).strip()
        if provider == "ollama":
            api_key_env = str(llm_cfg.get("api_key_env", "OLLAMA_API_KEY"))
            think = llm_cfg.get("think", None)
            return {
                "provider": "ollama",
                "host": str(llm_cfg.get("host", cfg.get("ollama", {}).get("host", "http://127.0.0.1:11434"))),
                "model": model or str(cfg.get("ollama", {}).get("model", "qwen2.5:7b")),
                "timeout_sec": timeout_sec,
                "api_key_env": api_key_env,
                "api_key": os.environ.get(api_key_env, "").strip(),
                "web_search": dict(llm_cfg.get("web_search", {})),
                "think": think,
            }
        if provider == "gemini":
            api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))
            return {
                "provider": "gemini",
                "model": model or "gemini-2.5-flash",
                "timeout_sec": timeout_sec,
                "api_base": str(llm_cfg.get("api_base", "https://generativelanguage.googleapis.com/v1beta")),
                "api_key_env": api_key_env,
                "api_key": os.environ.get(api_key_env, "").strip(),
            }
        if provider == "openai":
            api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
            return {
                "provider": "openai",
                "model": model or "gpt-5-mini",
                "timeout_sec": timeout_sec,
                "api_base": str(llm_cfg.get("api_base", "https://api.openai.com/v1")),
                "api_key_env": api_key_env,
                "api_key": os.environ.get(api_key_env, "").strip(),
            }
        if provider == "anthropic":
            api_key_env = str(llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
            return {
                "provider": "anthropic",
                "model": model or "claude-sonnet-4-20250514",
                "timeout_sec": timeout_sec,
                "api_base": str(llm_cfg.get("api_base", "https://api.anthropic.com/v1")),
                "api_key_env": api_key_env,
                "api_key": os.environ.get(api_key_env, "").strip(),
                "anthropic_version": str(llm_cfg.get("anthropic_version", "2023-06-01")),
            }
        if provider == "codex":
            return {
                "provider": "codex",
                "model": model or "",
                "timeout_sec": timeout_sec,
                "command": str(llm_cfg.get("command", "codex")).strip() or "codex",
                "sandbox": str(llm_cfg.get("sandbox", "read-only")).strip() or "read-only",
                "workdir": str(llm_cfg.get("workdir", os.getcwd())).strip() or os.getcwd(),
                "skip_git_repo_check": bool(llm_cfg.get("skip_git_repo_check", True)),
            }
        if provider == "gemini_cli":
            return {
                "provider": "gemini_cli",
                "model": model or "",
                "timeout_sec": timeout_sec,
                "command": str(llm_cfg.get("command", "gemini")).strip() or "gemini",
                "workdir": str(llm_cfg.get("workdir", os.getcwd())).strip() or os.getcwd(),
                "approval_mode": str(llm_cfg.get("approval_mode", "plan")).strip() or "plan",
                "output_format": str(llm_cfg.get("output_format", "text")).strip() or "text",
                "skip_trust": bool(llm_cfg.get("skip_trust", True)),
            }
        raise RuntimeError(f"unsupported llm provider: {provider}")

    ollama_cfg = cfg.get("ollama", {})
    return {
        "provider": "ollama",
        "host": str(ollama_cfg.get("host", "http://127.0.0.1:11434")),
        "model": str(ollama_cfg.get("model", "qwen2.5:7b")),
        "timeout_sec": int(ollama_cfg.get("timeout_sec", 120)),
        "api_key_env": "OLLAMA_API_KEY",
        "api_key": os.environ.get("OLLAMA_API_KEY", "").strip(),
        "web_search": {},
        "think": None,
    }


def _latest_user_text(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return normalize_text(msg.get("content", ""))
    return ""


def _ollama_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _ollama_web_search(host: str, api_key: str, query: str, max_results: int) -> list[dict[str, str]]:
    if not query:
        return []
    if "ollama.com" not in host:
        return []
    url = "https://ollama.com/api/web_search"
    resp = requests.post(
        url,
        headers=_ollama_headers(api_key),
        json={"query": query, "max_results": max(1, min(max_results, 10))},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    items: list[dict[str, str]] = []
    for item in results:
        title = normalize_text(str(item.get("title", "")))
        link = str(item.get("url", "")).strip()
        snippet = normalize_text(str(item.get("snippet", "")))
        if title or link or snippet:
            items.append({"title": title, "url": link, "snippet": snippet})
    return items


def _augment_messages_with_ollama_web_search(
    llm_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    web_search_cfg = dict(llm_cfg.get("web_search", {}))
    if not web_search_cfg.get("enabled", False):
        return messages
    query = _latest_user_text(messages)
    if not query:
        return messages
    try:
        results = _ollama_web_search(
            str(llm_cfg["host"]),
            str(llm_cfg.get("api_key", "")),
            query,
            int(web_search_cfg.get("max_results", 5)),
        )
    except Exception:
        return messages
    if not results:
        return messages
    lines = ["Web検索結果。必要なときだけ使うこと。"]
    for item in results:
        row = f"- {item['title']}"
        if item["url"]:
            row += f" | {item['url']}"
        if item["snippet"]:
            row += f" | {item['snippet']}"
        lines.append(row)
    return [{"role": "system", "content": "\n".join(lines)}] + messages


def _ollama_chat_messages(
    *,
    host: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    options: dict | None = None,
    web_search: dict[str, Any] | None = None,
    think: Any = None,
    image_paths: list[str] | None = None,
) -> str:
    normalized_images = _normalize_image_paths(image_paths)
    prepared_messages = _append_images_to_last_user_message(messages, normalized_images)
    messages = _augment_messages_with_ollama_web_search(
        {
            "provider": "ollama",
            "host": host,
            "api_key": api_key,
            "web_search": web_search or {},
        },
        _ollama_prepare_messages(prepared_messages),
    )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    r = requests.post(host.rstrip("/") + "/api/chat", json=payload, timeout=timeout_sec, headers=_ollama_headers(api_key))
    if r.status_code == 404:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        fallback = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            fallback["options"] = options
        if think is not None:
            fallback["think"] = think
        r = requests.post(host.rstrip("/") + "/api/generate", json=fallback, timeout=timeout_sec, headers=_ollama_headers(api_key))
        r.raise_for_status()
        return normalize_text(r.json().get("response", ""))
    r.raise_for_status()
    data = r.json()
    content = normalize_text(((data.get("message") or {}).get("content", "")))
    if content:
        return content
    thinking = normalize_text(((data.get("message") or {}).get("thinking", "")))
    if thinking:
        print(
            "WARN: ollama chat returned empty content but thinking present; "
            f"model={model} thinking_len={len(thinking)}"
        )

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    fallback = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        fallback["options"] = options
    if think is not None:
        fallback["think"] = think
    r2 = requests.post(
        host.rstrip("/") + "/api/generate",
        json=fallback,
        timeout=timeout_sec,
        headers=_ollama_headers(api_key),
    )
    r2.raise_for_status()
    response = normalize_text(r2.json().get("response", ""))
    if response:
        return response
    generated_thinking = normalize_text(r2.json().get("thinking", ""))
    if generated_thinking:
        print(
            "WARN: ollama generate returned empty response but thinking present; "
            f"model={model} thinking_len={len(generated_thinking)}"
        )
    return ""


def _gemini_chat_messages(
    *,
    api_base: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    image_paths: list[str] | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("gemini api key is missing")
    normalized_images = _normalize_image_paths(image_paths)
    prepared_messages = _append_images_to_last_user_message(messages, normalized_images)
    contents = []
    for msg in prepared_messages:
        role = "user" if msg.get("role") != "assistant" else "model"
        parts: list[dict[str, Any]] = []
        text = str(msg.get("content", "") or "")
        if text:
            parts.append({"text": text})
        for raw_image_path in msg.get("images") or []:
            image_path = Path(str(raw_image_path))
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _guess_mime_type(image_path),
                        "data": _image_base64(image_path),
                    }
                }
            )
        if not parts:
            parts.append({"text": ""})
        contents.append({"role": role, "parts": parts})
    url = f"{api_base.rstrip('/')}/models/{model}:generateContent"
    resp = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={"contents": contents},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return normalize_text(" ".join(texts))


def _openai_chat_messages(
    *,
    api_base: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    image_paths: list[str] | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("openai api key is missing")
    normalized_images = _normalize_image_paths(image_paths)
    prepared_messages = _append_images_to_last_user_message(messages, normalized_images)
    url = f"{api_base.rstrip('/')}/responses"
    input_items: list[dict[str, Any]] = []
    for msg in prepared_messages:
        role = str(msg.get("role", "user"))
        content: list[dict[str, Any]] = []
        text = str(msg.get("content", "") or "")
        if text:
            content.append({"type": "input_text", "text": text})
        for raw_image_path in msg.get("images") or []:
            image_path = Path(str(raw_image_path))
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{_guess_mime_type(image_path)};base64,{_image_base64(image_path)}",
                }
            )
        if not content:
            content.append({"type": "input_text", "text": ""})
        input_items.append({"role": role, "content": content})
    payload = {
        "model": model,
        "input": input_items,
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    output_text = data.get("output_text")
    if output_text:
        return normalize_text(output_text)
    outputs = data.get("output") or []
    texts: list[str] = []
    for item in outputs:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    return normalize_text(" ".join(texts))


def _anthropic_chat_messages(
    *,
    api_base: str,
    model: str,
    api_key: str,
    anthropic_version: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    image_paths: list[str] | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("anthropic api key is missing")
    normalized_images = _normalize_image_paths(image_paths)
    prepared_messages = _append_images_to_last_user_message(messages, normalized_images)
    system_parts = [msg.get("content", "") for msg in prepared_messages if msg.get("role") == "system"]
    chat_messages = [
        {
            "role": "assistant" if msg.get("role") == "assistant" else "user",
            "content": _anthropic_content_parts(msg),
        }
        for msg in prepared_messages
        if msg.get("role") != "system"
    ]
    url = f"{api_base.rstrip('/')}/messages"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": chat_messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    resp = requests.post(
        url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
        },
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    texts = [part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"]
    return normalize_text(" ".join(texts))


def _anthropic_content_parts(msg: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    text = str(msg.get("content", "") or "")
    if text:
        parts.append({"type": "text", "text": text})
    for raw_image_path in msg.get("images") or []:
        image_path = Path(str(raw_image_path))
        parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _guess_mime_type(image_path),
                    "data": _image_base64(image_path),
                },
            }
        )
    if not parts:
        parts.append({"type": "text", "text": ""})
    return parts


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"[System]\n{content}")
        elif role == "assistant":
            lines.append(f"[Assistant]\n{content}")
        else:
            lines.append(f"[User]\n{content}")
    lines.append("出力は返答本文だけにすること。")
    return "\n\n".join(lines).strip()


def _resolve_cli_command(command: str) -> str:
    body = command.strip()
    if not body:
        raise RuntimeError("cli command is empty")
    if os.path.sep in body:
        if not os.path.exists(body):
            raise RuntimeError(f"cli command not found: {body}")
        return body
    resolved = shutil.which(body)
    if not resolved:
        raise RuntimeError(f"cli command not found in PATH: {body}")
    return resolved


def _codex_chat_messages(
    *,
    command: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    sandbox: str,
    workdir: str,
    skip_git_repo_check: bool,
    image_paths: list[str] | None = None,
) -> str:
    prompt = _messages_to_prompt(messages)
    codex_bin = _resolve_cli_command(command)
    normalized_images = _normalize_image_paths(image_paths)
    with tempfile.NamedTemporaryFile(prefix="voicechat_codex_", suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            codex_bin,
            "exec",
            "--sandbox",
            sandbox,
            "--output-last-message",
            tmp_path,
            "--skip-git-repo-check" if skip_git_repo_check else "",
            "--",
        ]
        cmd = [part for part in cmd if part]
        if model:
            cmd[2:2] = ["--model", model]
        for image_path in normalized_images:
            cmd.extend(["--image", str(image_path)])
        proc = subprocess.run(
            cmd + [prompt],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            detail = normalize_text(proc.stderr or proc.stdout or f"rc={proc.returncode}")
            raise RuntimeError(f"codex exec failed: {detail}")
        if os.path.exists(tmp_path):
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as fh:
                text = normalize_text(fh.read())
            if text:
                return text
        return normalize_text(proc.stdout)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _gemini_cli_chat_messages(
    *,
    command: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    workdir: str,
    approval_mode: str,
    output_format: str,
    skip_trust: bool,
    image_paths: list[str] | None = None,
) -> str:
    if image_paths:
        raise RuntimeError("gemini_cli does not support image attachments in this integration")
    prompt = _messages_to_prompt(messages)
    gemini_bin = _resolve_cli_command(command)
    cmd = [
        gemini_bin,
        "--prompt",
        prompt,
        "--approval-mode",
        approval_mode,
        "--output-format",
        output_format,
    ]
    if skip_trust:
        cmd.append("--skip-trust")
    if model:
        cmd.extend(["--model", model])
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        detail = normalize_text(proc.stderr or proc.stdout or f"rc={proc.returncode}")
        raise RuntimeError(f"gemini cli failed: {detail}")
    body = (proc.stdout or "").strip()
    if output_format == "json":
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return normalize_text(body)
        if isinstance(data, dict):
            for key in ("text", "response", "content"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return normalize_text(value)
        return normalize_text(body)
    return normalize_text(body)


def llm_chat_messages(
    llm_cfg: dict[str, Any],
    messages: list[dict[str, str]],
    options: dict | None = None,
    image_paths: list[str] | None = None,
) -> str:
    provider = llm_cfg["provider"]
    if provider == "ollama":
        return _ollama_chat_messages(
            host=str(llm_cfg["host"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg.get("api_key", "")),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
            options=options,
            web_search=dict(llm_cfg.get("web_search", {})),
            think=llm_cfg.get("think"),
            image_paths=image_paths,
        )
    if provider == "gemini":
        return _gemini_chat_messages(
            api_base=str(llm_cfg["api_base"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg["api_key"]),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
            image_paths=image_paths,
        )
    if provider == "openai":
        return _openai_chat_messages(
            api_base=str(llm_cfg["api_base"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg["api_key"]),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
            image_paths=image_paths,
        )
    if provider == "anthropic":
        return _anthropic_chat_messages(
            api_base=str(llm_cfg["api_base"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg["api_key"]),
            anthropic_version=str(llm_cfg["anthropic_version"]),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
            image_paths=image_paths,
        )
    if provider == "codex":
        return _codex_chat_messages(
            command=str(llm_cfg["command"]),
            model=str(llm_cfg.get("model", "")),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
            sandbox=str(llm_cfg["sandbox"]),
            workdir=str(llm_cfg["workdir"]),
            skip_git_repo_check=bool(llm_cfg.get("skip_git_repo_check", True)),
            image_paths=image_paths,
        )
    if provider == "gemini_cli":
        return _gemini_cli_chat_messages(
            command=str(llm_cfg["command"]),
            model=str(llm_cfg.get("model", "")),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
            workdir=str(llm_cfg["workdir"]),
            approval_mode=str(llm_cfg["approval_mode"]),
            output_format=str(llm_cfg["output_format"]),
            skip_trust=bool(llm_cfg.get("skip_trust", True)),
            image_paths=image_paths,
        )
    raise RuntimeError(f"unsupported llm provider: {provider}")


def llm_chat(
    llm_cfg: dict[str, Any],
    system_prompt: str,
    user_text: str,
    options: dict | None = None,
    image_paths: list[str] | None = None,
) -> str:
    return llm_chat_messages(
        llm_cfg,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        options,
        image_paths,
    )


def llm_healthcheck(llm_cfg: dict[str, Any]) -> None:
    provider = llm_cfg["provider"]
    if provider == "ollama":
        r = requests.get(
            str(llm_cfg["host"]).rstrip("/") + "/api/version",
            timeout=2,
            headers=_ollama_headers(str(llm_cfg.get("api_key", ""))),
        )
        r.raise_for_status()
        return
    if provider == "gemini":
        if not str(llm_cfg.get("api_key", "")).strip():
            raise RuntimeError(f"gemini api key missing in env {llm_cfg.get('api_key_env', 'GEMINI_API_KEY')}")
        return
    if provider == "openai":
        if not str(llm_cfg.get("api_key", "")).strip():
            raise RuntimeError(f"openai api key missing in env {llm_cfg.get('api_key_env', 'OPENAI_API_KEY')}")
        return
    if provider == "anthropic":
        if not str(llm_cfg.get("api_key", "")).strip():
            raise RuntimeError(f"anthropic api key missing in env {llm_cfg.get('api_key_env', 'ANTHROPIC_API_KEY')}")
        return
    if provider == "codex":
        _resolve_cli_command(str(llm_cfg.get("command", "codex")))
        return
    if provider == "gemini_cli":
        _resolve_cli_command(str(llm_cfg.get("command", "gemini")))
        return
    raise RuntimeError(f"unsupported llm provider: {provider}")

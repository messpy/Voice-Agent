from __future__ import annotations

import os
from typing import Any

import requests


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


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
) -> str:
    messages = _augment_messages_with_ollama_web_search(
        {
            "provider": "ollama",
            "host": host,
            "api_key": api_key,
            "web_search": web_search or {},
        },
        messages,
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
) -> str:
    if not api_key:
        raise RuntimeError("gemini api key is missing")
    contents = []
    for msg in messages:
        role = "user" if msg.get("role") != "assistant" else "model"
        contents.append(
            {
                "role": role,
                "parts": [{"text": msg.get("content", "")}],
            }
        )
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
) -> str:
    if not api_key:
        raise RuntimeError("openai api key is missing")
    url = f"{api_base.rstrip('/')}/responses"
    payload = {
        "model": model,
        "input": [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages],
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
) -> str:
    if not api_key:
        raise RuntimeError("anthropic api key is missing")
    system_parts = [msg.get("content", "") for msg in messages if msg.get("role") == "system"]
    chat_messages = [
        {"role": "assistant" if msg.get("role") == "assistant" else "user", "content": msg.get("content", "")}
        for msg in messages
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


def llm_chat_messages(
    llm_cfg: dict[str, Any],
    messages: list[dict[str, str]],
    options: dict | None = None,
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
        )
    if provider == "gemini":
        return _gemini_chat_messages(
            api_base=str(llm_cfg["api_base"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg["api_key"]),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
        )
    if provider == "openai":
        return _openai_chat_messages(
            api_base=str(llm_cfg["api_base"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg["api_key"]),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
        )
    if provider == "anthropic":
        return _anthropic_chat_messages(
            api_base=str(llm_cfg["api_base"]),
            model=str(llm_cfg["model"]),
            api_key=str(llm_cfg["api_key"]),
            anthropic_version=str(llm_cfg["anthropic_version"]),
            messages=messages,
            timeout_sec=int(llm_cfg["timeout_sec"]),
        )
    raise RuntimeError(f"unsupported llm provider: {provider}")


def llm_chat(
    llm_cfg: dict[str, Any],
    system_prompt: str,
    user_text: str,
    options: dict | None = None,
) -> str:
    return llm_chat_messages(
        llm_cfg,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        options,
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
    raise RuntimeError(f"unsupported llm provider: {provider}")

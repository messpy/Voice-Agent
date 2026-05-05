from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.llm_api import llm_chat


DEFAULT_RAG_GLOBS = [
    "README.md",
    "config/**/*.yaml",
    "logs/**/*.txt",
]


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\u3000", " ").split()).strip()


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def char_ngrams(text: str, *, min_n: int = 2, max_n: int = 3) -> set[str]:
    compact = compact_text(text)
    grams: set[str] = set()
    for size in range(min_n, max_n + 1):
        if len(compact) < size:
            continue
        for idx in range(len(compact) - size + 1):
            grams.add(compact[idx : idx + size])
    return grams


def score_chunk(query: str, chunk: str) -> float:
    qgrams = char_ngrams(query)
    cgrams = char_ngrams(chunk)
    if not qgrams or not cgrams:
        return 0.0
    overlap = len(qgrams & cgrams)
    if overlap == 0:
        return 0.0
    coverage = overlap / max(1, len(qgrams))
    density = overlap / max(1, len(cgrams))
    return (coverage * 0.8) + (density * 0.2)


def load_rag_corpus(
    root: Path,
    patterns: list[str],
    chunk_size: int,
    chunk_overlap: int,
    max_files: int,
    max_file_chars: int,
) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    seen: set[Path] = set()
    step = max(1, chunk_size - chunk_overlap)
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if len(seen) > max_files:
                return chunks
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            body = normalize_text(text)[:max_file_chars]
            if not body:
                continue
            if len(body) <= chunk_size:
                chunks.append({"path": str(path), "text": body})
                continue
            for offset in range(0, len(body), step):
                chunk = normalize_text(body[offset : offset + chunk_size])
                if chunk:
                    chunks.append({"path": str(path), "text": chunk})
                if offset + chunk_size >= len(body):
                    break
    return chunks


def retrieve_rag_context(
    query: str,
    corpus: list[dict[str, str]],
    *,
    top_k: int,
    min_score: float,
) -> list[dict[str, str | float]]:
    body = normalize_text(query)
    if not body:
        return []
    rows: list[dict[str, str | float]] = []
    for item in corpus:
        score = score_chunk(body, item["text"])
        if score < min_score:
            continue
        rows.append(
            {
                "path": item["path"],
                "text": item["text"],
                "score": round(score, 4),
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows[:top_k]


def format_rag_context(results: list[dict[str, str | float]]) -> str:
    if not results:
        return ""
    lines = ["補正時の参考コンテキスト。固有名詞や用語の判断にだけ使い、元の意味を勝手に増やさないこと。"]
    for item in results:
        lines.append(f"[{item['path']}] {item['text']}")
    return "\n".join(lines)


def load_relevant_aliases(sqlite_path: Path, raw_text: str, *, top_k: int = 8, min_score: float = 0.18) -> list[dict[str, str | float]]:
    if not sqlite_path.exists():
        return []
    body = normalize_text(raw_text)
    if not body:
        return []
    rows: list[dict[str, str | float]] = []
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.execute(
            """
            select alias_type, target, alias, hits, enabled
            from recognition_aliases
            where enabled = 1
            order by hits desc, last_seen_ts desc
            """
        )
        for alias_type, target, alias, hits, enabled in cursor.fetchall():
            target_text = normalize_text(str(target))
            alias_text = normalize_text(str(alias))
            score = max(score_chunk(body, target_text), score_chunk(body, alias_text))
            if score < min_score:
                continue
            rows.append(
                {
                    "alias_type": normalize_text(str(alias_type)),
                    "target": target_text,
                    "alias": alias_text,
                    "hits": int(hits or 0),
                    "score": round(score, 4),
                }
            )
    rows.sort(key=lambda item: (float(item["score"]), int(item["hits"])), reverse=True)
    return rows[:top_k]


def format_alias_context(rows: list[dict[str, str | float]]) -> str:
    if not rows:
        return ""
    lines = ["補正時の alias 学習。次の言い換えや誤認識を優先候補として扱うこと。"]
    for item in rows:
        lines.append(f"[{item['alias_type']}] {item['alias']} -> {item['target']}")
    return "\n".join(lines)


def build_transcript_correction_context(
    *,
    root: Path,
    cfg: dict[str, Any],
    raw_text: str,
    sqlite_path: Path | None = None,
) -> str:
    assistant_cfg = cfg.get("assistant", {})
    rag_cfg = assistant_cfg.get("rag", {})
    rag_enabled = bool(rag_cfg.get("enabled", True))
    parts: list[str] = []
    if rag_enabled:
        patterns = rag_cfg.get("paths", DEFAULT_RAG_GLOBS)
        if not isinstance(patterns, list):
            patterns = DEFAULT_RAG_GLOBS
        corpus = load_rag_corpus(
            root,
            patterns,
            max(200, int(rag_cfg.get("chunk_size", 500))),
            max(0, int(rag_cfg.get("chunk_overlap", 80))),
            max(1, int(rag_cfg.get("max_files", 24))),
            max(500, int(rag_cfg.get("max_file_chars", 4000))),
        )
        rag_rows = retrieve_rag_context(
            raw_text,
            corpus,
            top_k=max(1, int(rag_cfg.get("top_k", 3))),
            min_score=float(rag_cfg.get("min_score", 0.12)),
        )
        rag_context = format_rag_context(rag_rows)
        if rag_context:
            parts.append(rag_context)
    if sqlite_path is not None:
        alias_rows = load_relevant_aliases(sqlite_path, raw_text)
        alias_context = format_alias_context(alias_rows)
        if alias_context:
            parts.append(alias_context)
    return "\n\n".join(part for part in parts if part)


def correct_transcript(
    *,
    llm_cfg: dict[str, Any],
    raw_text: str,
    rag_context: str = "",
) -> str:
    system_prompt = (
        "あなたは音声認識の補正器。"
        "誤変換、助詞の崩れ、不要な空白を自然な日本語に直す。"
        "話し言葉として不自然な箇所は、元の意味を保ったまま自然な口語に整える。"
        "短い相づちやくだけた言い回しは、勝手に硬くしすぎない。"
        "固有名詞や人名は確信がないならそのまま残す。"
        "意味を勝手に足さない。"
        "聞き取れない部分は無理に補わない。"
        "出力は補正後の本文だけにする。"
    )
    user_prompt = f"音声認識結果:\n{raw_text}\n\n補正後テキストだけを返して。"
    if rag_context:
        system_prompt = system_prompt + "\n\n" + rag_context
    corrected = llm_chat(llm_cfg, system_prompt, user_prompt)
    return corrected or raw_text

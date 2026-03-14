#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split()).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text)).lower()


def char_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> set[str]:
    body = compact_text(text)
    if not body:
        return set()
    grams: set[str] = set()
    for size in range(min_n, max_n + 1):
        if len(body) < size:
            continue
        for idx in range(len(body) - size + 1):
            grams.add(body[idx : idx + size])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="voicechat memory search")
    parser.add_argument("query", help="search text")
    parser.add_argument("--db", default="/tmp/voicechat/voicechat.db", help="sqlite db path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.18)
    parser.add_argument("--scan-limit", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"NG: db not found: {db_path}")

    rows: list[dict] = []
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            select id, date, event_type, mode, expected, recognized, payload_json
            from events
            order by ts desc
            limit ?
            """,
            (args.scan_limit,),
        )
        for row in cursor.fetchall():
            try:
                payload = json.loads(row[6] or "{}")
            except Exception:
                payload = {}
            memory_text = " ".join(
                normalize_text(str(value))
                for value in [
                    row[4] or "",
                    row[5] or "",
                    payload.get("raw_user", ""),
                    payload.get("corrected_user", ""),
                    payload.get("assistant", ""),
                ]
                if str(value).strip()
            )
            score = score_chunk(args.query, memory_text)
            if score < args.min_score:
                continue
            rows.append(
                {
                    "id": row[0],
                    "date": row[1],
                    "event_type": row[2],
                    "mode": row[3],
                    "expected": row[4] or "",
                    "recognized": row[5] or "",
                    "score": round(score, 4),
                }
            )

    rows.sort(key=lambda item: item["score"], reverse=True)
    print(json.dumps(rows[: args.top_k], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

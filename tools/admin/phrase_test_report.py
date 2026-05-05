import argparse
import json
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


@dataclass
class PhraseResult:
    rank_score: float
    date: str
    backend: str
    model: str
    expected: str
    recognized: str
    elapsed_sec: float
    similarity: float
    phrase_id: str
    phrase_note: str


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_paths(cfg: dict) -> tuple[Path, Path]:
    workdir = Path(str(cfg["paths"]["workdir"]))
    sqlite_path = workdir / str(cfg.get("storage", {}).get("sqlite", {}).get("path", "voicechat.db"))
    return workdir, sqlite_path


def fetch_phrase_results(
    sqlite_path: Path,
    *,
    expected_filter: str,
    phrase_id_filter: str,
    limit: int,
    accuracy_weight: float,
    speed_weight: float,
) -> list[PhraseResult]:
    if not sqlite_path.exists():
        return []

    query = """
        select
            date,
            backend,
            model,
            expected,
            recognized,
            elapsed_sec,
            payload_json
        from events
        where event_type = 'phrase_test'
        order by ts desc
    """
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(query).fetchall()

    raw_items: list[dict[str, Any]] = []
    for row in rows:
        payload = {}
        if row[6]:
            try:
                payload = json.loads(row[6])
            except json.JSONDecodeError:
                payload = {}
        expected = str(payload.get("expected") or row[3] or "")
        phrase_id = str(payload.get("phrase_id") or "")
        if expected_filter and expected != expected_filter:
            continue
        if phrase_id_filter and phrase_id != phrase_id_filter:
            continue
        raw_items.append(
            {
                "date": str(row[0] or ""),
                "backend": str(row[1] or ""),
                "model": str(row[2] or ""),
                "expected": expected,
                "recognized": str(payload.get("recognized") or row[4] or ""),
                "elapsed_sec": float(payload.get("elapsed_sec") or row[5] or 0.0),
                "similarity": float(payload.get("similarity") or 0.0),
                "phrase_id": phrase_id,
                "phrase_note": str(payload.get("phrase_note") or ""),
            }
        )

    if not raw_items:
        return []

    max_elapsed = max(item["elapsed_sec"] for item in raw_items) or 1.0
    min_elapsed = min(item["elapsed_sec"] for item in raw_items)
    elapsed_span = max(max_elapsed - min_elapsed, 1e-9)

    results: list[PhraseResult] = []
    for item in raw_items:
        speed_score = 1.0 - ((item["elapsed_sec"] - min_elapsed) / elapsed_span)
        rank_score = (item["similarity"] * accuracy_weight) + (speed_score * speed_weight)
        results.append(
            PhraseResult(
                rank_score=rank_score,
                date=item["date"],
                backend=item["backend"],
                model=item["model"],
                expected=item["expected"],
                recognized=item["recognized"],
                elapsed_sec=item["elapsed_sec"],
                similarity=item["similarity"],
                phrase_id=item["phrase_id"],
                phrase_note=item["phrase_note"],
            )
        )

    results.sort(key=lambda item: (-item.rank_score, -item.similarity, item.elapsed_sec, item.date), reverse=False)
    return results[:limit]


def format_model_name(model: str) -> str:
    name = Path(model).name
    return name or model or "(unknown)"


def build_summary_text(results: list[PhraseResult], expected_filter: str) -> str:
    subject = expected_filter or (results[0].expected if results else "phrase_test")
    lines = [f"順位発表です。対象は{subject}です。"]
    for idx, item in enumerate(results, start=1):
        lines.append(
            f"{idx}位、{format_model_name(item.model)}。"
            f"精度{item.similarity:.2f}、時間{item.elapsed_sec:.2f}秒です。"
        )
    return "".join(lines)


def print_results(results: list[PhraseResult]) -> None:
    if not results:
        print("NO DATA: matching phrase_test rows were not found.")
        return

    print("rank | score   | similarity | elapsed | backend    | model                | expected | recognized")
    for idx, item in enumerate(results, start=1):
        print(
            f"{idx:>4} | "
            f"{item.rank_score:>7.4f} | "
            f"{item.similarity:>10.4f} | "
            f"{item.elapsed_sec:>7.3f} | "
            f"{item.backend[:10]:<10} | "
            f"{format_model_name(item.model)[:20]:<20} | "
            f"{item.expected} | "
            f"{item.recognized}"
        )


def speak_voicevox(text: str, cfg: dict, workdir: Path) -> Path:
    tts_cfg = cfg.get("tts", {})
    audio_cfg = cfg.get("audio", {})
    host = str(tts_cfg["engine_host"]).rstrip("/")
    speaker = int(tts_cfg.get("speaker", 3))
    volume_scale = float(tts_cfg.get("voicevox_volume_scale", 1.0))
    audio_out = str(audio_cfg.get("output", "")).strip()

    query_resp = requests.post(
        host + "/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    query_resp.raise_for_status()
    query = query_resp.json()
    query["volumeScale"] = volume_scale

    synth_resp = requests.post(
        host + "/synthesis",
        params={"speaker": speaker},
        json=query,
        timeout=120,
    )
    synth_resp.raise_for_status()

    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="phrase_test_report_", suffix=".wav", dir=workdir, delete=False) as fp:
        out_wav = Path(fp.name)
        out_wav.write_bytes(synth_resp.content)

    cmd = ["aplay"]
    if audio_out:
        cmd += ["-D", audio_out]
    cmd.append(str(out_wav))
    subprocess.run(cmd, check=True)
    return out_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank phrase_test results by accuracy and time.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected", default="ベリベリ", help="Filter by expected phrase text.")
    parser.add_argument("--phrase-id", default="", help="Filter by phrase_test phrase_id.")
    parser.add_argument("--limit", type=int, default=5, help="How many ranked rows to show.")
    parser.add_argument("--accuracy-weight", type=float, default=0.8, help="Weight for similarity score.")
    parser.add_argument("--speed-weight", type=float, default=0.2, help="Weight for elapsed time score.")
    parser.add_argument("--speak", action="store_true", help="Announce the ranking with VOICEVOX.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args.config_path)
    workdir, sqlite_path = resolve_paths(cfg)

    if args.accuracy_weight < 0 or args.speed_weight < 0:
        raise SystemExit("NG: weights must be non-negative")
    if args.accuracy_weight == 0 and args.speed_weight == 0:
        raise SystemExit("NG: at least one weight must be positive")

    results = fetch_phrase_results(
        sqlite_path,
        expected_filter=args.expected,
        phrase_id_filter=args.phrase_id,
        limit=args.limit,
        accuracy_weight=args.accuracy_weight,
        speed_weight=args.speed_weight,
    )
    print_results(results)
    if not results:
        return 1

    if args.speak:
        summary = build_summary_text(results, args.expected)
        wav_path = speak_voicevox(summary, cfg, workdir)
        print(f"\nvoice: {wav_path}")
        print(f"text: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

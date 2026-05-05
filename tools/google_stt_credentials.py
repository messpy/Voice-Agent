from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_value_from_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def ensure_json(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Provided Google key is not valid JSON") from exc


def write_key(output_path: Path, data: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Google Speech-to-Text credentials")
    parser.add_argument("--output", default="~/.config/voicechat/google_stt.json", help="path to write the service account JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-value", help="JSON data provided inline")
    group.add_argument("--from-file", help="path to existing JSON file")
    group.add_argument("--from-env", help="environment variable that holds JSON data")
    parser.add_argument("--show-setter", action="store_true", help="print export line for GOOGLE_APPLICATION_CREDENTIALS")
    parser.add_argument("--check", action="store_true", help="verify GOOGLE_APPLICATION_CREDENTIALS is set and readable")
    args = parser.parse_args()

    if args.check:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred_path:
            raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS is not set")
        path_obj = Path(cred_path).expanduser()
        if not path_obj.exists():
            raise SystemExit(f"Credential file does not exist: {path_obj}")
        print(f"OK: credentials exist at {path_obj}")
        return

    raw_value: str | None = None
    if args.from_value:
        raw_value = args.from_value
    elif args.from_file:
        raw_value = load_value_from_file(args.from_file)
    elif args.from_env:
        raw_value = os.environ.get(args.from_env)
        if raw_value is None:
            raise SystemExit(f"Environment variable {args.from_env} is not set")

    if raw_value is None:
        raise SystemExit("Provide --from-value / --from-file / --from-env to write credentials")

    data = ensure_json(raw_value)
    out_path = Path(os.path.expanduser(args.output))
    write_key(out_path, data)
    print(f"Wrote credentials to {out_path}")
    if args.show_setter:
        print(f"export GOOGLE_APPLICATION_CREDENTIALS={out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy .env.example to .env (if not exists)")
    parser.add_argument("--force", action="store_true", help="overwrite .env even if it exists")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    example = repo_root / ".env.example"
    destination = repo_root / ".env"

    if not example.exists():
        raise SystemExit("No .env.example found")

    if destination.exists() and not args.force:
        print(".env already exists; use --force to overwrite")
        return

    shutil.copy(example, destination)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

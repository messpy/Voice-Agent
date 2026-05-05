#!/usr/bin/env python3
"""
STT モデル一括ダウンロードスクリプト

config/whisper_models.yaml に登録されている whisper.cpp / Moonshine モデルを
一括でダウンロードします。

使い方:
  # 全モデルをダウンロード
  uv run python tools/download_models.py

  # 特定のモデルのみ
  uv run python tools/download_models.py --ids kotoba_v2_q5_0 small

  # 保存先を変更
  uv run python tools/download_models.py --out ./whisper.cpp/models/

  # ダウンロード済みスキップ
  uv run python tools/download_models.py --skip-existing
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "whisper_models.yaml"


def download_file(url: str, dest: Path, *, retries: int = 3) -> None:
    """curl でファイルをダウンロード（リトライ対応）"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        print(f"  [{attempt}/{retries}] Downloading: {url}")
        print(f"       → {dest}")

        cmd = [
            "curl", "-L", "--fail", "--progress-bar",
            "-o", str(dest),
            url,
        ]
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✅ Done: {size_mb:.1f} MB")
            return
        else:
            print(f"  ⚠️  失敗、リトライします...")
            if dest.exists():
                dest.unlink()
            time.sleep(5)

    raise RuntimeError(f"ダウンロード失敗: {url}")


def load_catalog() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="STT モデルダウンロード")
    parser.add_argument("--ids", nargs="+", help="対象モデルID")
    parser.add_argument("--out", type=Path, help="保存先ディレクトリ")
    parser.add_argument("--skip-existing", action="store_true", help="既存ファイルをスキップ")
    args = parser.parse_args()

    catalog = load_catalog()
    models = catalog.get("models", [])

    if args.ids:
        wanted = set(args.ids)
        models = [m for m in models if m["id"] in wanted]
    if not models:
        raise SystemExit("NG: 対象モデルがありません")

    # デフォルト保存先
    default_out = ROOT / ".runtime" / "models"

    print("=" * 60)
    print("  STT モデル ダウンロード")
    print("=" * 60)

    for row in models:
        rid = row["id"]
        label = row.get("label", rid)
        family = row.get("family", "")
        url = row.get("download_url", "")

        if not url:
            print(f"  ⏭️  {rid}: ダウンロードURLがありません ({label})")
            continue

        if family == "moonshine_onnx":
            print(f"  ⏭️  {rid}: Moonshine は pip でインストールしてください ({label})")
            continue

        if family != "whisper.cpp":
            print(f"  ⏭️  {rid}: 未対応のファミリーです ({label})")
            continue

        # 保存先決定
        if args.out:
            dest = args.out / Path(url).name
        else:
            # candidate_paths の最初のパスを使用
            paths = row.get("candidate_paths", [])
            if paths:
                dest = Path(paths[0])
                if not dest.is_absolute():
                    dest = ROOT / dest
            else:
                dest = default_out / Path(url).name

        # スキップ判定
        if args.skip_existing and dest.exists() and dest.stat().st_size > 0:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ⏭️  {rid}: 既に存在します ({size_mb:.1f} MB)")
            continue

        print(f"\n📥 {rid}: {label}")
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  ❌ エラー: {e}")

    print("\n" + "=" * 60)
    print("  完了")
    print("=" * 60)

    # 既存モデル一覧表示
    out_dir = args.out or default_out
    if out_dir.exists():
        files = sorted(out_dir.glob("ggml-*.bin"))
        if files:
            print(f"\n📁 {out_dir} のモデル:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   {f.name:50s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conversation_core.config_loader import load_cfg
from conversation_core.llm_api import llm_chat, resolve_llm_config


DEFAULT_SYSTEM_PROMPT = (
    "あなたは画像認識アシスタント。"
    "見えている内容だけを述べる。"
    "不確かな点は不確かと明示する。"
    "求められていない推測は増やさない。"
    "出力は簡潔でよいが、必要なら箇条書きで整理してよい。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze one or more images with the configured LLM backend.")
    parser.add_argument("images", nargs="+", help="Image file paths")
    parser.add_argument(
        "-p",
        "--prompt",
        default="この画像に何が写っているか説明して。",
        help="User prompt for the image analysis",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt override",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path override",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_cfg(args.config)
    llm_cfg = resolve_llm_config(cfg)
    image_paths = [str(Path(item).expanduser()) for item in args.images]
    reply = llm_chat(
        llm_cfg,
        args.system_prompt,
        args.prompt,
        image_paths=image_paths,
    )
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

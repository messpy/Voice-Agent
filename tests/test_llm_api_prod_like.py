from __future__ import annotations

import unittest

import yaml

from localagent.llm_api import llm_chat, llm_healthcheck, resolve_llm_config
from tests.test_support import ProdLikeEnv


class LlmApiProdLikeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ProdLikeEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_codex_provider_supports_image_inputs(self) -> None:
        cfg = yaml.safe_load(self.env.config_path.read_text(encoding="utf-8"))
        llm_cfg = resolve_llm_config(cfg)

        reply = llm_chat(
            llm_cfg,
            "画像を見て説明して",
            "この画像の内容を一文で説明して",
            image_paths=[str(self.env.image_path)],
        )

        self.assertEqual(reply, "codex image answer")
        rows = self.env.read_jsonl(self.env.codex_log)
        self.assertTrue(rows)
        argv = rows[-1]["argv"]
        self.assertIn("--image", argv)
        self.assertIn(str(self.env.image_path), argv)
        self.assertIn("--output-last-message", argv)

    def test_gemini_cli_provider_returns_text_and_healthcheck(self) -> None:
        cfg = yaml.safe_load(self.env.config_path.read_text(encoding="utf-8"))
        cfg["llm"] = {
            "provider": "gemini_cli",
            "model": "gemini-2.5-flash",
            "timeout_sec": 30,
            "command": "gemini",
            "approval_mode": "plan",
            "output_format": "text",
            "skip_trust": True,
            "workdir": str(self.env.tmp_path),
        }
        llm_cfg = resolve_llm_config(cfg)

        llm_healthcheck(llm_cfg)
        reply = llm_chat(llm_cfg, "短く答える", "テスト応答を返して")

        self.assertEqual(reply, "gemini cli text answer")
        rows = self.env.read_jsonl(self.env.gemini_log)
        self.assertTrue(rows)
        argv = rows[-1]["argv"]
        self.assertIn("--prompt", argv)
        self.assertIn("--approval-mode", argv)
        self.assertIn("plan", argv)

    def test_gemini_cli_rejects_image_inputs(self) -> None:
        cfg = yaml.safe_load(self.env.config_path.read_text(encoding="utf-8"))
        cfg["llm"] = {
            "provider": "gemini_cli",
            "model": "gemini-2.5-flash",
            "timeout_sec": 30,
            "command": "gemini",
            "approval_mode": "plan",
            "output_format": "text",
            "skip_trust": True,
            "workdir": str(self.env.tmp_path),
        }
        llm_cfg = resolve_llm_config(cfg)

        with self.assertRaisesRegex(RuntimeError, "does not support image attachments"):
            llm_chat(
                llm_cfg,
                "画像を見て説明して",
                "この画像を説明して",
                image_paths=[str(self.env.image_path)],
            )

    def test_resolve_llm_config_defaults_to_local_ollama_when_missing(self) -> None:
        llm_cfg = resolve_llm_config({})
        self.assertEqual(llm_cfg["provider"], "ollama")
        self.assertEqual(llm_cfg["host"], "http://127.0.0.1:11434")

    def test_codex_healthcheck_fails_for_missing_command(self) -> None:
        cfg = {
            "llm": {
                "provider": "codex",
                "model": "gpt-5",
                "timeout_sec": 30,
                "command": "definitely-missing-codex-binary",
                "sandbox": "read-only",
                "skip_git_repo_check": True,
                "workdir": str(self.env.tmp_path),
            }
        }
        llm_cfg = resolve_llm_config(cfg)
        with self.assertRaisesRegex(RuntimeError, "cli command not found"):
            llm_healthcheck(llm_cfg)


if __name__ == "__main__":
    unittest.main()

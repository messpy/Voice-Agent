from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

from voiceai.llm_api import llm_chat, llm_healthcheck, resolve_llm_config
from tests.test_support import FakeLlmServer, ProdLikeEnv


class LlmApiHttpProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = FakeLlmServer()
        self.env = ProdLikeEnv()

    def tearDown(self) -> None:
        self.server.close()
        self.env.close()

    def test_ollama_provider_posts_images_as_base64(self) -> None:
        self.server.routes[("POST", "/api/chat")] = {
            "status": 200,
            "body": {"message": {"content": "ollama ok"}},
        }
        cfg = {
            "llm": {
                "provider": "ollama",
                "model": "test-model",
                "timeout_sec": 30,
                "host": self.server.base_url,
                "api_key_env": "OLLAMA_API_KEY",
            }
        }
        llm_cfg = resolve_llm_config(cfg)

        reply = llm_chat(
            llm_cfg,
            "画像を見て説明して",
            "この画像の内容を説明して",
            image_paths=[str(self.env.image_path)],
        )

        self.assertEqual(reply, "ollama ok")
        payload = self.server.requests[-1]["payload"]
        self.assertTrue(payload["messages"][-1]["images"])
        self.assertIsInstance(payload["messages"][-1]["images"][0], str)

    def test_openai_provider_sends_input_image(self) -> None:
        self.server.routes[("POST", "/responses")] = {
            "status": 200,
            "body": {"output_text": "openai ok"},
        }
        old_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            cfg = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "timeout_sec": 30,
                    "api_base": self.server.base_url,
                    "api_key_env": "OPENAI_API_KEY",
                }
            }
            llm_cfg = resolve_llm_config(cfg)
            llm_healthcheck(llm_cfg)

            reply = llm_chat(
                llm_cfg,
                "画像を見て説明して",
                "この画像の内容を説明して",
                image_paths=[str(self.env.image_path)],
            )
        finally:
            if old_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_api_key

        self.assertEqual(reply, "openai ok")
        payload = self.server.requests[-1]["payload"]
        content = payload["input"][-1]["content"]
        self.assertTrue(any(part["type"] == "input_image" for part in content))

    def test_gemini_provider_sends_inline_image_data(self) -> None:
        self.server.routes[("POST", "/models/gemini-2.5-flash:generateContent")] = {
            "status": 200,
            "body": {"candidates": [{"content": {"parts": [{"text": "gemini ok"}]}}]},
        }
        old_api_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-key"
        try:
            cfg = {
                "llm": {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "timeout_sec": 30,
                    "api_base": self.server.base_url,
                    "api_key_env": "GEMINI_API_KEY",
                }
            }
            llm_cfg = resolve_llm_config(cfg)
            llm_healthcheck(llm_cfg)
            reply = llm_chat(
                llm_cfg,
                "画像を見て説明して",
                "この画像の内容を説明して",
                image_paths=[str(self.env.image_path)],
            )
        finally:
            if old_api_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_api_key

        self.assertEqual(reply, "gemini ok")
        payload = self.server.requests[-1]["payload"]
        parts = payload["contents"][-1]["parts"]
        self.assertTrue(any("inline_data" in part for part in parts))

    def test_anthropic_provider_sends_base64_image_parts(self) -> None:
        self.server.routes[("POST", "/messages")] = {
            "status": 200,
            "body": {"content": [{"type": "text", "text": "anthropic ok"}]},
        }
        old_api_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            cfg = {
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "timeout_sec": 30,
                    "api_base": self.server.base_url,
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "anthropic_version": "2023-06-01",
                }
            }
            llm_cfg = resolve_llm_config(cfg)
            llm_healthcheck(llm_cfg)
            reply = llm_chat(
                llm_cfg,
                "画像を見て説明して",
                "この画像の内容を説明して",
                image_paths=[str(self.env.image_path)],
            )
        finally:
            if old_api_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = old_api_key

        self.assertEqual(reply, "anthropic ok")
        payload = self.server.requests[-1]["payload"]
        image_parts = payload["messages"][-1]["content"]
        self.assertTrue(any(part.get("type") == "image" for part in image_parts))

    def test_missing_image_path_fails_fast_before_provider_call(self) -> None:
        cfg = {
            "llm": {
                "provider": "ollama",
                "model": "test-model",
                "timeout_sec": 30,
                "host": self.server.base_url,
                "api_key_env": "OLLAMA_API_KEY",
            }
        }
        llm_cfg = resolve_llm_config(cfg)

        with self.assertRaisesRegex(RuntimeError, "image not found"):
            llm_chat(
                llm_cfg,
                "画像を見て説明して",
                "この画像の内容を説明して",
                image_paths=[str(self.env.tmp_path / "missing.png")],
            )
        self.assertEqual(self.server.requests, [])

    def test_vision_cli_uses_configured_provider(self) -> None:
        from tools.vision_analyze import main

        import yaml

        cfg = yaml.safe_load(self.env.config_path.read_text(encoding="utf-8"))
        cfg["llm"]["provider"] = "codex"
        self.env.write_config(cfg)

        argv = sys.argv
        sys.argv = [
            "vision_analyze.py",
            str(self.env.image_path),
            "--prompt",
            "画像を説明して",
        ]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main()
        finally:
            sys.argv = argv

        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _install_stubs() -> None:
    config_loader = types.ModuleType("src.config_loader")
    config_loader.load_cfg = lambda: {}
    sys.modules["src.config_loader"] = config_loader

    voicechat_audio = types.ModuleType("src.voicechat_audio")
    voicechat_audio.voicechat_speak = lambda *args, **kwargs: True
    voicechat_audio.pipo_sound = lambda *args, **kwargs: True
    voicechat_audio.popi_sound = lambda *args, **kwargs: True
    voicechat_audio.pi_sound = lambda *args, **kwargs: True
    sys.modules["src.voicechat_audio"] = voicechat_audio

    vosk_runner = types.ModuleType("src.vosk_runner")
    vosk_runner.vosk_once = lambda *args, **kwargs: (0, 0.0, "")
    sys.modules["src.vosk_runner"] = vosk_runner


_install_stubs()
MODULE = importlib.import_module("src.test_voicechat_loop")


class TestGetCurrentVolume(unittest.TestCase):
    @patch("src.test_voicechat_loop.subprocess.run")
    def test_parses_amixer_output_percentage(self, mock_run):
        mock_run.return_value = SimpleNamespace(stdout="Mono: Playback 74 [74%] [-20.50dB] [on]")
        self.assertEqual(MODULE.get_current_volume(), 74)

    @patch("src.test_voicechat_loop.subprocess.run")
    def test_falls_back_to_default_on_failure(self, mock_run):
        mock_run.side_effect = RuntimeError("amixer failed")
        self.assertEqual(MODULE.get_current_volume(), 50)


if __name__ == "__main__":
    unittest.main()

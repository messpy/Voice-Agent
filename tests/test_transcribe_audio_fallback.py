from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import wake_vad_record


class SpeechRecognitionFallbackTests(unittest.TestCase):
    def test_speech_recognition_falls_back_to_local_when_empty(self) -> None:
        with (
            patch.object(wake_vad_record, "transcribe_speech_recognition", return_value=("", 0.25, "speech_recognition:google")) as sr_mock,
            patch.object(wake_vad_record, "transcribe_local", return_value=("電気消して", 1.5, "/models/ggml-small.bin")) as local_mock,
        ):
            raw, corrected, elapsed, model = wake_vad_record.transcribe_audio(
                backend="speech_recognition",
                whisper_bin=Path("/bin/whisper"),
                whisper_model=Path("/models/ggml-small.bin"),
                vosk_model=None,
                wav=Path("/tmp/input.wav"),
                lang="ja",
                threads=4,
                beam=6,
                best=6,
                temp=0.0,
                workdir=Path("/tmp"),
                remote_cfg={},
                llm_cfg={},
                speech_recognition_cfg={"fallback_to_local": True},
            )

        self.assertEqual(raw, "電気消して")
        self.assertEqual(corrected, "電気消して")
        self.assertEqual(elapsed, 1.75)
        self.assertEqual(model, "speech_recognition:google+fallback:/models/ggml-small.bin:empty")
        sr_mock.assert_called_once()
        local_mock.assert_called_once()

    def test_speech_recognition_skips_fallback_when_verify_disabled(self) -> None:
        with (
            patch.object(wake_vad_record, "transcribe_speech_recognition", return_value=("エアコン消して", 0.3, "speech_recognition:google")),
            patch.object(wake_vad_record, "transcribe_local") as local_mock,
        ):
            raw, corrected, elapsed, model = wake_vad_record.transcribe_audio(
                backend="speech_recognition",
                whisper_bin=Path("/bin/whisper"),
                whisper_model=Path("/models/ggml-small.bin"),
                vosk_model=None,
                wav=Path("/tmp/input.wav"),
                lang="ja",
                threads=4,
                beam=6,
                best=6,
                temp=0.0,
                workdir=Path("/tmp"),
                remote_cfg={},
                llm_cfg={},
                speech_recognition_cfg={"fallback_to_local": True},
            )

        self.assertEqual(raw, "エアコン消して")
        self.assertEqual(corrected, "エアコン消して")
        self.assertEqual(elapsed, 0.3)
        self.assertEqual(model, "speech_recognition:google")
        local_mock.assert_not_called()

    def test_speech_recognition_uses_local_when_google_text_is_too_short(self) -> None:
        with (
            patch.object(wake_vad_record, "transcribe_speech_recognition", return_value=("電気", 0.3, "speech_recognition:google")),
            patch.object(wake_vad_record, "transcribe_local", return_value=("電気を消して", 2.0, "/models/ggml-large-v3-turbo-q5_0.bin")) as local_mock,
        ):
            raw, corrected, elapsed, model = wake_vad_record.transcribe_audio(
                backend="speech_recognition",
                whisper_bin=Path("/bin/whisper"),
                whisper_model=Path("/models/ggml-small.bin"),
                vosk_model=None,
                wav=Path("/tmp/input.wav"),
                lang="ja",
                threads=4,
                beam=6,
                best=6,
                temp=0.0,
                workdir=Path("/tmp"),
                remote_cfg={},
                llm_cfg={},
                speech_recognition_cfg={
                    "fallback_to_local": True,
                    "verify_with_local": True,
                    "verify_min_chars": 5,
                    "fallback_model": "/models/ggml-large-v3-turbo-q5_0.bin",
                },
            )

        self.assertEqual(raw, "電気を消して")
        self.assertEqual(corrected, "電気を消して")
        self.assertEqual(elapsed, 2.3)
        self.assertEqual(
            model,
            "speech_recognition:google+fallback:/models/ggml-large-v3-turbo-q5_0.bin:short",
        )
        local_mock.assert_called_once()

    def test_speech_recognition_keeps_google_when_local_is_not_better(self) -> None:
        with (
            patch.object(wake_vad_record, "transcribe_speech_recognition", return_value=("エアコン消して", 0.3, "speech_recognition:google")),
            patch.object(wake_vad_record, "transcribe_local", return_value=("エアコンして", 2.0, "/models/ggml-small.bin")),
        ):
            raw, corrected, elapsed, model = wake_vad_record.transcribe_audio(
                backend="speech_recognition",
                whisper_bin=Path("/bin/whisper"),
                whisper_model=Path("/models/ggml-small.bin"),
                vosk_model=None,
                wav=Path("/tmp/input.wav"),
                lang="ja",
                threads=4,
                beam=6,
                best=6,
                temp=0.0,
                workdir=Path("/tmp"),
                remote_cfg={},
                llm_cfg={},
                speech_recognition_cfg={
                    "fallback_to_local": True,
                    "verify_with_local": True,
                    "prefer_local_when_longer_chars": 4,
                },
            )

        self.assertEqual(raw, "エアコン消して")
        self.assertEqual(corrected, "エアコン消して")
        self.assertEqual(elapsed, 2.3)
        self.assertEqual(model, "speech_recognition:google+verified:/models/ggml-small.bin")


if __name__ == "__main__":
    unittest.main()

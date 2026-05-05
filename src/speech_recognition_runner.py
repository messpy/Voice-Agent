from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
import signal


def _load_speech_recognition():
    try:
        import speech_recognition as sr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "speech_recognition is required for the speech_recognition backend. "
            "Install it via `uv sync` or `pip install SpeechRecognition`."
        ) from exc

    return sr


def _boost_wav(wav: Path, boost_volume: float) -> Path:
    if boost_volume == 1.0:
        return wav

    boosted_wav = wav.parent / f"{wav.stem}_boosted{wav.suffix}"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav),
            "-af",
            f"volume={boost_volume}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(boosted_wav),
        ],
        capture_output=True,
    )
    if result.returncode == 0 and boosted_wav.exists():
        return boosted_wav
    return wav


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("transcript", "hypothesis", "text"):
            value = result.get(key) if hasattr(result, "get") else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        alternatives = result.get("alternative") if hasattr(result, "get") else None
        if isinstance(alternatives, list):
            for item in alternatives:
                if isinstance(item, dict):
                    transcript = str(item.get("transcript", "")).strip()
                    if transcript:
                        return transcript
    return ""


@contextmanager
def _time_limit(seconds: float):
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum, frame):  # pragma: no cover - signal handler
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def speech_recognition_once(
    *,
    wav: Path,
    engine: str = "google",
    language: str = "ja-JP",
    show_all: bool = False,
    boost_volume: float = 10.0,
    request_timeout_sec: float = 2.5,
) -> tuple[int, float, str]:
    sr = _load_speech_recognition()

    if not wav.exists() or wav.stat().st_size == 0:
        return 1, 0.0, f"NG: wav missing/empty: {wav}"

    wav_to_use = _boost_wav(wav, boost_volume)
    cleanup = wav_to_use if wav_to_use != wav else None
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False

    started_at = time.time()
    try:
        with sr.AudioFile(str(wav_to_use)) as source:
            audio = recognizer.record(source)

        engine_name = engine.strip().lower() or "google"
        if engine_name in {"google", "web", "webspeech"}:
            try:
                with _time_limit(request_timeout_sec):
                    result = recognizer.recognize_google(
                        audio, language=language, show_all=show_all
                    )
            except (TimeoutError, sr.RequestError):
                elapsed_sec = round(time.time() - started_at, 3)
                return 0, elapsed_sec, ""
        elif engine_name in {"sphinx", "pocketsphinx"}:
            result = recognizer.recognize_sphinx(
                audio, language=language, show_all=show_all
            )
        else:
            return 1, 0.0, f"NG: unsupported speech_recognition engine: {engine}"

        text = _extract_text(result) or (result.strip() if isinstance(result, str) else "")
        elapsed_sec = round(time.time() - started_at, 3)
        return 0, elapsed_sec, text
    except sr.UnknownValueError:
        elapsed_sec = round(time.time() - started_at, 3)
        return 0, elapsed_sec, ""
    except sr.RequestError as exc:
        elapsed_sec = round(time.time() - started_at, 3)
        return 0, elapsed_sec, ""
    except Exception as exc:
        elapsed_sec = round(time.time() - started_at, 3)
        return 1, elapsed_sec, f"NG: speech_recognition recognition failed: {exc}"
    finally:
        if cleanup is not None:
            try:
                cleanup.unlink()
            except OSError:
                pass

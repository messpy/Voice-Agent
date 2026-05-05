from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _load_google_client():
    try:
        from google.cloud import speech
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "google-cloud-speech is required for the Google Speech-to-Text backend. "
            "Install it via `pip install google-cloud-speech`."
        ) from exc

    return speech


def _resolve_audio_encoding(
    encoding_name: str, speech_module: Any
) -> Any:
    upper = encoding_name.strip().upper()
    return getattr(
        speech_module.RecognitionConfig.AudioEncoding,
        upper,
        speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
    )


def _build_speech_contexts(
    raw_contexts: list[object] | None, speech_module: Any
) -> list[Any]:
    if not raw_contexts:
        return []
    contexts: list[Any] = []
    for context in raw_contexts:
        if isinstance(context, dict):
            phrases = [str(item) for item in context.get("phrases", []) if str(item).strip()]
        elif isinstance(context, (list, tuple)):
            phrases = [str(item) for item in context if str(item).strip()]
        else:
            phrases = []
        if phrases:
            contexts.append(speech_module.SpeechContext(phrases=phrases))
    return contexts


def transcribe_google(
    wav: Path,
    cfg: dict[str, object] | None = None,
) -> tuple[str, float]:
    speech_module = _load_google_client()
    cfg = cfg or {}
    encoding = _resolve_audio_encoding(str(cfg.get("encoding", "LINEAR16")), speech_module)
    language_code = str(cfg.get("language_code", "ja-JP"))
    sample_rate = int(cfg.get("sample_rate_hertz", 16000))
    model = str(cfg.get("model", "latest_short"))
    use_enhanced = bool(cfg.get("use_enhanced", True))
    enable_punctuation = bool(cfg.get("enable_automatic_punctuation", True))
    enable_word_time_offsets = bool(cfg.get("enable_word_time_offsets", False))
    channel_count = int(cfg.get("audio_channel_count", 1))
    max_alternatives = max(1, int(cfg.get("max_alternatives", 1)))
    speech_contexts = _build_speech_contexts(cfg.get("speech_contexts"), speech_module)

    config = speech_module.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        audio_channel_count=channel_count,
        model=model,
        use_enhanced=use_enhanced,
        enable_automatic_punctuation=enable_punctuation,
        enable_word_time_offsets=enable_word_time_offsets,
        max_alternatives=max_alternatives,
        speech_contexts=speech_contexts,
    )

    audio = speech_module.RecognitionAudio(content=wav.read_bytes())
    client = speech_module.SpeechClient()
    started = time.perf_counter()
    response = client.recognize(config=config, audio=audio)
    elapsed = round(time.perf_counter() - started, 3)

    parts: list[str] = []
    for result in response.results:
        if result.alternatives:
            alt = result.alternatives[0]
            transcript = (alt.transcript or "").strip()
            if transcript:
                parts.append(transcript)
    text = " ".join(parts).strip()
    return text, elapsed

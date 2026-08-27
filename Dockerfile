FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/voicechat-venv \
    PATH=/opt/voicechat-venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        alsa-utils \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libasound2-dev \
        libffi-dev \
        libsndfile1 \
        pkg-config \
        procps \
        pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/voicechat

RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip setuptools wheel

COPY vendor/conversation_core ./vendor/conversation_core
COPY vendor/voiceaudio ./vendor/voiceaudio
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY config/config.example.yaml ./config/config.example.yaml
COPY assets ./assets

RUN pip install -e ./vendor/conversation_core \
    && pip install -e ./vendor/voiceaudio \
    && pip install \
        "numpy>=2.4.2" \
        "pyyaml>=6.0.2" \
        "requests>=2.32.5" \
        "setuptools<81" \
        "soundfile>=0.13.1" \
        "SpeechRecognition>=3.10.4" \
        "vosk>=0.3.45" \
        "webrtcvad>=2.0.10" \
        pvporcupine \
        "google-cloud-speech>=2.27.0" \
        yt-dlp

COPY voicechat.sh ./voicechat.sh

ENV VOICECHAT_RUNNER=python \
    VOICECHAT_PYTHON=/opt/voicechat-venv/bin/python \
    VOICECHAT_REQUIREMENTS_CHECK=1 \
    VOICECHAT_CONFIG=/app/voicechat/config/config.yaml \
    CONVERSATION_CORE_APP_ROOT=/app/voicechat \
    PYTHONPATH=/app/voicechat

CMD ["./voicechat.sh", "run"]

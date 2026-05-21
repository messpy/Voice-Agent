from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"

BENCHMARK_RESULTS_DIR = RUNTIME_DIR / "benchmark_results"
TEST_AUDIO_SUITE_DIR = RUNTIME_DIR / "test_audio_suite"
SAMPLES_DIR = RUNTIME_DIR / "samples"
ARCHIVES_DIR = RUNTIME_DIR / "archives"
VOSK_DIR = RUNTIME_DIR / "vosk"


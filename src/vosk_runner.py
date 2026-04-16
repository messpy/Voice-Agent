from __future__ import annotations
from pathlib import Path
import wave
import json
import time


def vosk_once(
    model_path: Path,
    wav: Path,
    log_path: Path | None = None,
    boost_volume: float = 1.0,
) -> tuple[int, float, str]:
    from vosk import Model, KaldiRecognizer
    import subprocess

    if not model_path.exists():
        return 1, 0.0, f"NG: vosk model missing: {model_path}"
    if not wav.exists() or wav.stat().st_size == 0:
        return 1, 0.0, f"NG: wav missing/empty: {wav}"

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply volume boost if needed
    wav_to_use = wav
    if boost_volume != 1.0:
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
        if result.returncode == 0:
            wav_to_use = boosted_wav

    t0 = time.time()
    try:
        model = Model(str(model_path))
    except Exception as e:
        return 1, 0.0, f"NG: vosk model load failed: {e}"

    try:
        rec = KaldiRecognizer(model, 16000)
        with wave.open(str(wav_to_use), "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
    except Exception as e:
        return 1, 0.0, f"NG: vosk recognition failed: {e}"

    dt = time.time() - t0

    if log_path:
        log_path.write_text(text + "\n", encoding="utf-8")

    return 0, dt, text

import os
import sys
import time
from pathlib import Path

from localagent.recorder import record_wav
from localagent.whisper_runner import whisper_cpp_transcribe

def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    raise SystemExit(code)

def get_env_path(name: str) -> str:
    v = os.environ.get(name, "").strip()
    return v

def main():
    whisper_bin = get_env_path("WHISPER_BIN") or str(Path.cwd() / "whisper.cpp/build/bin/whisper-cli")
    whisper_model = get_env_path("WHISPER_MODEL") or str(Path.cwd() / "whisper.cpp/models/ggml-base.bin")
    audio_in = get_env_path("AUDIO_IN") or "default"
    rec_sec = int(get_env_path("REC_SEC") or "10")
    lang = get_env_path("LANG_CODE") or "ja"

    wb = Path(whisper_bin)
    wm = Path(whisper_model)
    if not wb.exists():
        die(f"NG: WHISPER_BIN not found: {wb}")
    if not wm.exists():
        die(f"NG: WHISPER_MODEL not found: {wm}")

    out_dir = Path("/tmp/voicechat_quick")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"rec_{rec_sec}s.wav"
    outbase = out_dir / "out"

    print(f"INFO: 録音 {rec_sec}s（{audio_in}）")
    for i in (3,2,1):
        print(f"INFO: {i}...")
        time.sleep(1)

    record_wav(wav, audio_in, rec_sec, 16000, 1, "S16_LE", 0)
    text, _elapsed = whisper_cpp_transcribe(
        whisper_bin=wb,
        model_path=wm,
        wav=wav,
        out_prefix=outbase,
        lang=lang,
        threads=1,
        beam=5,
        temperature=0.0,
        extra_args=["-nt"],
    )
    print("===== TRANSCRIPT =====")
    if text:
        print(text)
    else:
        print("(no transcript)")
    print("======================")
    print(f"OK: WAV={wav}")
    print(f"OK: TXT={outbase}.txt")

if __name__ == "__main__":
    main()

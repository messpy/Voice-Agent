import requests
from pathlib import Path
from typing import Optional

DEFAULT_AUDIO_OUT = "plughw:CARD=B01,DEV=0"


def speak_voicevox(
    text: str,
    audio_out: Path,
    host: str = "http://127.0.0.1:50021",
    speaker: int = 3,
    speed: float = 1.0,
    pitch: float = 0.0,
    intonation: float = 1.0,
    volume: float = 1.0,
) -> bool:
    try:
        query = requests.post(
            f"{host}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=30,
        ).json()
        query["speedScale"] = speed
        query["pitchScale"] = pitch
        query["intonationScale"] = intonation
        query["volumeScale"] = volume

        audio = requests.post(
            f"{host}/synthesis",
            params={"speaker": speaker},
            json=query,
            timeout=120,
        ).content

        audio_out.write_bytes(audio)
        return True
    except Exception as e:
        print(f"TTS error: {e}")
        return False


def play_beep(
    name: str,
    audio_out: Optional[str] = None,
    workdir: Optional[Path] = None,
    volume: float = 0.5,
) -> bool:
    import subprocess

    effect_file = Path(__file__).parent.parent / "assets" / "effects" / f"{name}.wav"

    if not effect_file.exists():
        effect_file = Path("assets/effects") / f"{name}.wav"

    if effect_file.exists():
        cmd = ["aplay"]
        if audio_out:
            cmd += ["-D", audio_out]
        else:
            cmd += ["-D", DEFAULT_AUDIO_OUT]
        cmd += [str(effect_file)]
        subprocess.run(cmd, capture_output=True)
        return True

    return False


def pipo_sound(audio_out: Optional[str] = None) -> bool:
    return play_beep("ready_pipo", audio_out)


def popi_sound(audio_out: Optional[str] = None) -> bool:
    return play_beep("recorded_popi", audio_out)


def voicechat_speak(
    text: str,
    speaker: int = 3,
    host: str = "http://127.0.0.1:50021",
    audio_out_device: Optional[str] = None,
    workdir: Path = Path("/tmp"),
) -> bool:
    wav_path = workdir / "voicechat_tts.wav"

    if not speak_voicevox(text, wav_path, host, speaker):
        return False

    import subprocess

    cmd = ["aplay"]
    if audio_out_device:
        cmd += ["-D", audio_out_device]
    else:
        cmd += ["-D", DEFAULT_AUDIO_OUT]
    cmd += [str(wav_path)]
    subprocess.run(cmd, capture_output=True)
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python voicechat_audio.py <text> [speaker] [audio_out_device]")
        print("  pipo  - play start beep")
        print("  popi  - play end beep")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "pipo":
        pipo_sound()
    elif cmd == "popi":
        popi_sound()
    else:
        text = cmd
        speaker = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        audio_out = sys.argv[3] if len(sys.argv) > 3 else None
        voicechat_speak(text, speaker=speaker, audio_out_device=audio_out)

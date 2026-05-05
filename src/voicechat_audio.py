import argparse
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


def pi_sound():
    """ピ音のみ（wake検出時の開始音）"""
    import subprocess

    # 高い音
    cmd = [
        "aplay",
        "-D",
        DEFAULT_AUDIO_OUT,
        str(Path(__file__).parent.parent / "assets/effects/ready_pipo.wav"),
    ]
    subprocess.run(cmd, capture_output=True)


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
    parser = argparse.ArgumentParser(description="VOICEVOX 再生ユーティリティ")
    parser.add_argument("text", nargs="?", help="喋らせる文。pipo / popi も指定できる")
    parser.add_argument("speaker", nargs="?", type=int, default=3, help="話者ID")
    parser.add_argument("audio_out_device", nargs="?", default=None, help="再生デバイス")
    parser.add_argument("--text", dest="text_opt", default=None, help="喋らせる文")
    parser.add_argument("--speaker", dest="speaker_opt", type=int, help="話者ID")
    parser.add_argument("--audio-out", dest="audio_out_opt", default=None, help="再生デバイス")
    parser.add_argument("--host", default="http://127.0.0.1:50021", help="VOICEVOX エンジン URL")
    args = parser.parse_args()

    text = args.text_opt if args.text_opt is not None else args.text
    if not text:
        print("Usage: python voicechat_audio.py <text|pipo|popi> [speaker] [audio_out_device]")
        print("       python voicechat_audio.py --text <text> [--speaker 3] [--audio-out plughw:...]")
        raise SystemExit(1)

    cmd = text
    speaker = args.speaker_opt if args.speaker_opt is not None else args.speaker
    audio_out = args.audio_out_opt if args.audio_out_opt is not None else args.audio_out_device

    if cmd == "pipo":
        pipo_sound(audio_out)
    elif cmd == "popi":
        popi_sound(audio_out)
    else:
        voicechat_speak(cmd, speaker=speaker, host=args.host, audio_out_device=audio_out)

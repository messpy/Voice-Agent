import subprocess
import sys
import wave
import time
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voicechat_audio import voicechat_speak, pipo_sound, popi_sound
from src.vosk_runner import vosk_once


def record_audio(duration_sec: int = 10) -> Path:
    wav_path = Path("/tmp/voicechat_test_recording.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "alsa",
            "-i",
            "plughw:CARD=Device,DEV=0",
            "-t",
            str(duration_sec),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(wav_path),
        ],
        capture_output=True,
    )
    return wav_path


def main():
    # ずんだもんが「ベリベリと読んで」と言う
    print("=== ずんだもん: ベリベリと読んで ===")
    voicechat_speak("ベリベリと読んで", speaker=3)

    # 少し待ってからピポ音（録音開始）
    print("=== ピポ（録音開始）===")
    time.sleep(0.5)  # 0.5秒待つ
    pipo_sound()

    # ユーザー音声入力待機（5秒間録音）
    print("=== 録音中...（5秒）===")
    wav_path = Path("/tmp/voicechat_test_recording.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "alsa",
            "-i",
            "plughw:CARD=Device,DEV=0",
            "-t",
            "5",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(wav_path),
        ],
        capture_output=True,
    )

    # ポピ音（録音終了）
    print("=== ポピ（録音終了）===")
    popi_sound()

    # 音声認識
    print("=== 音声認識中 ===")
    wav = Path("/tmp/voicechat_test_recording.wav")
    if not wav.exists() or wav.stat().st_size < 1000:
        print("録音がありません")
        return

    # VOSKで文字起こし
    vosk_model = ROOT / ".runtime/vosk/vosk-model-ja-0.22"

    rc, elapsed, text = vosk_once(vosk_model, wav, boost_volume=10.0)
    print(f"認識結果: {text}")

    if text.strip():
        # ずんだもんが認識結果を読み上げる
        print(f"=== ずんだもん: 「{text}」===")
        voicechat_speak(text, speaker=3)
    else:
        print("=== ずんだもん: 聞こえなかったよ ===")
        voicechat_speak("聞こえなかったよ", speaker=3)


if __name__ == "__main__":
    main()

import subprocess
import sys
import wave
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_cfg
from src.voicechat_audio import voicechat_speak, pipo_sound, popi_sound, pi_sound
from src.vosk_runner import vosk_once

LOG_FILE = Path("/tmp/voicechat_log.txt")
WORD_MAP_FILE = Path("/tmp/voicechat_word_map.json")


EXCLUDED_KEY = "_excluded"


def load_word_map() -> dict:
    """Load word map. Structure: {target_word: [recognized_patterns], "_excluded": [excluded_patterns]}"""
    if WORD_MAP_FILE.exists():
        data = json.loads(WORD_MAP_FILE.read_text(encoding="utf-8"))
        if EXCLUDED_KEY not in data:
            data[EXCLUDED_KEY] = []
        return data
    return {EXCLUDED_KEY: []}


def save_word_map(word_map: dict):
    if EXCLUDED_KEY not in word_map:
        word_map[EXCLUDED_KEY] = []
    WORD_MAP_FILE.write_text(
        json.dumps(word_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def normalize_phrase_test_entry(item: object, idx: int) -> dict[str, str]:
    if isinstance(item, dict):
        text = str(item.get("text", "")).strip()
        if not text:
            return {}
        entry_id = str(item.get("id", "")).strip() or f"phrase_{idx:02d}"
        note = str(item.get("note", "")).strip()
        return {"id": entry_id, "text": text, "note": note}
    text = str(item).strip()
    if not text:
        return {}
    return {"id": f"phrase_{idx:02d}", "text": text, "note": ""}


def load_phrase_test_items() -> list[dict[str, str]]:
    cfg = load_cfg()
    phrase_test_cfg = (((cfg.get("assistant") or {}).get("modes") or {}).get("phrase_test") or {})
    phrases = phrase_test_cfg.get("phrases")
    if isinstance(phrases, list) and phrases:
        items = [
            normalize_phrase_test_entry(item, idx)
            for idx, item in enumerate(phrases, start=1)
        ]
        return [item for item in items if item]

    phrases_file = str(phrase_test_cfg.get("phrases_file", "")).strip()
    if not phrases_file:
        return []

    path = Path(phrases_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise RuntimeError(f"phrase_test phrases_file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("phrases", [])
        if not isinstance(data, list):
            raise RuntimeError(f"phrase_test json must be a list or {{\"phrases\": [...]}}: {path}")
        items = [
            normalize_phrase_test_entry(item, idx)
            for idx, item in enumerate(data, start=1)
        ]
        return [item for item in items if item]

    items: list[dict[str, str]] = []
    line_no = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        line_no += 1
        items.append({"id": f"phrase_{line_no:02d}", "text": text, "note": ""})
    return items


def log_result(user_text: str, corrected: str = ""):
    """Log the conversation result."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ユーザー: {user_text}\n")
        if corrected and corrected != user_text:
            f.write(f"[{timestamp}] 補正: → {corrected}\n")
        f.write("---\n")


def correct_text(text: str, word_map: dict) -> tuple[str, bool]:
    """Correct text using word map. Returns (corrected_text, was_corrected)"""
    excluded = word_map.get(EXCLUDED_KEY, [])
    for target, patterns in word_map.items():
        if target == EXCLUDED_KEY:
            continue
        for pattern in patterns:
            # 除外リストにあるパターンは無視
            if pattern in excluded:
                continue
            if pattern in text or text in pattern:
                return target, True
    return text, False


def register_word(target_word: str, recognized_text: str, word_map: dict):
    """Register a new word mapping."""
    if target_word == EXCLUDED_KEY:
        return
    if target_word not in word_map:
        word_map[target_word] = []
    if recognized_text not in word_map[target_word]:
        word_map[target_word].append(recognized_text)
    save_word_map(word_map)
    print(f"登録: 「{target_word}」← 「{recognized_text}」")


def check_and_fix_conflicts(word_map: dict) -> dict:
    """Check for conflicting patterns (same pattern in 2+ targets).

    If a pattern appears in 2+ target lists → move it to excluded list.
    """
    # pattern -> list of targets
    pattern_to_targets: dict[str, list[str]] = {}
    for target, patterns in word_map.items():
        if target == EXCLUDED_KEY:
            continue
        for pattern in patterns:
            if pattern not in pattern_to_targets:
                pattern_to_targets[pattern] = []
            pattern_to_targets[pattern].append(target)

    # Find conflicts: same pattern in 2+ different targets
    conflicts = {
        p: targets for p, targets in pattern_to_targets.items() if len(targets) >= 2
    }

    if conflicts:
        print(f"\n⚠️ コンフリクト検出:")
        for pattern, targets in conflicts.items():
            targets_str = "と".join(targets)
            msg = f"「{pattern}」は「{targets_str}」と重複ているので無効化しました"
            print(f"  {msg}")
            voicechat_speak(msg, speaker=3)

            # 除外リストに追加
            if EXCLUDED_KEY not in word_map:
                word_map[EXCLUDED_KEY] = []
            if pattern not in word_map[EXCLUDED_KEY]:
                word_map[EXCLUDED_KEY].append(pattern)

            # 各targetから削除
            for target in targets:
                if target in word_map and pattern in word_map[target]:
                    word_map[target].remove(pattern)
        save_word_map(word_map)

    return word_map


def voicechat_loop(target_word: str, word_map: dict) -> str:
    """Main voicechat loop. Returns the recognized/corrected text."""
    vosk_model = ROOT / ".runtime/vosk/vosk-model-ja-0.22"

    # ずんだもん「〇〇と呼んで」
    prompt = f"{target_word}と呼んで"
    print(f"=== ずんだもん: {prompt} ===")
    voicechat_speak(prompt, speaker=3)

    # ピポ → 録音 → ポピ
    print("=== ピポ（録音開始）===")
    time.sleep(0.5)
    pipo_sound()

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

    print("=== ポピ（録音終了）===")
    popi_sound()

    # VOSKで文字起こし
    print("=== 音声認識中 ===")
    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        print("録音がありません")
        return ""

    rc, elapsed, text = vosk_once(vosk_model, wav_path, boost_volume=10.0)
    print(f"VOSK認識: {text}")

    # 補正チェック
    corrected_text, was_corrected = correct_text(text, word_map)
    if was_corrected:
        print(f"補正: 「{text}」→「{corrected_text}」")
        final_text = corrected_text
        # 補正後のマップを確認・修復
        word_map = check_and_fix_conflicts(word_map)
    else:
        # 自動登録（確認なし）
        register_word(target_word, text, word_map)
        final_text = target_word
        # 登録後にコンフリクトチェック
        word_map = check_and_fix_conflicts(word_map)

    # ログ
    log_result(text, final_text)

    return final_text


def wake_mode():
    """ウェイクモード: ベリベリかアレクサを検出したらコマンドモードへ"""
    vosk_model = ROOT / ".runtime/vosk/vosk-model-ja-0.22"
    word_map = load_word_map()

    print("=== ウェイクモード開始 ===")

    while True:
        print("=== ウェイク待機中... ===")

        # 録音開始（音なし）
        wav_path = Path("/tmp/voicechat_wake.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "alsa",
                "-i",
                "plughw:CARD=Device,DEV=0",
                "-t",
                "3",
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

        # VOSKで文字起こし
        print("=== 認識中 ===")
        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            continue

        rc, elapsed, text = vosk_once(vosk_model, wav_path, boost_volume=10.0)
        print(f"認識: {text}")

        # ウェイク単語チェック
        corrected, was_corrected = correct_text(text, word_map)
        print(f"補正: {corrected}")

        # ベリベリかアレクサを検出
        if corrected in ["ベリベリ", "アレクサ"]:
            print(f"=== ウェイク検出: {corrected} ===")
            # ピポ（開始音）
            pipo_sound()
            voicechat_speak("どうぞ", speaker=3)
            command_mode(word_map)
        else:
            # ポピ（終了音・タイムアウト）
            popi_sound()


def command_mode(word_map: dict):
    """コマンドモード: ユーザーがコマンドを話す"""
    vosk_model = ROOT / ".runtime/vosk/vosk-model-ja-0.22"

    print("=== コマンドモード開始 ===")

    # ピポ → 録音 → ポピ
    print("=== ピポ（録音開始）===")
    time.sleep(0.5)
    pipo_sound()

    print("=== 録音中...（10秒）===")
    wav_path = Path("/tmp/voicechat_command.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "alsa",
            "-i",
            "plughw:CARD=Device,DEV=0",
            "-t",
            "10",
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

    print("=== ポピ（録音終了）===")
    popi_sound()

    # VOSKで文字起こし
    print("=== コマンド認識中 ===")
    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        print("録音がありません")
        return

    rc, elapsed, text = vosk_once(vosk_model, wav_path, boost_volume=10.0)
    print(f"コマンド: {text}")

    # 音量コマンド処理
    response = process_volume_command(text)
    if response:
        print(f"=== ずんだもん: 「{response}」===")
        voicechat_speak(response, speaker=3)
    else:
        # ずんだもんがコマンドを読み上げ
        print(f"=== ずんだもん: 「{text}」===")
        voicechat_speak(text, speaker=3)

    print("=== コマンドモード終了 ===")


def process_volume_command(text: str) -> str:
    """音量コマンドを処理"""
    import re

    # 音量をXXに設定
    match = re.search(r"音量.?(\d+)", text)
    if match:
        volume = int(match.group(1))
        set_system_volume(volume)
        return f"音量を{volume}にしました"

    # 音を上げる
    if any(x in text for x in ["上げて", "上げ", "アゲ"]):
        current = get_current_volume()
        new_volume = min(100, current + 10)
        set_system_volume(new_volume)
        return f"音量を{new_volume}にしました"

    # 音を下げる
    if any(x in text for x in ["下げて", "下げ", "サゲ"]):
        current = get_current_volume()
        new_volume = max(0, current - 10)
        set_system_volume(new_volume)
        return f"音量を{new_volume}にしました"

    return ""


def get_current_volume() -> int:
    """現在の音量を取得"""
    try:
        result = subprocess.run(
            ["amixer", "sget", "Master"], capture_output=True, text=True
        )
        # Parse percentage
        match = re.search(r"(\d+)%", result.stdout)
        if match:
            return int(match.group(1))
    except:
        pass
    return 50


def set_system_volume(volume: int):
    """音量を設定"""
    volume = max(0, min(100, volume))
    subprocess.run(["amixer", "sset", "Master", f"{volume}%"], capture_output=True)
    print(f"音量設定: {volume}%")


def main():
    import sys

    word_map = load_word_map()
    print(f"単語マップ: {len(word_map)} entries")
    print(f"現在のマップ: {word_map}")

    # コマンドライン引数で指定
    if len(sys.argv) > 1:
        if sys.argv[1] == "--wake":
            # ウェイクモード開始
            wake_mode()
        elif sys.argv[1] == "--all":
            # 全言葉をループ
            words_to_test = [item["text"] for item in load_phrase_test_items()]
            if not words_to_test:
                raise RuntimeError("phrase_test items are empty: config/phrase_test_words.json")
        else:
            # 特定の言葉だけ
            words_to_test = sys.argv[1:]
    else:
        print("\n使用法:")
        print("  uv run python src/test_voicechat_loop.py <言葉>      # 学習")
        print("  uv run python src/test_voicechat_loop.py --all       # 全単語学習")
        print("  uv run python src/test_voicechat_loop.py --wake      # ウェイクモード")
        return

    print(f"\nテスト対象: {words_to_test}")
    print()

    for i, target_word in enumerate(words_to_test):
        print(f"\n{'=' * 50}")
        print(f"[{i + 1}/{len(words_to_test)}] {target_word}")
        print(f"{'=' * 50}")
        voicechat_loop(target_word, word_map)
        word_map = load_word_map()  # リロード
        print()

    print("最終マップ:")
    print(load_word_map())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
5分テスト用音声生成 — VOICEVOXで長文を合成

複数のセグメントに分割して合成し、最後に結合します。
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests

VOICEVOX_URL = "http://127.0.0.1:50021"
SPEAKER_ID = 3  # ずんだもん
OUT_DIR = Path("/home/kennypi/work/voicechat/test_audio_suite")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 5分用の長文テストテキスト
# （約3500文字、VOICEVOXで約5分相当）
LONG_TEXT = """
音声認識技術は、近年急速に発展している人工知能の重要な分野の一つです。
この技術の歴史を振り返ると、初期の音声認識は隠れマルコフモデルとガウス混合モデルを組み合わせた手法が主流でした。
しかし、これらの従来手法には、認識精度の限界や、環境ノイズに対する脆弱性といった課題がありました。

転機となったのは、ディープラーニング技術の登場です。
二〇一〇年以降、畳み込みニューラルネットワークやリカレントニューラルネットワークを音声認識に応用する研究が活発化しました。
特に、long short-term memory、略してLSTMと呼ばれるリカレントニューラルネットワークの一種は、
時系列データの処理に優れており、音声認識の精度を大幅に向上させました。

そして、二〇二二年にOpenAIが発表したWhisperは、音声認識の歴史において画期的なモデルとなりました。
Whisperは、六十八万時間もの多言語の音声データを用いて事前学習されたTransformerモデルで、
多言語対応、話者非依存、そして高い認識精度という三つの特徴を同時に実現しました。

日本語の音声認識には、英語とは異なる特有の難しさがあります。
まず、同音異義語が非常に多いことです。
例えば、コウショウという一つの発音に対して、交渉、高尚、考証、公称、口証、
工場、公証、興信など、多数の漢字表記が存在します。
人間は文脈から適切な漢字を選択できますが、音声認識モデルにとってこれは困難な課題です。

次に、日本語はモーラという音の単位で構成されており、
長音や撥音、促音などの特殊音節が意味の区別に重要な役割を果たします。
例えば、おばさんとおばあさん、にじんとにっじょうなど、
一音の違いで全く異なる意味になることがあります。

また、日本語の話者は、会話の中で頻繁に省略表現を使います。
「よくわかんない」「そっちでいいよ」「まぁ、そんな感じで」といった表現は、
書き言葉としては不正確ですが、日常会話ではごく自然に用いられます。
音声認識システムがこれらの表現を正確に認識できるかどうかは、
実用性を左右する重要な要素です。

Kotoba-techが開発したKotoba-Whisperは、
OpenAIのWhisperモデルを日本語の大規模コーパスでファインチューニングしたモデルです。
このモデルは、固有名词、専門用語、そして口語表現の認識精度が大幅に向上しており、
ビジネス会議の議事録作成、医療現場での診察記録、
そしてカスタマーセンターの通話分析など、様々な分野での活用が期待されています。

音声認識技術の応用例として、まずスマートスピーカーが挙げられます。
Amazon Echo、Google Home、Apple HomePodなどの製品は、
音声による家電制御、音楽再生、天気予報の提供など、様々な機能を提供しています。

次に、自動文字起こしサービスです。
会議やインタビュー、講義などの音声をリアルタイムでテキストに変換することで、
議事録の作成やアクセシビリティの向上に貢献しています。
特に、リモートワークが普及した現代において、
オンライン会議の自動文字起こしは多くの企業で活用されています。

さらに、コールセンターでの活用も進んでいます。
顧客との通話をリアルタイムで文字起こしし、
オペレーターに適切な回答をサジェストするシステムや、
通話内容を自動で分類・分析するシステムが導入され始めています。

医療分野でも音声認識の活用が進んでいます。
医師が診察中に患者の症状や治療法を音声で入力し、
電子カルテに自動で記録するシステムは、
医師の負担軽減と診療効率の向上に大きく貢献しています。

教育分野では、外国語学習における発音練習や、
聴覚障害者向けのリアルタイム字幕生成などに音声認識技術が活用されています。
特に、教室での授業をリアルタイムで文字化するシステムは、
インクルーシブ教育の実現に不可欠なツールとなっています。

音声認識技術の課題についても触れておきましょう。
第一に、マルチリンガル対応の難しさです。
複数の言語が混在する環境、いわゆるコードスイッチングへの対応は、
まだ発展途上の段階にあります。

第二に、感情や話者特性の認識です。
現在の音声認識システムは、話されている内容をテキストに変換することはできますが、
話者の感情、性別、年齢、そして個人を特定する情報は、
別のモデルで処理する必要があります。

第三に、プライバシーとセキュリティの問題です。
音声データは個人の生体情報を含む機微なデータであり、
その取り扱いには細心の注意が必要です。
また、常時音声入力を前提としたシステムでは、
意図しない録音やデータ漏洩のリスクも考慮する必要があります。

今後の音声認識技術の展望として、
大規模言語モデルとの統合が挙げられます。
音声認識の結果を直接言語モデルに入力し、
文脈を考慮したより正確なテキスト生成や、
対話型AIアシスタントとの自然なやり取りを実現する研究が進んでいます。

また、エッジデバイスでのリアルタイム処理の最適化も重要な課題です。
スマートグラスやイヤホン型のウェアラブルデバイスなど、
小型デバイス上で高精度な音声認識を実現するためには、
モデルの軽量化と低消費電力化が不可欠です。

さらに、ゼロショット学習や少数ショット学習による
話者適応やドメイン適応の研究も活発化しています。
これは、新しい話者や新しい分野に適応するために、
大量の学習データを必要とせず、
数分の音声や数テキストのサンプルで
モデルを適応させる技術です。

音声認識技術は、単なる音声をテキストに変換するツールから、
人間のコミュニケーションを理解し、支援する総合的なAIシステムへと進化しつつあります。
この進化は、私たちの日常生活や仕事の仕方を変え、
より自然で直感的な人間と機械のインタラクションを実現するでしょう。

技術の進歩とともに、音声認識の精度は日々向上しています。
しかし、完全な認識を実現するには、まだ多くの課題が残されています。
ノイズ環境下での認識精度向上、方言やアクセントへの対応、
子供や高齢者の音声への対応、
そして何より、話者の意図を正く理解すること。

これらの課題を一つずつ解決していくことで、
音声認識技術はさらに発展し、
私たちの生活をより豊かで便利なものにしていくことでしょう。

このテストは、様々な音声認識モデルの性能を比較し、
日本語の音声認識において、
どのモデルが最も優れているかを評価することを目的としています。
処理速度、認識精度、そしてラズパイのようなリソースの限られた環境での動作可否。
それらの要素を総合的に評価することで、
実際の運用に最適なモデルを選定することができます。

以上が、音声認識技術に関する概要説明です。
この長いテキストが、各モデルによってどれだけ正確に認識されるか、
そして処理にどの程度の時間がかかるか。
その結果が、音声認識システムの選択における重要な判断材料となるでしょう。
""".replace("\n", " ").strip()


def gen_segment(text: str, out_path: Path, speaker_id: int = SPEAKER_ID) -> None:
    """VOICEVOX で1セグメントの音声を生成"""
    base = VOICEVOX_URL.rstrip("/")
    query_resp = requests.post(
        base + "/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=60,
    )
    query_resp.raise_for_status()
    query = query_resp.json()

    synth_resp = requests.post(
        base + "/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=300,
    )
    synth_resp.raise_for_status()

    out_path.write_bytes(synth_resp.content)
    duration = len(synth_resp.content) / (48000 * 2 * 2)
    return duration


def main():
    print("=" * 60)
    print("  5分テスト音声 生成")
    print("=" * 60)

    # テキストをセグメントに分割（VOICEVOXの制限を考慮して約200文字/セグメント）
    chars_per_seg = 200
    text = LONG_TEXT
    segments = []
    for i in range(0, len(text), chars_per_seg):
        seg = text[i:i + chars_per_seg]
        # 文の途中で切らないように。最後の句点/読点で切る
        last_punct = max(seg.rfind("。"), seg.rfind("、"), seg.rfind("」"))
        if last_punct > chars_per_seg // 2:
            seg = seg[:last_punct + 1]
        segments.append(seg.strip())

    total_chars = sum(len(s) for s in segments)
    print(f"  テキスト: {len(text)}文字 → {len(segments)}セグメントに分割")
    print(f"  合計: {total_chars}文字")
    print()

    # 各セグメントを個別に合成
    seg_wavs = []
    total_duration = 0
    for i, seg in enumerate(segments, 1):
        seg_wav = OUT_DIR / f"seg_{i:02d}.wav"
        preview = seg[:30] + "..." if len(seg) > 30 else seg
        print(f"  [{i:2d}/{len(segments)}] {preview}")
        try:
            dur = gen_segment(seg, seg_wav)
            total_duration += dur
            seg_wavs.append(seg_wav)
            print(f"        ✅ {seg_wav.name} ({dur:.1f}秒)")
        except Exception as e:
            print(f"        ❌ エラー: {e}")
            return

    # セグメントを結合
    print(f"\n  結合中...（合計約{total_duration / 60:.1f}分）")
    out_wav = OUT_DIR / "test_5min.wav"

    # ffmpeg で concat
    concat_list = OUT_DIR / "_concat_list.txt"
    concat_list.write_text(
        "\n".join(f"file '{w.name}'" for w in seg_wavs) + "\n",
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    size_mb = out_wav.stat().st_size / (1024 * 1024)
    print(f"\n  ✅ 完成: {out_wav.name}")
    print(f"     サイズ: {size_mb:.1f} MB")
    print(f"     再生時間: 約{total_duration / 60:.1f}分")
    print(f"     セグメント数: {len(seg_wavs)}")

    # テキストも保存
    out_txt = OUT_DIR / "test_5min.txt"
    out_txt.write_text(text, encoding="utf-8")
    print(f"     テキスト: {out_txt} ({len(text)}文字)")

    print(f"\n  ベンチマーク実行:")
    print(f"    uv run python tools/stt_benchmark.py {out_wav} --ref-file {out_txt}")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()

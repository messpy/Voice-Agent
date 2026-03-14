# voicechat シーケンス設計書

最終更新: 2026-03-13

## 1. always_on シーケンス

1. 起動
2. 設定読込
3. sanity check
4. `ベリベリ` 待受録音
5. wake STT
6. wake 判定
7. 発話録音
8. 本 STT
9. AI 補正
10. command 判定
11. command 実行 または assistant 応答
12. TTS 読み上げ
13. ログ保存
14. 4 に戻る

## 2. phrase_test シーケンス

1. 起動
2. フレーズ一覧読込
3. style cycle 決定
4. 出題読み上げ
5. 復唱録音
6. STT
7. AI 補正
8. 評価
9. ログ保存
10. 次フレーズへ

## 3. timed_record シーケンス

1. 起動
2. `スタート` 読み上げ
3. 固定時間録音
4. `5分終了` 読み上げ
5. STT
6. AI 補正
7. ファイル保存
8. DB 保存
9. 終了

## 4. ssh_remote シーケンス

1. Pi 側で wav 作成
2. `ssh` で remote workdir 準備
3. `scp` で wav 送信
4. remote script 実行
5. remote 側で STT
6. remote 側で AI 補正
7. 結果ファイル生成
8. `scp` で Pi 側へ回収
9. Pi 側ログ保存

# voicechat テスト設計書

最終更新: 2026-03-13

## 1. 方針

このプロジェクトは音声、外部プロセス、外部サーバー依存が強い。

そのためテストは次の 3 層で考える。

- 設定妥当性確認
- 関数単位の軽量確認
- 実機統合確認

## 2. 設定確認

確認対象:

- YAML の重複キーがないこと
- `runtime.run_mode` が想定値であること
- `transcription_backend` が `local` / `ssh_remote` であること
- SSH 鍵パスが正しいこと
- `command_router.commands` が配列であること

## 3. 機能確認項目

### 3.1 phrase_test

- フレーズ一覧が読める
- style cycle が回る
- expected / recognized / elapsed が保存される

### 3.2 always_on

- wake word が検出できる
- コマンド一致時に shell command が呼ばれる
- command 未一致時に assistant 応答へ流れる

### 3.3 timed_record

- `スタート` が読まれる
- 指定秒数録音される
- `5分終了` が読まれる
- テキストと DB が保存される

### 3.4 ssh_remote

- `ssh` 接続できる
- `scp` 送信できる
- remote script 実行できる
- 結果回収できる

## 4. 実機試験観点

- マイク入力レベル
- スピーカーへの TTS 出力
- YouTube 混線時の誤認率
- ローカルとリモートの速度比較
- AI 補正あり/なし比較

## 5. 現状の自動確認

現時点では最低限として Python 構文確認を行う。

例:

```bash
cd <repo-root>
./.venv/bin/python -m py_compile tools/wake_vad_record.py
```

## 6. 今後の追加候補

- SQLite 出力の自動確認
- command router の純粋関数化と単体試験
- remote backend の疎通モック化

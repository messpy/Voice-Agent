# voicechat リモートオフロード設計書

最終更新: 2026-03-13

## 1. 目的

Raspberry Pi で録音した音声を別 PC に送り、重いモデルや別 backend で文字起こしする。

## 2. 適用条件

- Pi 上のローカル認識が遅い
- モデル比較をしたい
- 別 PC に `faster-whisper` 実行環境がある

## 3. 処理方式

1. Pi 側で wav 作成
2. `ssh` で remote 作業ディレクトリ作成
3. `scp` で wav 転送
4. remote python script 実行
5. remote で raw/corrected/json 作成
6. `scp` で Pi 側へ回収

## 4. 設定項目

対象: [config/config.yaml](config/config.yaml)

- `runtime.transcription_backend: ssh_remote`
- `remote.host`
- `remote.user`
- `remote.port`
- `remote.ssh_key`
- `remote.remote_workdir`
- `remote.remote_python`
- `remote.remote_script`
- `remote.whisper_model`
- `remote.device`
- `remote.compute_type`
- `remote.skip_correction`
- `llm.provider`
- provider ごとの API キー

## 5. 期待効果

- Pi の CPU 負荷低減
- モデル比較の柔軟化

## 6. 制約

- ネットワーク依存
- SSH 鍵認証依存
- remote 側性能が低いと改善幅が小さい

# systemd user service

`voicechat` をログイン後に自動起動する user service です。

## インストール

```bash
mkdir -p ~/.config/systemd/user
install -m 644 /home/kennypi/work/voicechat/systemd/voicechat.service ~/.config/systemd/user/voicechat.service
install -m 644 /home/kennypi/work/voicechat/systemd/voicechat-web.service ~/.config/systemd/user/voicechat-web.service
systemctl --user daemon-reload
systemctl --user enable --now voicechat.service
```

依存サービスごとまとめて扱う場合は、共通 unit を [../systemd/README.md](/home/kennypi/work/systemd/README.md) の手順で入れて `voicechat.target` を使います。

`voicechat.service` は起動後に `voicechat-web.service` を毎回 `start` するので、`voicechat` の再起動時にも `web-console` が自動で立ち上がります。`voicechat-web.service` を単体で `enable` する必要はありません。

既定の公開先は `0.0.0.0:8787` です。LAN 内の別端末からは `http://<このPCのIP>:8787` で開けます。

## 操作

```bash
systemctl --user status voicechat.service
systemctl --user status voicechat-web.service
systemctl --user restart voicechat.service
systemctl --user restart voicechat-web.service
systemctl --user stop voicechat.service
systemctl --user stop voicechat-web.service
```

## 解除

```bash
systemctl --user disable --now voicechat.service
rm ~/.config/systemd/user/voicechat.service
rm ~/.config/systemd/user/voicechat-web.service
systemctl --user daemon-reload
```

# systemd user service

`voicechat` をログイン後に自動起動する user service です。

## インストール

```bash
mkdir -p ~/.config/systemd/user
install -m 644 /home/kennypi/work/voicechat/systemd/voicechat.service ~/.config/systemd/user/voicechat.service
systemctl --user daemon-reload
systemctl --user enable --now voicechat.service
```

依存サービスごとまとめて扱う場合は、共通 unit を [../systemd/README.md](/home/kennypi/work/systemd/README.md) の手順で入れて `voicechat.target` を使います。

## 操作

```bash
systemctl --user status voicechat.service
systemctl --user restart voicechat.service
systemctl --user stop voicechat.service
```

## 解除

```bash
systemctl --user disable --now voicechat.service
rm ~/.config/systemd/user/voicechat.service
systemctl --user daemon-reload
```

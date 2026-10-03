# Ubuntu Deploy

這份文件只描述現在的正式交付流程。

## 1. 取得交付包

如果你是從 repo 本體打包：

```bash
bash scripts/package_ubuntu_bundle.sh
```

會得到：

```text
dist/voice_pick_ubuntu_bundle.tar.gz
```

如果你已經拿到壓縮檔，直接把它放到 Ubuntu 主機。

## 2. 解壓

```bash
tar -xzf voice_pick_ubuntu_bundle.tar.gz
cd voice_pick_ubuntu_bundle
```

如果你搬的是整個 repo，也可以直接：

```bash
scp -r voice_pick user@ubuntu-host:~/
ssh user@ubuntu-host
cd ~/voice_pick
```

## 3. 執行自動部署

```bash
bash scripts/deploy_ubuntu.sh
```

這一步會做：

- 安裝 Ubuntu 套件
- 建立 `venv`
- 安裝 `requirements.txt`
- 建立 `data/runtime/logs`、`data/recordings`、`data/models/yolo`
- 若不存在則建立 `config/env/ubuntu.local.yaml`

## 4. 修改機器專屬設定

只先改：

```text
config/env/ubuntu.local.yaml
```

至少要檢查：

- `demo.arm.connections[*].host`
- `demo.arm.connections[*].port`
- `demo.gripper.endpoints[*].host`
- `demo.gripper.endpoints[*].port`
- `demo.sensor_api.base_url`
- `demo.cameras.cam1.serial`
- `demo.cameras.cam2.serial`
- `demo.cameras.claw.source`
- `demo.teach.auto_label.model_path`

詳細對照表看：

- `docs/CONFIG_MAP.md`

## 5. 跑 preflight

```bash
bash scripts/preflight_ubuntu.sh
```

它會檢查：

- launcher config 載入
- RealSense serial
- `/dev/video*`
- `/dev/serial/by-id`
- arm / gripper / sensor API reachability

## 6. 啟動系統

前景：

```bash
bash scripts/run_demo.sh full
```

如果要直接用舊版 UI：

```bash
bash scripts/run_demo.sh full --ui-mode classic
```

背景：

```bash
venv/bin/python -m src.launcher start full --env-config config/env/ubuntu.local.yaml
```

狀態：

```bash
bash scripts/run_demo.sh status
bash scripts/run_demo.sh logs
bash scripts/run_demo.sh stop
```

## 7. 開 UI

```text
http://<ubuntu-ip>:8090
```

## 建議搬機順序

1. 先只確認網路與 IP
2. 再確認 arm / gripper / sensor API
3. 再接 RealSense
4. 最後接 claw cam

這樣比較容易定位問題，不會一開始全部混在一起。

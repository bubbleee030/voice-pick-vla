# Ubuntu Troubleshooting

## `pyrealsense2` import 失敗

先看：

```bash
venv/bin/pip show pyrealsense2
```

再看裝置：

```bash
lsusb | grep -i realsense
```

如果套件有、裝置也有，但還是失敗，通常是：

- wheel 與 Python 版本不對
- 裝置權限或 USB 層有問題

## 找不到 RealSense serial

先跑：

```bash
bash scripts/preflight_ubuntu.sh
```

如果 preflight 列不出 serial，先不要懷疑 UI，先處理 USB / device 層。

## 找不到 claw cam

先看：

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

再把對應 device 寫進：

```text
config/env/ubuntu.local.yaml
demo.cameras.claw.source
```

如果 index 常變，優先直接寫 `/dev/videoX`。

## `run_demo.sh` 起不來

先查：

```bash
bash scripts/run_demo.sh status
bash scripts/run_demo.sh logs
```

常見原因：

- `config/env/ubuntu.local.yaml` 沒改
- port `8090` 被佔用
- `venv` 沒建好
- 某個硬體模組在初始化時直接失敗

如果只是要先進 UI 排錯，可先用：

```bash
bash scripts/run_demo.sh debug-modules
```

## 手臂仍然很慢

先看有沒有 speed readback mismatch。

再檢查：

- `config/modbus_config.yaml -> safety.max_speed_percent`
- `config/demo_config.yaml -> speed.presets`
- `config/env/ubuntu.local.yaml` 是否覆蓋到舊值

最後再做實機對照：

- UI `high`
- 廠商程式 `high`

如果兩者差很多，表示 controller 之外還有裝置端限制。

## Gripper API 連不上

先測：

```bash
curl -sS http://<gripper-ip>:<port>/
```

如果你配置了多個 endpoint，確認：

- `ubuntu.local.yaml` 內順序正確
- 失敗的 endpoint 沒有卡太久 timeout

## Sensor API 連不上

先確認 `demo.sensor_api.base_url` 可從 Ubuntu 主控筆電直連。

如果 API 在 AGX 或其他遠端機器，優先排除：

- 防火牆
- route
- subnet 不同
- 服務沒有 listen `0.0.0.0`

## Dataset Capture 沒有輸出完整檔案

先檢查：

- `data/recordings/<object>/episode_xxx/metadata.json`
- `bash scripts/run_demo.sh logs`

常見原因：

- 選到的 stream 沒有真的啟動
- claw cam 在錄製期間掉線
- sensor API 沒有回資料

## Teach Replay 行為不對

先確認你選的是：

- 原始 teach recording
- 還是 `.merged` recording

如果是要帶 claw / external timeline 的 replay，優先用已驗證過的 merged recording。

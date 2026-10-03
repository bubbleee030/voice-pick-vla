# Operations

這份文件描述 Ubuntu 主控版日常操作。

## 啟動模式

### `full`

完整主控台。

包含：

- UI
- arm
- gripper
- dual RealSense
- claw cam
- dataset capture
- teach / replay

```bash
bash scripts/run_demo.sh full
```

切回舊版 UI：

```bash
bash scripts/run_demo.sh full --ui-mode classic
```

如果要用 `VLAtest` 那份 classic snapshot，而不是內建 fallback 頁，先抓一次：

```bash
bash scripts/fetch_vlatest_classic_ui.sh
```

### `recording`

偏錄製用途。

適合：

- 錄 dataset
- 錄 teach path
- 檢查 camera 與 telemetry

```bash
bash scripts/run_demo.sh recording
```

### `debug-modules`

模組分批 debug。

適合：

- 某個硬體還沒接好
- 想先只看 UI / config / log
- 想降級啟動排錯

```bash
bash scripts/run_demo.sh debug-modules
```

也可以直接在 `config/env/ubuntu.local.yaml` 設：

```yaml
demo:
  ui:
    mode: "classic"
```

可用值只有：

- `modern`
- `classic`

## Dataset Capture

在 UI 的 `Recording Console` 建立 episode。

輸出路徑：

```text
data/recordings/<object>/episode_xxx/
```

預設內容：

- `cam1_rgb`
- `cam1_depth`
- `cam2_rgb`
- `cam2_depth`
- `claw_rgb`
- `trajectory.csv`
- `gripper_stream.csv`
- `sensor_stream.csv`
- `metadata.json`

`metadata.json` 會記錄：

- object
- duration
- fps
- telemetry_hz
- selected streams
- notes

## Teach & Replay

Teach recordings 放在：

```text
data/teach_recordings/
```

UI 可做：

- start teach
- save waypoint
- stop teach
- select replay target

Replay 模式：

- `raw`
- `phase_axis_split`

## Quick Demo Replay

有固定 pose 的物件可直接走 fixed pick。

如果要走 teach path：

1. 選 object
2. 選 teach recording 或 merged recording
3. 選 replay mode
4. 送出 pick request

## Calibration

校正工具：

```bash
venv/bin/python tools/calibrate.py --help
```

現有校正資料放在：

```text
data/calibration/
```

## Dataset Post-Process

轉 dataset：

```bash
venv/bin/python tools/prepare_dataset.py --help
```

genlock merge：

```bash
venv/bin/python tools/genlock_merge.py --help
```

## 管理命令

```bash
bash scripts/run_demo.sh status
bash scripts/run_demo.sh logs
bash scripts/run_demo.sh stop
```

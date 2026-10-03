# Ubuntu Main-Control Refactor Summary

這次重構把專案正式收斂成 Ubuntu x86 laptop 主控版，不再把 macOS、Windows、舊測試流程一起當成主路徑。

## 重構結果

- 主控入口統一成 `launcher + Flask UI backend`
- arm / gripper / RealSense / claw cam 都拆成 service
- dataset capture 與 teach replay 走同一套 backend
- 設定集中到 `config/*.yaml` 與 `config/env/ubuntu.local.yaml`
- 手臂速度控制改成和 `VLAtest` 對齊的寫入與 readback verify
- UI 改成控制台式版面

## 核心檔案

執行骨架：

- `src/runtime_config.py`
- `src/launcher.py`
- `tools/voice_pick_demo.py`

硬體 / 工作流 service：

- `src/services/arm_service.py`
- `src/services/gripper_service.py`
- `src/services/realsense_service.py`
- `src/services/claw_camera_service.py`
- `src/services/recording_service.py`
- `src/services/health_service.py`

設定：

- `config/demo_config.yaml`
- `config/modbus_config.yaml`
- `config/launch_profiles.yaml`
- `config/env/ubuntu.example.yaml`

## 現在的主邏輯

- Ubuntu 筆電是唯一正式交付平台
- 機器差異只應該透過 `config/env/ubuntu.local.yaml` 覆蓋
- `full / recording / debug-modules` 由 launcher profile 切換
- 錄 dataset 和跑 demo replay 都從同一個 UI 操作

## 這次順便清掉了什麼

- Windows 舊打包內容
- macOS RealSense daemon / tunnel / sudo wrapper 舊流程
- 舊 CLI demo/API
- 測試腳本、session handoff、過時 SOP、暫存打包檔

## 你接下來最該做的事

1. 在 Ubuntu 上解壓 bundle
2. 跑 `bash scripts/deploy_ubuntu.sh`
3. 修改 `config/env/ubuntu.local.yaml`
4. 跑 `bash scripts/preflight_ubuntu.sh`
5. 跑 `bash scripts/run_demo.sh full`

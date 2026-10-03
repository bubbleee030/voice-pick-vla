# Gripper/Tactile V2 Summary

這次改動把正式資料錄製的 gripper+tactile 來源固定成 AGX 二合一資料流。

## 改了什麼

- 新增 [tools/gripper_record_CSV_API_V2.py](../tools/gripper_record_CSV_API_V2.py)
  - 提供 `v2` API
  - 同時讀三指位置與三路 tactile 值
  - 支援 `GET /health`、`GET /state`、`POST /recording/start`、`POST /recording/stop`
  - 支援 `sensor_connected` 與 `sample_valid`
- 保留舊版 [tools/gripper_record_CSV_API_LSTM.py](../tools/gripper_record_CSV_API_LSTM.py) 當 fallback
  - 補了 `/health`
  - `GET /state` 補齊 `server_time_unix`、`tactile_data`、`sensor_connected`
- 主控 laptop 端的 [src/services/gripper_service.py](../src/services/gripper_service.py)
  - 會先找 `v2`
  - 找不到才退回舊版
  - dataset capture 會自動觸發 AGX raw CSV start/stop
- 錄製 runtime 與後處理
  - `gripper_stream.csv` 改成正式欄位
  - `prepare_dataset.py` 可讀新舊格式
  - VLA 輸出新增 `tactile.npy`、`tactile_valid.npy`、`tactile_connected.npy`

## 新舊流程差異

舊流程：

- AGX 自己錄 `tactile_data_*.csv`
- laptop 只知道 gripper API 是不是能控制
- tactile 斷線時可能還會沿用最後一筆值

新流程：

- laptop episode 是正式資料主體
- AGX raw CSV 保留作備份
- AGX `v2` 明確回傳：
  - `tactile_data`
  - `tactile_timestamp_unix`
  - `sensor_connected`
  - `sample_valid`
- 錄製開始與結束由 laptop 自動通知 AGX

## 你接下來該做什麼

1. 在 AGX 上優先跑 `v2`：
   - `python tools/gripper_record_CSV_API_V2.py`
2. 在 Ubuntu 主控筆電確認 `config/env/ubuntu.local.yaml` 的 gripper endpoint 順序：
   - `5003` 放前面
   - `5002` 放 fallback
3. 跑一次 `bash scripts/preflight_ubuntu.sh`
4. 啟動主控：
   - `bash scripts/run_demo.sh full`
5. 先錄一個短 episode，確認：
   - `metadata.json` 裡有 `gripper_recording`
   - `gripper_stream.csv` 有 `val1/val2/val3`
   - AGX 端也有 raw CSV

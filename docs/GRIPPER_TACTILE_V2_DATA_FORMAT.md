# Gripper/Tactile V2 Data Format

## AGX `GET /state`

正式欄位：

- `api_version`
- `server_time_unix`
- `elapsed_s`
- `current_pos`
- `tactile_data`
- `tactile_timestamp_unix`
- `sensor_connected`
- `sample_valid`
- `sensor_last_update_unix`
- `sensor_disconnect_count`
- `recording`
- `recording_session_id`
- `raw_csv_path`
- `last_error`

## AGX raw CSV

欄位固定為：

```text
timestamp_unix,elapsed_s,pos1,pos2,pos3,val1,val2,val3,tactile_timestamp_unix,sensor_connected,sample_valid
```

語意：

- `pos1/pos2/pos3`: 三指位置
- `val1/val2/val3`: 三路 tactile / 貼片電阻值
- `sensor_connected`: 物理 sensor 是否仍在 freshness window 內
- `sample_valid`: 這一筆 tactile 是否可當有效樣本使用

## laptop `gripper_stream.csv`

主控版寫出的欄位：

```text
timestamp_unix,elapsed_s,server_time_unix,remote_elapsed_s,pos1,pos2,pos3,val1,val2,val3,tactile_timestamp_unix,sensor_connected,sample_valid,rtt_ms,clock_offset_sec,api_version,recording_session_id,status,error
```

這份是 train pipeline 應優先使用的 episode-local gripper/tactile 記錄。

## VLA 輸出新增檔案

`prepare_dataset.py` 產出：

- `gripper_pos.npy`
- `gripper_open.npy`
- `tactile.npy`
- `tactile_valid.npy`
- `tactile_connected.npy`

## Metadata 新欄位

`metadata.json` 會新增：

- `gripper_recording.api_version`
- `gripper_recording.start`
- `gripper_recording.stop`
- `gripper_recording.integrated_tactile`
- `gripper_recording.fallback_used`
- `gripper_recording.sensor_disconnect_count`

# Gripper/Tactile V2 Operations

這份文件是正式錄製前的操作清單。

## 1. AGX 上啟動哪支程式

優先：

```bash
python tools/gripper_record_CSV_API_V2.py
```

fallback：

```bash
python tools/gripper_record_CSV_API_LSTM.py
```

建議不要兩支都綁同一個 port。

預設：

- `v2`: `5003`
- legacy: `5002`

## 2. Ubuntu 筆電怎麼設定

確認 `config/env/ubuntu.local.yaml`：

```yaml
demo:
  gripper:
    endpoints:
      - host: "192.168.1.100"
        port: 5003
        label: "agx_v2_primary"
      - host: "192.168.1.100"
        port: 5002
        label: "agx_legacy_fallback"
```

如果你正式要靠 AGX 二合一資料流，`5003` 應該放第一個。

## 3. 啟動前要檢查什麼

在 Ubuntu 筆電：

```bash
curl http://192.168.1.100:5003/health
curl http://192.168.1.100:5003/state
```

你要看到：

- `api_version: "v2"`
- `current_pos`
- `tactile_data`
- `sensor_connected`

如果 `5003` 沒起來，再測：

```bash
curl http://192.168.1.100:5002/health
curl http://192.168.1.100:5002/state
```

## 4. 正式錄製怎麼跑

啟動主控：

```bash
bash scripts/run_demo.sh full
```

在 UI 開始 dataset capture 後，主控會自動：

1. 對 AGX 發 `recording/start`
2. 錄 `cam1/cam2/arm/gripper/tactile`
3. 停止時對 AGX 發 `recording/stop`

## 5. 錄完後你要看哪裡

laptop episode 內：

- `gripper_stream.csv`
- `metadata.json`

AGX 端：

- `data/agx_tactile_raw/` 下的 raw CSV

`metadata.json` 要看：

- `gripper_recording.start`
- `gripper_recording.stop`
- `gripper_recording.api_version`
- `gripper_recording.sensor_disconnect_count`

如果 `fallback_used=true`，代表這次不是用 `v2` 正常錄的。

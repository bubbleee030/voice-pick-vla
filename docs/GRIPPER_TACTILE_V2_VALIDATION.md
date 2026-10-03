# Gripper/Tactile V2 Validation

這份文件是錄完後的檢查清單。

## 1. 先檢查 episode 內容

在 laptop episode 目錄內至少要有：

- `cam1_rgb/`
- `cam2_rgb/`
- `trajectory.csv`
- `gripper_stream.csv`
- `metadata.json`

如果有用 claw：

- `claw_rgb/`

## 2. 檢查 `gripper_stream.csv`

先看 header：

```bash
head -n 5 data/recordings/<object>/episode_XXX/gripper_stream.csv
```

你應該看到：

- `pos1,pos2,pos3`
- `val1,val2,val3`
- `sensor_connected`
- `sample_valid`

## 3. 檢查是不是其實在 fallback

看：

```bash
jq '.gripper_recording' data/recordings/<object>/episode_XXX/metadata.json
```

重點：

- `api_version` 最好是 `v2`
- `fallback_used` 最好是 `false`
- `start.ok` 應該是 `true`
- `stop.ok` 應該是 `true`

## 4. 檢查 sensor 是否有斷線

看：

```bash
jq '.gripper_recording.sensor_disconnect_count' data/recordings/<object>/episode_XXX/metadata.json
```

如果不是 `0`，代表錄製過程中有發生 tactile 物理斷線或 timeout。

再看資料列：

```bash
python3 - <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
bad = sum(1 for r in rows if str(r.get("sample_valid", "0")) in {"0", "false", "False", ""})
disc = sum(1 for r in rows if str(r.get("sensor_connected", "0")) in {"0", "false", "False", ""})
print("rows =", len(rows))
print("invalid tactile rows =", bad)
print("disconnected rows =", disc)
PY data/recordings/<object>/episode_XXX/gripper_stream.csv
```

## 5. 轉 VLA 後要看什麼

轉完後檢查：

- `tactile.npy`
- `tactile_valid.npy`
- `tactile_connected.npy`

如果 `tactile_valid.npy` 幾乎全是 0，這批資料不適合直接拿來 train tactile-conditioned 模型。

## 6. 最小可接受標準

一批正式資料至少應滿足：

- episode 內影像完整
- `trajectory.csv` 正常
- `gripper_stream.csv` 有位置與 tactile 欄位
- `metadata.json` 記錄 `api_version`
- `sensor_disconnect_count == 0` 或可明確標出哪些片段需排除

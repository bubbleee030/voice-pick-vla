# Config Map

Ubuntu 搬機後，優先改 overlay，不要第一輪就直接改 base config。

## 主檔案

```text
config/env/ubuntu.local.yaml
```

`deploy_ubuntu.sh` 第一次會從 `config/env/ubuntu.example.yaml` 複製出這個檔案。

## 必改鍵值

| Key | 用途 | 範例 |
| --- | --- | --- |
| `demo.arm.connections[0].host` | 手臂 controller IP | `192.168.1.33` |
| `demo.arm.connections[0].port` | Modbus TCP port | `1502` |
| `demo.gripper.endpoints[0].host` | gripper API IP | `192.168.1.50` |
| `demo.gripper.endpoints[0].port` | gripper `v2` API port | `5003` |
| `demo.gripper.endpoints[1].port` | gripper legacy fallback port | `5002` |
| `demo.sensor_api.base_url` | 額外外部 sensor API，不是 AGX tactile v2 主來源 | `http://192.168.1.60:5000` |
| `demo.cameras.cam1.serial` | RealSense 1 serial | `123456789012` |
| `demo.cameras.cam2.serial` | RealSense 2 serial | `234567890123` |
| `demo.cameras.claw.source` | claw cam UVC source | `0` 或 `/dev/video4` |
| `demo.cameras.claw.performance_profile` | claw cam profile | `balanced` |
| `demo.teach.auto_label.model_path` | YOLO model path | `data/models/yolo/yolo12sbest.pt` |

## 常改但第二層才動

| Key | 用途 |
| --- | --- |
| `demo.speed.presets.low` | UI low speed |
| `demo.speed.presets.medium` | UI medium speed |
| `demo.speed.presets.high` | UI high speed |
| `demo.ready_pose` | ready pose |
| `demo.home_pose` | staged home fallback |
| `demo.home_routing.mode` | `native_1405` 或 `staged_route` |
| `demo.teach.dataset_capture.image_fps` | capture fps |
| `demo.teach.dataset_capture.telemetry_hz` | telemetry rate |
| `demo.modules.*` | UI / module enablement |

## 通常不要先改的檔案

- `config/demo_config.yaml`
- `config/modbus_config.yaml`
- `config/launch_profiles.yaml`

原則：

- 先改 overlay
- 確定需求真的是全局預設才去改 base config
- 如果正式資料要用 AGX tactile v2，優先確認 `demo.gripper.endpoints`，不要先把 `demo.sensor_api` 當成 tactile 主來源

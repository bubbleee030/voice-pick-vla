# Speed Control Notes

這份文件說明 AutoVLA、固定 fallback 與 Modbus controller 的實際速度鏈。

## AutoVLA 不是只有一個速度

`config/demo_config.yaml` 的三個 AutoVLA 速度控制不同階段：

| Key | 實際控制範圍 |
|---|---|
| `vla_auto.speed_percent` | VLA 預測步與爪相機 servo 的 X/Y 修正 |
| `vla_auto.travel_speed_percent` | hover、lift、release travel、奶油刀連續上抬 |
| `vla_auto.descend_speed_percent` | vision seek、抓取下降、release-down |

CLI `--speed N` 只覆寫第一項 correction speed；它不會改 lift 或 descend。
固定 fallback 則使用：

- `fixed_fallback.travel_speed_percent`
- `fixed_fallback.descend_speed_percent`

每次 AutoVLA 或固定 fallback 開始時都會重新讀取 `demo_config.yaml`，不用為這些
值重啟 server。run log 會先列出 correction/travel/descend 三個值，避免把不同
階段誤認為同一個速度。

## Controller 寫入與確認順序

`robot_speed_override` (`0x0246`) 只在 `servo_on()` 寫一次，避免它和每次 GO
非同步競賽。每個 `move_to()` 的順序是：

1. target pose
2. acceleration / deceleration（有啟用時）
3. user/tool frame 與 posture（有啟用時）
4. move mode
5. speed percent (`0x0324`)
6. 讀回 speed、ACC、DEC
7. 只有 speed readback 正確才寫 motion command (`0x0300`)

`verify_speed_write: true` 時，controller 依
`speed_verify_attempts` / `speed_verify_retry_s` 做有限次確認。若設定 `60%` 但
讀回不是 `60%`，該步會在 GO **之前中止**，不會沿用上一個速度偷偷移動。
完成一個 move 後，AutoVLA log 會顯示 controller-confirmed speed、MovP/MovL、
ACC 與 DEC。

## ACC / DEC 會套用在哪裡

來源是 `config/modbus_config.yaml -> motion`：

- `set_acceleration` / `acceleration_raw`
- `set_deceleration` / `deceleration_raw`

AutoVLA 與固定 fallback 最終都呼叫同一個 `ArmController.move_to()`，因此兩條
路徑的**每一個 move**都會套用已啟用的 ACC/DEC。server 連線後再修改這些值也
可以；下一次 AutoVLA 或固定 fallback 開始前會只更新 motion/safety 設定，不會
熱換連線 IP 或 register map。

## 為什麼相同百分比體感仍可能不同

預設 `use_linear_move: false` 使用 MovP。百分比作用在 joint speed，因此純 Z、
純 Y 與多軸 swing 的 Cartesian mm/s 不會相同；ACC/DEC ramp 也會影響短路徑的
平均速度。這不是 log-only：先看每步的 `controller confirmed speed=...`，再用
Arm Monitor 的實測 mm 變化或 `scripts/arm_speed_audit.py` 比較移動時間。

若需要較一致的 Cartesian tool speed，可在低速、工作空間中央先測
`use_linear_move: true` (MovL)；直線路徑接近 singularity 時 MovL 可能 alarm。

## 安全上限與驗收

最終速度仍受 `config/modbus_config.yaml -> safety.max_speed_percent` 限制。
建議用安全短路徑分別測 25%、60%、100%，每組確認：

- run log 的 requested 與 controller-confirmed speed 相同；
- Arm Monitor 的 X/Y/Z 在移動途中持續更新；
- 實測移動時間隨百分比有明顯變化；
- backend 沒有 `speed register rejected command before GO`。

若 readback 正確但體感仍不變，再查 operation mode、global override、MovP/MovL、
ACC/DEC 與裝置端速度限制。

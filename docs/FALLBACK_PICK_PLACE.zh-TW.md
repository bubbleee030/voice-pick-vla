# 後備夾取放置（免模型）— 使用說明

針對 **trapezoid（梯形）、board（板子）、tweezers（鑷子）** 的一套確定性、**免模型**
夾取→放置流程，於 VLA 模型在筆電上無法使用或不穩定時作為後備。它**只用**手臂
（Modbus）與夾爪（HTTP）—— **不使用相機、RealSense、AGX 觸覺、CUDA** —— 因此排除了
過去造成錄製／重播當掉的脆弱子系統。

你只需**教一次**（夾取姿態 + 夾持 + 手指鬆開），數值會被**燒錄進設定檔**，執行時在
UI 上**每個物件按一個按鈕**即可。

---

## 1. 運作方式

針對所選物件，執行器會跑這段固定流程：

```
ready → hover → pregrasp → grasp（夾緊）→ 停留 → lift
      → ready（安全過渡點）
      → place_above → 下降到放置高度 → release（依物件）→ 收回 → ready
```

- **夾取**由單一錄製的 `grasp_pose` 垂直向下合成（hover/pregrasp/lift 都是其上方的 z 偏移）。
  只錄一個姿態，動作完全確定。
- **放置**移動到一個共用的接近點，下降到該物件的高度，再執行該物件的手指鬆開序列。

### 穩定性保護（為何不會再像以前那樣「死掉」）
| 故障模式 | 保護機制 |
|---|---|
| 手臂中途凍結 | 執行期間**暫停姿態輪詢**（獨佔 Modbus，無鎖競爭）＋ **每步到位看門狗**，移動未到位即中止。 |
| 重播無聲中止 | **每步都記錄** START／完成；**任何**例外都導向安全停止 —— 不會有執行緒悄悄死掉。 |
| 夾爪卡住／失敗 | 有限次**重試**；**鬆開**失敗視為硬中止（物件絕不會被悄悄卡住不放）。 |

**任何失敗時：停止並保持。** 手臂送出 `motion_stop`、**保持伺服 ON**、原地凍結且仍夾著
物件，並記錄明顯錯誤。後續由操作員手動處理。

### 設定檔分工（預設值 vs 校正值）
- `config/objects.yaml` → `fallback:` —— 已提交、人類可讀的**種子值**（佔位用）。
- `config/demo_config.yaml` → `fallback:` —— 共用設定（接近姿態、速度、看門狗）。
- `config/fallback_taught.yaml` —— 由教學工具**自動產生**；每次執行時深層合併覆蓋種子值。
  重教只會重新產生這個檔，永遠不會弄壞已整理好的設定檔。若此檔不存在，則使用種子值。

---

## 2. 前置條件

- **教學前先停掉 demo 伺服器。** 手臂控制器只允許單一 Modbus 連線，伺服器與教學工具
  不能同時佔用。
- 手臂可連線（例如 `192.168.1.232:502`）、夾爪可連線（例如 `192.168.1.100:5003`）。
- 全部在 **`voice_pick` conda 環境**中執行。
- 本機 UI 為 **classic** 模式；按鈕出現在 **Quick Pick** 面板的 **「Fallback P→P」**
  那一排（Trapezoid / Board / Tweezers）。modern 模式也有。

---

## 3. 教學（階段 A）— CLI

停掉伺服器後：

```bash
python scripts/fallback_teach.py --object trapezoid
# 預設會自動偵測 env overlay：config/env/ubuntu.local.yaml
```

你自己移動手臂（teach pendant 或手動引導）；工具只**讀取**姿態並**控制夾爪**，
絕不會移動手臂。

### 指令
| 指令 | 功能 |
|---|---|
| `pick` | 一次擷取夾取**姿態 + 夾緊狀態**（建議） |
| `pose` / `grip` | 只擷取夾取姿態 / 只擷取手指夾持位置 |
| `open [d]` / `close [d]` / `set a b c` | 微調夾爪（用來擺手指位置）；可選穩定延遲 |
| `rstep [delay]` | 把**目前手指位置**附加為一個鬆開步驟 |
| `ropen [delay]` | 附加最後的**完全張開**鬆開步驟 |
| `rundo` / `rclear` | 移除最後一個 / 清空所有鬆開步驟 |
| `z [mm]` | 擷取放置**下降高度**（目前手臂 z，或直接給 mm 值） |
| `approach` | 擷取**共用**的放置接近姿態（目前手臂姿態） |
| `state` / `show` | 印出即時手臂+夾爪 / 已擷取的草稿 |
| `preview` | **乾跑**完整流程（草稿疊加已存設定）—— 無動作 |
| `save` | 將草稿寫入 `config/fallback_taught.yaml`（合併） |
| `servo on\|off` | 伺服控制（注意：`off` 可能讓手臂因重力下垂） |
| `quit` | 離開（有未存擷取時會警告） |

### 範例流程
```
teach[trapezoid]> close          # 在物件上夾緊手指
teach[trapezoid]> pick           # 擷取夾取姿態 + 夾持
teach[trapezoid]> approach       # 先移到放置區上方，再擷取
teach[trapezoid]> z              # 在該處擷取下降高度
teach[trapezoid]> set 40 40 40   # 手指張開到一半
teach[trapezoid]> rstep          # 記錄為鬆開步驟 1
teach[trapezoid]> ropen          # 最後完全張開
teach[trapezoid]> preview        # 檢查流程
teach[trapezoid]> save
teach[trapezoid]> quit
```
對 `board` 與 `tweezers` 重複以上步驟。

---

## 4. 預覽（乾跑，無動作）

非互動式，不需硬體：

```bash
python scripts/fallback_teach.py --object trapezoid --preview
```

它會印出執行器將執行的完整有序流程 —— 姿態（原始值 + mm）、速度、夾持模式、鬆開
步驟 —— 並對仍在使用的佔位值提出警告。預覽使用與執行器**相同的規劃器**，所以你看到
的就是會實際執行的。

---

## 5. 執行（階段 B）— UI

1. **重啟一次 demo 伺服器**（載入新的程式碼／UI／設定）。之後重教不需重啟 ——
   每次按按鈕都會重新讀取教學檔。
2. 在 UI 連接手臂 + 夾爪。
3. 在 **Quick Pick** 面板的「Fallback P→P」按 **Trapezoid / Board / Tweezers**。
4. 觀察記錄：`START → move ready → … → release → … → COMPLETE`。

---

## 6. 設定檔參考

`config/objects.yaml`：
```yaml
fallback:
  trapezoid:
    grasp_pose: [x, y, z, rx, ry, rz]   # 控制器單位（um / 0.001 度）
    grasp_finger_pos: [f1, f2, f3]      # null -> 使用 close()
    place:
      z_mm: 312.5                       # 放置區的下降高度
      release_steps:                    # 有序手指鬆開
        - {pos: [f1, f2, f3], delay_s: 0.3}
        - open
```

`config/demo_config.yaml`：
```yaml
fallback:
  place_approach_pose: [x, y, z, rx, ry, rz]
  speeds: {travel: 60, descend: 20}
  settles: {grip_hold_s: 0.8}
  hover_z_offset_um: 180000
  pregrasp_z_offset_um: 35000
  lift_z_offset_um: 200000
  watchdog: {per_step_timeout_s: 25.0, motion_start_timeout_s: 3.0, gripper_retries: 2}
```

`config/fallback_taught.yaml`（自動產生；請勿手動編輯）：
```yaml
fallback:
  trapezoid: { ... 教學數值 ... }
shared:
  place_approach_pose: [ ... ]
```

---

## 7. 調校（階段 3）

預設將 `place.z_mm` 與 `place_approach_pose` 設為安全的 **ready 高度**，讓首次執行能
**不下降到桌面**的情況下驗證動作。在實機上調校：
- 用 `approach` + `z` 設定真正的放置位置／高度，
- 用 `set`/`rstep`/`ropen` 錄製乾淨的每物件手指鬆開，
- 每次改完先 `preview`，再 `save`。

---

## 8. 疑難排解

| 症狀 | 可能原因／處理 |
|---|---|
| `arm not connected` | 伺服器仍佔用 Modbus 連接埠 —— 先停掉它。檢查手臂 IP/port。 |
| `gripper not reachable` | 夾爪端點未啟動 —— 檢查 `192.168.1.100:5003/5002`。鬆開需要夾爪，沒有它流程不會啟動。 |
| `safety reject <step>` | 某姿態超出 `safety_boundary`/`pose_limits`。請在範圍內重教。 |
| `watchdog: '<step>' did not reach in-position` | 手臂未及時到位 —— 中止 + 停止保持。檢查手臂、到位旗標、速度。 |
| UI 看不到按鈕 | 你在 classic UI 且更新後未重啟伺服器 —— 重啟它。 |
| 流程中止、手臂仍夾著物件 | 這是失敗時預期的「停止並保持」。請手動處理（伺服維持 ON、夾爪維持不變）。 |

---

## 9. 相關檔案

- `scripts/fallback_teach.py` —— 教學 + 燒錄 + 預覽 CLI
- `src/services/arm_service.py` —— `build_fallback_plan()`（規劃器）+ `run_fallback()`（執行器 + 保護）
- `src/runtime_config.py` —— `load_taught_fallback()` / `merge_fallback()`
- `tools/voice_pick_demo.py` —— `fallback_run` socket 處理器
- `tools/static/index.classic.html` / `app.classic.js`（及 modern 的 `index.html` / `app.js`）—— 三個按鈕
- `config/objects.yaml`、`config/demo_config.yaml`、`config/fallback_taught.yaml` —— 設定

# 三物件 Demo — 終端機逐步測試手冊(繁體中文)

原則:**先在終端機把每一層測通,最後才用 UI。**
順序由安全到危險:離線 → 只看不動 → 模擬 → 低速實機 → UI。
(英文操作手冊:`docs/MANUAL_THREEOBJ_DEMO.md`;規格:`HANDOFF_threeobj_laptop.md` §3)

---

## 第 0 步 — 前置檢查(每次測試前)

```bash
# 1. Demo server:離線測試(第 1 步)/ --observe / --local 要「關」;
#    但一般自動流程(第 5.5/6 步)server「開著」也行 —— CLI 會自動 attach 到
#    server,網頁保持開著能看爪相機畫面(相機/Modbus 仍是 server 單一擁有)。
ps aux | grep "src.launcher" | grep -v grep     # 要關時:kill <確切PID>(絕不用 pkill -f)

# 2. 一律用 conda 環境 voice_pick(不要用 venv,GPU 會壞)
conda activate voice_pick    # 或每條指令前加 conda run -n voice_pick

# 3. 手臂連線(自動測試路由)
ping -c 2 192.168.1.232
```

安全守則(不變):
- 側邊兩台相機**絕對不能移動**(動了模型就失效)。
- 第一次實機自動流程:低速、手放在停止鈕旁。
- 手臂絕不自動回 home。


---

## 第 0.5 步 — 重啟 Demo server／手指馬達(可直接複製)

> **先做安全確認：**停止中的手臂、手指沒有夾住物件，且手遠離手指活動範圍。
> 以下「home」都會把三根手指張開到 home 位置 `[3072,3072,2048]`。

### A. 重啟 Ubuntu Demo server（UI、相機、手臂控制）

在 Ubuntu 筆電、本專案根目錄執行：

```bash
# 先停止目前由 launcher 管理的 server，確認已停止後再啟動
bash scripts/run_demo.sh stop
bash scripts/run_demo.sh status

# 啟動完整 Demo；若你平常使用 classic UI，改成：
# bash scripts/run_demo.sh full --ui-mode classic
bash scripts/run_demo.sh full
```

開啟 `http://127.0.0.1:8090`，確認 UI、Claw Cam 與 Arm 狀態正常。若 server 是在另一台機器，將 `127.0.0.1` 換成該 Ubuntu 的 IP。

**重要：重啟 Demo server 不會重啟 AGX 手指服務，也不會 reboot 手指馬達。** 它只重新建立本機 server 對 AGX `:5003` 的連線。

### B. 手指卡住但 AGX 服務仍正常：只 reboot 馬達（首選）

適用於某一根手指可讀到位置、但不會動的 Dynamixel 過載 torque-off 鎖存。這會 reboot 馬達、重新開 torque、再張開回 home；**不會**重載 AGX 模型，也不需要重啟 Demo server。

```bash
conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py --reboot-fingers

# 確認 AGX 回報的馬達位置是真實 home 值，而不是 [0,0,0]
curl -s http://192.168.1.100:5003/lstm/status
```

也可在 Web UI 的 Gripper 面板按 **Reboot Fingers**；兩種方式效果相同。

### C. AGX 手指服務失聯／`motor_pos` 是 `[0,0,0]`：重啟 AGX 服務

先確認症狀：

```bash
curl -s http://192.168.1.100:5003/health
curl -s http://192.168.1.100:5003/lstm/status
```

若 health 沒回應，或 status 的 `motor_pos` 是 `[0,0,0]`，在 Ubuntu 筆電執行下列命令。服務重啟後手指會自動 home：

**方式 1：從 Ubuntu 筆電遠端單行重啟（推薦）**

```bash
ssh agx "pkill -f lstm_gripper_service.py"
ssh agx "cd ~/Desktop/GP && nohup $(pipenv --venv)/bin/python lstm_gripper_service.py > lstm_service.log 2>&1 &"
sleep 5
curl -s http://192.168.1.100:5003/health
curl -s http://192.168.1.100:5003/lstm/status
```

**方式 2：SSH 進入 AGX 互動式重啟**

```bash
ssh agx
cd ~/Desktop/GP
pkill -f lstm_gripper_service.py
nohup $(pipenv --venv)/bin/python lstm_gripper_service.py > lstm_service.log 2>&1 &
exit
```


成功條件：`/health` 回傳 `"ok":true`，且 `/lstm/status` 的 `motor_pos` 接近 `[3072,3072,2048]`，不是 `[0,0,0]`。若仍為零，檢查馬達 USB 裝置：

```bash
ssh agx 'ls -l /dev/serial/by-id/ | grep FT6RW7NN'
ssh agx 'tail -40 ~/Desktop/GP/lstm_service.log'
```

> 注意：`pkill -f lstm_gripper_service.py` 只會針對這支服務名稱；請勿在手指仍夾住物件時執行重啟（重啟時手指會自動張開 Home）。

---

## 第 0.6 步 — 設定並執行「固定座標 fallback」（新流程）

三物件的 fallback 現在是**固定座標 + 固定手指位置**，不使用相機、locator、
VLA 或 LSTM。`scripts/fallback_teach.py`、`config/fallback_taught.yaml` 與
`scripts/run_LSTM5_toVLA.py` 只保留給舊工具參考；它們的錄影值不再控制這三個
物件的新 fallback，也不需要再錄一套鍵盤教導。

在 `config/demo_config.yaml` 編輯：

```yaml
fixed_fallback:
  # 共用速度；固定 fallback 不受 auto_run_cli.py --speed 影響
  travel_speed_percent: 75
  descend_speed_percent: 30
  objects:
    trapezoid:
      pick_x_mm: 490.116       # 可改：梯形 X
      pick_y_mm: 177.869       # 可改：梯形 Y
      grasp_z_mm: 170.725      # 已校準常數，通常不要改
      contact_dwell_s: 0.0
      fixed_grasp_position: null  # 改成 [finger1, finger2, finger3]
    board:
      pick_x_mm: 505.0         # 可改：板子 X
      pick_y_mm: 214.0         # 可改：板子 Y
      grasp_z_mm: 128          # 已校準常數，通常不要改
      contact_dwell_s: 1.0
      fixed_grasp_position: null  # 故意為 null；板子不夾、手指保持張開
    butter_knife:
      pick_x_mm: 490.1         # 可改：奶油刀 X
      pick_y_mm: 0.0           # 可改：奶油刀 Y
      grasp_z_mm: 128          # 已校準常數，通常不要改
      contact_dwell_s: 3.0
      fixed_grasp_position: null  # 改成 [finger1, finger2, finger3]
```

把你已實測可抓住物件的三顆馬達位置直接填入梯形與奶油刀的
`fixed_grasp_position`，例如格式是 `[2140, 2150, 1130]`（這只是格式示意，
**不要照抄示意值**）。三個值必須是整數且在馬達安全範圍內。若梯形或奶油刀
仍為 `null`，server 會在**手臂尚未移動前**拒絕並在 log 指出要改的完整 key。

行為固定如下：

- 梯形：到 `(X,Y,170.725)` 後立即送固定抓取位置，實測馬達收斂後才上抬。
- Board：到 `(X,Y,125)` 後依 `contact_dwell_s`（目前 1 秒）等待，手指保持
  張開，然後上抬。
- 奶油刀：到 `(X,Y,126)` 後依 `contact_dwell_s`（目前 3 秒）等待；只送一個
  `fixed_fallback.travel_speed_percent` 的連續上抬到約
  `Z=425`，實測 `Z>=170` 時在手臂仍移動中送固定抓取位置。`Z=170`
  不是停點。
- 三者放開都送 `[3072,3072,2048]`，等實測馬達真正到位並穩定；沒有 8 秒
  「當作成功」的捷徑。STOP 是唯一取消方式。

`fixed_grasp_position` 的收斂只證明馬達到達固定目標，**不等於觸覺證明物件
一定在手上**。任何固定 fallback 都會先確認 AGX 可連線且馬達位置不是
`[0,0,0]`。

設定存檔後，Demo server 可以保持運行；每次按鈕都會重新讀取這份 YAML。
在 Web UI 的 **Fallback P→P** 依序測：

1. `Trapezoid`
2. `Board`
3. `Butter Knife`（目視確認手指是在連續上抬途中開始關閉）

系統不會自動重啟 Demo server 或 AGX。實機測試時手放在 STOP 旁；UI Stop、
Emergency Stop，以及 attached CLI 的 Ctrl-C 都會取消共享 runner 並停止手臂。
正常 auto 流程也會從同一個 `fixed_fallback` 表讀取 X/Y、固定 Z 與放置幾何，
但手指仍使用 `vla_auto.objects.*.lstm` 的 LSTM action。

Normal auto 的 board／奶油刀另有「人工提早放開」目標，位置在同一檔案的
`vla_auto.objects`（梯形故意不設定，維持原本單階段 release）：

```yaml
vla_auto:
  objects:
    board:
      manual_release_position: [2872, 3072, 2048]
    butter_knife:
      manual_release_position: [2872, 3072, 2048]
```

日後若要改第一段放開姿勢，只改這兩個 `manual_release_position`；三個值都必須
是安全範圍內的馬達整數。改完用 Reload Config／`/api/reload_config`，下一個
normal-auto run 生效。這個 key 不控制固定 fallback。

### 執行時速度、ACC/DEC、Arm Monitor 與 dwell

- Normal auto 的 `vla_auto.speed_percent` 只控制 VLA/爪相機 servo 修正；
  `travel_speed_percent` 控制 hover/lift/release travel，
  `descend_speed_percent` 控制 seek/抓取下降/release-down。CLI `--speed` 也只
  覆寫 correction speed。
- 固定 fallback 使用 `fixed_fallback.travel_speed_percent` 與
  `fixed_fallback.descend_speed_percent`。
- `config/modbus_config.yaml -> motion.acceleration_raw/deceleration_raw` 會套用到
  normal auto 與固定 fallback 的每一個 move。修改後不用重啟 server；下一次
  run 開始前會重新套用 motion/safety 設定。
- 每一步送 GO 前，`0x0324` 必須讀回要求的 speed。失敗時會看到
  `speed register rejected command before GO`，該步不會開始移動。
- auto 移動時背景 pose poller 雖暫停，但 move watchdog 會持續送實測
  `arm_pose`，所以 Web UI **Arm Monitor** 的 X/Y/Z 應在途中持續變化。
- normal auto 與固定 fallback 的 dwell 都讀
  `fixed_fallback.objects.<物件>.contact_dwell_s`；log 顯示的秒數就是實際等待。

### UI 面板與 Quick Pick 顯示開關

面板顯示只改 `config/demo_config.yaml`：

```yaml
modules:
  vla: { enabled: true, show_in_ui: false }
  dataset_capture: { enabled: true, show_in_ui: false }
  quick_pick: { enabled: true, show_in_ui: true }
```

- `modules.vla.show_in_ui` 控制 VLA Inference。
- `modules.dataset_capture.show_in_ui` 控制 Dataset Capture／Recording Console。
- `enabled: true` 且 `show_in_ui: false` 代表後端仍可供 CLI 使用，只隱藏 UI。
- `config/demo_config.yaml` 是顯示設定的唯一來源；launch profile 只控制
  `enabled`。

Quick Pick 內部控制位於 `ui.controls.quick_pick`：

```yaml
ui:
  controls:
    quick_pick:
      show_arm_controls: false
      show_reload_config: false
      show_catalog_picker: false
      fixed_fallback_objects:
        - trapezoid
        - board
        - butter_knife
```

以上預設讓 Classic Quick Pick 只留下 Trapezoid、Board、Butter Knife。
Modern UI 也只保留這三個 fixed fallback，隱藏 Object／Method／Run Selected
與動態物件按鈕；Modern System Overview 的 Ready／Home 不受影響。

因為 Classic 的 Reload Config 按鈕也被隱藏，改完 YAML 後在另一個終端執行：

```bash
curl -X POST http://127.0.0.1:8090/api/reload_config
```

這只重新發送 UI 設定，不會重啟或啟停手臂、AGX、gripper、相機或 VLA。
若修改會影響硬體生命週期的 `enabled`，仍需依 server 重啟步驟處理。

---

## 第 1 步 — 離線自檢(不需要任何硬體)

以下硬體無關測試應全部 PASS：

```bash
# (a) 固定 fallback：設定、三物件流程、同步上抬、STOP、UI
conda run --no-capture-output -n voice_pick \
  python -m unittest discover -s tests -p test_fixed_fallback.py -v

# (b) UI module 的 show_in_ui 權限與 classic/modern 套用
conda run --no-capture-output -n voice_pick \
  python -m unittest discover -s tests -p test_ui_module_visibility.py -v

# (c) 既有 knife/LSTM/連續上抬 regression
conda run --no-capture-output -n voice_pick \
  python -m unittest discover -s tests -p test_knife_grasp_diagnostics.py -v

# (d) 抵達判斷+伺服方向,對錄影資料評測 — 3 物件 0 誤判抵達
conda run --no-capture-output -n voice_pick python scripts/arrival_offline_eval.py

# (e) 側相機定位(locator),對錄影資料評測 — 22/22 集,誤差 ≤12mm
conda run --no-capture-output -n voice_pick python scripts/locator_offline_eval.py
```

**看什麼:** (a)–(c) 最後都是 `OK`；(d) 每個物件
`far(>30mm) 0/xx false-arrivals`；(e) 每個物件 `OK (<±100mm needed)`。
**失敗時:** 通常是校準檔被改壞 — 檢查 `data/calibration/claw_servo_calibration.json`
與 `data/calibration/claw_arrival.json` 是否存在且格式正確。

---

## 第 2 步 — AGX 手指服務(2026-07-14 已部署並驗證)

服務檔:AGX 上的 `~/Desktop/GP/lstm_gripper_service.py`(取代舊的
gripper_record_api_v2,同時相容其 API)。**注意:跟隊友的
`run_LSTM5_toVLA.py` 互斥 — 兩個不能同時跑(共用同一串列埠)。**

```bash
# 啟動 / 重啟 (方式 1: 從筆電一鍵重啟)
ssh agx "pkill -f lstm_gripper_service.py"
ssh agx "cd ~/Desktop/GP && nohup $(pipenv --venv)/bin/python lstm_gripper_service.py > lstm_service.log 2>&1 &"

# 啟動 / 重啟 (方式 2: SSH 進 AGX 操作)
# ⚠️ 服務啟動時會自動 fingers.home() → 重啟=手指張開回 home。先確認爪子沒夾東西!
ssh agx
cd ~/Desktop/GP
pkill -f lstm_gripper_service.py
nohup $(pipenv --venv)/bin/python \
    lstm_gripper_service.py > lstm_service.log 2>&1 &
exit
```

# 從筆電逐項測(全部已驗證過一次):


curl http://192.168.1.100:5003/health
#  → {"api_version":"lstm_v1","ok":true,"status":"idle"}

curl http://192.168.1.100:5003/state
#  → current_pos 接近 [3072,3072,2048](home),tactile 有數值

# 預載(梯形/奶油刀:各兩個模型 grasp+release;board:一個模型 release)
curl -X POST http://192.168.1.100:5003/lstm/preload \
     -H 'Content-Type: application/json' -d '{"object":"trapezoid"}'
#  → {"ok":true,"seconds":5.x,...};第二次會快很多(已在 GPU)

# 手指動作測試(只動手指,不動手臂;空抓 3 秒再停)
curl -X POST http://192.168.1.100:5003/lstm/start \
     -H 'Content-Type: application/json' -d '{"action":"trapezoid_grasp"}'
sleep 3
curl http://192.168.1.100:5003/lstm/status     # status:running, step 在增加, tactile 有值
curl -X POST http://192.168.1.100:5003/lstm/stop   # 手指凍結在原位
curl -X POST http://192.168.1.100:5003/command \
     -H 'Content-Type: application/json' -d '{"action":"o"}'   # 張開回 home


**失敗時:**
- `health` 沒回應 → 服務沒起來,`ssh agx 'tail -20 ~/Desktop/GP/lstm_service.log'`。
- `SerialException ... multiple access` → 有別的程式佔用串列埠
  (`ssh agx 'fuser -v /dev/ttyUSB*'`),把隊友的 run_LSTM5 關掉再重啟服務。
- 新 fixed fallback 不呼叫 LSTM，也不會在抓取／放開前偷偷 home；它直接從
  當下實測位置走到 YAML 裡的固定 motor goal。

---

## 第 3 步 — 校準即時驗證(--observe,完全不下任何動作指令)

在手臂旁,先用手動方式(pendant / 之前的 UI)把手臂移到物件上方
「視覺平面」高度(梯形/刀 z≈425,**board 要降到 z≈365 才看得到**),然後:

```bash
conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py \
    --object trapezoid --observe
```

每 0.5 秒印一行:`conf / err(cx,cy) / 判定 / servo_would_move`。

**驗收:**
1. 手臂正對物件正上方時 → 顯示 `ARRIVED`,err 接近 (0,0)。
2. 手臂往 +x 挪 20mm → `servo_would_move` 的 x 應該指回來(負向、~12mm 上限)。
3. y 同理。方向錯 = jacobian 正負號問題,回報給 Claude。
4. board 在 z425 應該是 `no box`(正常!),降到 z365 附近才出現 box。

> **--observe 只是「驗證」,不是「校準」。** 真正的量測校準工具是
> `scripts/servo_calibrate.py`(2026-07-14 已對三個物件跑過,結果存在
> `data/calibration/claw_servo_calibration.json`,是系統的權威來源)。
> 需要**重跑**的情況只有:rig 搬動/爪相機被撞、更換 hover 高度、新增物件,
> 或上面第 2/3 點方向驗證失敗。指令(互動式,你動手臂、它只讀):
>
> ```bash
> conda run --no-capture-output -n voice_pick python scripts/servo_calibrate.py \
>     --object trapezoid    # 換 board / butter_knife 各跑一次
> ```
>
> 流程:已知可抓位置升到 hover 按 Enter → 只挪 x ~20mm 按 Enter → 回基準 →
> 只挪 y → 自動算 2×2 jacobian、存檔、並提供即時殘差驗證迴圈。

---

## 第 4 步 — 模型預測測試(stepthrough,每步都要你按 Enter 才動)

```bash
conda run --no-capture-output -n voice_pick python scripts/vla_stepthrough.py \
    --blind-state --freeze-z --linear \
    --embed data/vla_embed_trapezoid_zh256.pt
```

按鍵:`Enter`=執行這步 / `x+20 y-15`=手動修正 / `r`=重新預測 /
`t`=標記目前位置為真值 / `+`/`-`=調速 / `l`=MovP↔MovL / `s`=停。

**驗收:** 預測的 (x,y) 朝物件方向收斂;z 被 freeze 在 hover。
偏差穩定的話記下平均 dx,dy(結束時會印 `--x-cal` 建議值)。

---

## 第 5 步 — 自動流程「模擬」(--dry,手臂零動作)

跟 UI 的 Auto Run 完全同一條程式路徑,但手臂是模擬的、
閘門用終端機按 Enter 代替按鈕。相機和 GPU 是真的。

```bash
conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py \
    --object trapezoid --dry
```

**看什麼(記錄檔會逐行印):**
- `stage -> approach` → `move vlaN [DRY]` 逐步逼近
- board 會先出現 `vision plane z=365 … will seek down` 和 `move seek_zN`
- 模型收斂後 `move servoN`(伺服修正)→ `ARRIVED`
- `stage -> descend` → `GATE [pick_done]`(按 Enter)→ 抬起 → 放置點
  → `GATE [released]`(按 Enter)→ 回 ready → `COMPLETE`
- dry 模式**不會**呼叫 AGX 手指(設計如此)。

---

## 第 5.5 步 — 收斂測試 --approach-only(VLA + 校準的實機驗收,§1.4)

**這一步就是「模型會不會收斂到物件正上方」的測試**:VLA 預測 + 校準後的
伺服修正一起上、手臂真的動,但**到達(ARRIVED)就停在視覺平面上方** —
不下降、不碰 AGX 手指、不回 ready。手臂停著讓你目視驗收對得準不準。

> 注意:第 5 步 `--dry` 測不了收斂 — 手臂是模擬的,相機看到的畫面不會變,
> 模型永遠覺得自己沒靠近。真正的收斂只能在這步(實機)驗。

```bash
# 梯形:VLA 模式(現有 checkpoint 只訓練過梯形)
conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py \
    --object trapezoid --approach-only --speed 10

# board / 刀:還沒有訓練好的 VLA → 用 locator 模式(cam1 一張照片粗定位,
# 之後的爪相機伺服收斂跟梯形共用同一套校準與程式碼,一樣能驗收斂)
... --object board --mode locator --approach-only --speed 10
... --object butter_knife --mode locator --approach-only --speed 10
#   ↑ 刀的仿射只看過 x≈490 — 刀要放在 x≈490 附近!
```

> 對 board/knife 選 VLA 模式會被 CLI 擋下來警告(checkpoint 沒看過它們,
> 預測是垃圾)。等 day 2 `threeobj` checkpoint 通過離線評測、換上之後,
> 三個物件就都能走 VLA 模式。
> 梯形也可以測 locator 模式(仿射 9 點、誤差 ~10mm)— 順便交叉比較
> VLA 與 locator 的落點準度。

**想要預測「一步到位」?** 改 `config/demo_config.yaml → vla_auto:`:
`stride: 64`(取模型預測的終點,而不是往前 6 小步)+ `max_step_mm: 250`
(放寬單步上限)。安全盒照常把關、到位後伺服照常修正;缺點是中途沒有
重新預測的修正機會 — 先用 `--approach-only` 低速比較兩種設定再決定。

**驗收:**
- 記錄檔逐步印 `move vlaN`(模型大步)→ `move servoN`(伺服小步修正)
  → `APPROACH-ONLY: ARRIVED in N moves at (x,y,z) — final err=(…) conf=…`。
- 手臂最後停的位置,目視應在物件正上方(爪相機串流上 box 貼著設定點)。
- 想量化:在停住的位置直接用 pendant 垂直下降,看會不會正好對到物件。
- board 記得會先 seek 下降到 z365 才開始用視覺。
- 收斂不了/方向跑偏 → 回第 3 步 `--observe` 檢查 jacobian;
  ARRIVED 觸發太鬆/太緊 → 調 `arrival_tol_mm`(預設 12mm)。

三個物件都驗過收斂後,才進第 6 步完整流程。

---

## 第 5.6 步 — 放置點測試 --place-only(只跑放置腿)

單獨驗證放置路線:**跳過接近/抓取**,從目前位置 → 抬到放置接近高度 →
移到放置點(`config/demo_config.yaml -> fixed_fallback.place_approach_pose`:
x=490.1, y=-204.0, z=424.9)→ **確認閘門** → 下降到該物件的放置 z →
release 閘門 → 縮回 → 回 ready。不需要物件、不需要模型;
手指要動的話 AGX 服務開著即可(空手測加 `--no-agx`)。

```bash
conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py \
    --object board --place-only --no-agx --speed 10
# 三個物件各跑一次(放置 z 不同):trapezoid / board / butter_knife
```

**每個物件的放置 z:**

| 物件 | 放置 z (mm) | 來源 |
|---|---|---|
| trapezoid | 180 | `fixed_fallback.objects.trapezoid.release_z_mm` |
| board | 170 | `fixed_fallback.objects.board.release_z_mm` |
| butter_knife | 170 | `fixed_fallback.objects.butter_knife.release_z_mm` |

目前 `vla_auto.confirm_descend: false`，所以 normal auto 不會等放置下降閘門；
第一次驗證可暫時改成 `true`，看到 `about to PLACE-descend ...` 後再按 Enter。
想更新數值就改
`config/demo_config.yaml -> fixed_fallback.objects.<物件>.release_z_mm`；
三物件 runtime 不再讀 `config/objects.yaml` 或 `fallback_taught.yaml` 的放置值。

---

## 第 6 步 — 自動流程「實機」(第一次務必低速)

物件放在教導點,AGX 服務跑著(第 2 步),然後:

```bash
conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py \
    --object trapezoid --speed 10
# 不想動 AGX 手指時加 --no-agx(閘門變純確認)
# board 走 locator 那條路(不用 VLA):--object board --mode locator
```

**每個物件的下降 z(防撞桌,先看這張表):**

| 物件 | 抓取 z (mm) | 來源 | 可信度 |
|---|---|---|---|
| trapezoid | **170.725** | `fixed_fallback` 校準常數 | 設定後實機確認 |
| board | **128** | `fixed_fallback` 校準常數 | 設定後實機確認 |
| butter_knife | **128** | `fixed_fallback` 校準常數 | 設定後實機確認 |

- 這三個 Z 由 `fixed_fallback.objects.<物件>.grasp_z_mm` 統一提供；
  `fallback_teach.py` 與 `fallback_taught.yaml` 不再有優先權。
- 目前 `confirm_descend: false`。第一次驗證 normal auto 若想保留防撞確認，
  先暫時改成 `true`；固定座標 fallback 則依已校準路線直接下降。
- UI 模式同一道閘門是橘色 **DESCEND ✓** 按鈕。
- 全部教導+驗證過後,想要全自動流程再把
  `config/demo_config.yaml → vla_auto.confirm_descend` 改成 `false`。

流程與 dry 相同,但:
- 一開始就自動對 AGX `preload`(手臂靠近的時間把模型載好 — 不會卡)。
- 梯形在抓取高度啟動 LSTM;奶油刀則先在接觸高度 prepare 真實閉合 goal,
  再於連續上抬途中啟用。board 手指維持張開鏟起來,不啟動抓取。
  **奶油刀 2026-07-18 起改用真的 LSTM 觸覺抓取**(`knife_grasp`,見 ADR 0002)。
- **梯形 LSTM**:至少 2 指相對 action start 真閉合 60 ticks，且至少 2/3 指
  追蹤 goal、motor/goal 全部穩定 1.2 秒後 → 凍結 → 手臂上抬。
  接觸卡住的第 3 指不再造成無限等待。正常情況不用按 Enter；real-close
  成立後 CLI 才會顯示 guarded Enter，可接受當下已抓住的姿勢。
- **奶油刀 LSTM(2026-07-18)**:下降到 `grasp_z=128` → 保持接觸 3 秒 →
  在底部呼叫 `/lstm/prepare`:1 秒 baseline+推論可以進行,但**不寫馬達 goal**。
  3 秒結束後若尚未出現真閉合 goal,手臂留在 Z=128 無限等待(STOP 可取消);
  第一個相對實測起點至少 2/3 手指關 60 ticks 的 goal 會被鎖住。接著用
  **同一個不中斷的 move**以 `vla_auto.travel_speed_percent` 朝 `Z=425` 上抬,
  實測 Z 第一次 `>=170` 時同步寫入 cached goal 並恢復 LSTM,所以手指在手臂
  持續上抬時開始關閉。Z=170 不是 waypoint;CLI `--speed 30` 不控制這段 lift。
  抓取沒有 8 秒逾時:實際馬達與 goal 都要相對 action start 至少 2 指關 60 ticks,
  再穩定 1.2 秒才成功。`motor_pos=[0,0,0]` 永遠不算成功;舊 backoff 維持 0。
- **board(lift-under)**:下降到 Z=128 → 依 `contact_dwell_s`（目前 1 秒）接觸等待
  → 自動上抬,不出現
  `gate_pick`、不等 `pick_done`/Enter。此行為由 fixed geometry 的
  `requires_grasp: false` 決定。
- 放置點（board 固定在 Z=170）:自動啟動 release 模型 → goal 與 motor
  相對 release action start 出現真實 excursion → **手指開完(收斂)** →
  LSTM stop/hold → stepped HOME `[3072,3072,2048]` → **量測 HOME 驗證**
  （必要時自動 reboot 清過載鎖）→ 才允許手臂上抬到 Z=425／回 ready。
  若模型判定卡住，board／奶油刀可走下節的兩段 Enter：先到
  `[2872,3072,2048]` 並量測收斂，再由第二次 Enter 回 HOME。兩段期間
  手臂都留在 release Z；沒有 elapsed-time 成功捷徑，只有 STOP 可取消。
  board 的 `PCB_release` 主要動 finger 1，所以使用
  `release_min_excursion_fingers: 1`；其他物件預設 2。
- **一般 LSTM 若沒啟動**:大聲印 `⚠️ AGX LSTM … 沒啟動`。梯形等一般抓取
  不自動夾(`close()` 只是 20-tick 微動,不是真抓),改停在 pick 閘門等你手動完成。
  奶油刀 two-phase prepare 失敗則直接中止,不讓未抓住的刀自動上抬。
  **放開**:仍自動 `set_position([3072,3072,2048], stepped)` 張開回 home,再等你按 released 閘門。
- Attached 與 local CLI 使用和 UI 相同的階段提示：抓取 real-close 後可
  Enter 接受 grip；board／奶油刀 release 依序提示「移到 release pose」與
  「回 HOME」。任何時候 Ctrl-C / STOP = 全停(手臂停、手指凍結)。

**驗收:** 完整 接近→伺服→下降→抓→運→放→回位 一輪;
過程中第二個終端機可開 `arm_speed_audit.py` 順便記錄速度。

---

## 第 7 步 — 最後才是 UI

以上全部通過後,啟動 server,用 VLA 面板的 Auto Run 按鈕跑同一條路:

```bash
conda run -n voice_pick python -m src.launcher start full --host 0.0.0.0 \
    --port 8090 --ui-mode classic --env-config config/env/ubuntu.local.yaml
```

- 面板:物件選單 + VLA/locator 模式 + Auto Run。LSTM 正常時抓取／放開
  自動判定；CLI 的 guarded Enter 只在抓取 real-close 成立後出現。
  `PICK DONE ✓ / RELEASED ✓` 只供沒有 LSTM telemetry 的人工 fallback gate。
- 現場調參(免重啟):conf / tol_mm(公釐容差,經 jacobian 換算)/
  stay / srv,改完按 Apply。
- Teach Mode 面板已移除(config 裡 `modules.teach.show_in_ui: false`)。

### UI：提早完成目前的手指階段

Classic 與 Modern UI 的 **Quick Pick** 區都有同一顆階段式按鈕；平常隱藏，
只有目前動作可安全完成時才出現：

- **Complete Grasp (Enter)**：只有 backend 已確認 goal 與實測馬達至少 2 指
  真閉合 60 ticks 後才出現。按下後保留現在已抓住的姿勢，停止繼續等
  tracking／穩定判定，流程進入下一階段。未達 real-close 時不能強制出現。
- **Complete Release + Open (Enter)**：梯形仍使用此單階段動作。按下後停止
  LSTM release，接著 stepped HOME `[3072,3072,2048]` 並讀回驗證。
- **Move to Release Pose (Enter)**：只在 board／奶油刀的 LSTM
  `放開_settle` 出現。**第一次 Enter** 會停止 LSTM，從當下實測位置以
  stepped `set_position` 移到
  `manual_release_position: [2872, 3072, 2048]`。必須等實測馬達到目標並
  穩定；移動途中按鈕隱藏，手臂保持在 release Z，沒有秒數到就算成功。
- **Return Fingers HOME (Enter)**：只有上一個 release pose 已量測收斂後才出現。
  **第二次 Enter** 才會送 HOME `[3072,3072,2048]` 並讀回量測值驗證；
  **HOME 成功後手臂才會上抬／縮回**。HOME 失敗時 run 報錯並留在 release
  高度。第一次的按鍵事件與第二次使用不同 event，長按／快速重複 Enter
  不會一次跳過兩段。

可直接點按鈕，也可讓頁面取得焦點後按 `Enter`。游標若正在
`input / textarea / select / button`，全域 Enter 會被忽略，避免跟文字送出或
其他按鈕衝突。這顆按鈕不是緊急停止；需要立即停止手臂與凍結手指時仍用
**STOP / Emergency Stop / CLI Ctrl-C**。

---

## 調參速查表 — 「我想改 X,要去哪改?」

固定 fallback 每次按按鈕都重新讀 YAML；`show_in_ui` 可用 Reload Config／
瀏覽器重新整理套用。只有硬體生命週期的 `enabled`、相機或連線設定等才需要
重啟 server。

### 1. 預測步幅 / 一步到位
`config/demo_config.yaml → vla_auto:`
| 想要 | 改什麼 |
|---|---|
| 每次往前走小步(預設,較穩) | `stride: 6` |
| **一步到位**(直接跳到模型預測的終點) | `stride: 64` **並且** `max_step_mm: 250` |
| 放寬/收緊單一步的上限 | `max_step_mm`(mm) |
| 最多走幾步就放棄 | `max_moves`(或 CLI `--max-moves`) |

> ⚠️ `stride` / `max_step_mm` **只管 VLA 預測**。一旦爪相機接手(log 出現 `servo engaged — VLA paused`),移動大小改由**另一顆旋鈕** `servo_max_step_mm` 控制,跟 stride 完全無關 —— 見下面 1b。

### 1b. 爪相機接手(伺服)後的步幅 —— 跟 stride 無關!
**症狀**:VLA 收斂到物件附近後,爪相機每次只移動一點點(「爬行」),按 stride 改多大都沒用。
**原因**:交給爪相機後,每次修正 = `J⁻¹·誤差` 再**夾限到 `servo_max_step_mm`(預設 12mm)**;物件若在 y 方向差 ~160mm,12mm 一步要走 ~13 次,但 `max_corrections`(預設 6)先到就重來 → 就是你看到的爬行。stride 在這裡**零作用**。

| 想要 | 去哪改 | 生效方式 |
|---|---|---|
| **現場即時**加大爪相機步幅 | UI「VLA 面板 → `srv` 框」把 12 調到 20~60,按 Apply | 立即套用到跑動中的伺服(免重啟) |
| 永久/更大 + 更多次修正 | `data/calibration/claw_servo.yaml`:`servo_max_step_mm: 35`、`max_corrections: 10` | **要重啟 server** |
| 完全不用爬行(相信 VLA 一次到位) | 見 §1「一步到位」`stride:64`+`max_step_mm`,並把 `vla_sanity_mm` 設 0(否則 VLA 大步會被 #4 攔下改交伺服) | 重啟 server / 重跑 CLI |

- `srv` 框上限已於 2026-07-15 由 20 放寬到 **60**,所以現在現場可直接調到 60。
- `max_corrections` **不在 UI**,只能改 yaml(要重啟)。
- 根本原因:VLA 太早把棒子交給伺服(#4 sanity gate + 12mm cap)。加大 `servo_max_step_mm` 是治標、對 demo 夠用;治本是別讓 VLA 的大步修正被轉交伺服(即之前提過、你先擱置的「方向性 sanity gate」)。

### 2. 三物件固定座標／高度

- **抓取 X/Y**：`config/demo_config.yaml -> fixed_fallback.objects.<物件>.pick_x_mm / pick_y_mm`。
- **抓取 Z**：同一物件的 `grasp_z_mm`；這是已建立的校準常數，通常不改。
- **固定抓取手指**：梯形／奶油刀的 `fixed_grasp_position: [f1,f2,f3]`。
- **共用放置 hover**：`fixed_fallback.place_approach_pose`。
- **放置高度**：`fixed_fallback.objects.<物件>.release_z_mm`。
- **Normal auto 人工 release 姿勢**：`vla_auto.objects.board.manual_release_position`
  與 `vla_auto.objects.butter_knife.manual_release_position`；目前兩者都是
  `[2872,3072,2048]`。梯形不設此 key。
- Normal auto 的幾何也讀此表，但 VLA／locator 與 LSTM action 仍照 normal auto 設定。
- `fallback_taught.yaml`／`objects.yaml` 只留給 legacy 工具，不控制這三物件 runtime。

### 3. 速度

| 流程／階段 | 旋鈕 |
|---|---|
| Normal auto 的 VLA 大步 + 伺服修正 | `vla_auto.speed_percent`（CLI `--speed` 只蓋這個） |
| Normal auto hover／抬起／移到放置點 | `vla_auto.travel_speed_percent` |
| Normal auto 下降 | `vla_auto.descend_speed_percent` |
| **固定 fallback** hover／上抬／放置移動 | `fixed_fallback.travel_speed_percent`（目前 70） |
| **固定 fallback** 抓取／放置下降 | `fixed_fallback.descend_speed_percent`（目前 30） |

`auto_run_cli.py --speed 30` 不會把固定 fallback 或奶油刀連續上抬改成 30%。

### 4. 閘門開不開
| 閘門 | 開關 |
|---|---|
| 下降前確認(防撞桌) | `vla_auto.confirm_descend: true/false`(全教導驗證過才改 false) |
| Normal auto 的 AGX 手指 | CLI `--no-agx` → 改成人工確認，不碰 LSTM |
| **固定 fallback** | 沒有 pick/released 閘門；固定馬達目標真正收斂才繼續，STOP 可取消 |

### 5. 新行為(2026-07-15 修)
| 項目 | 說明 / 旋鈕 |
|---|---|
| **#1 爪相機常斷** | 根因=要求了**非原生**解析度。相機只支援 1920×1080 MJPG @24/15/10。已把 `cameras.claw.profiles.*` 的 `capture_w/h` 設 1920×1080、`stream_fps` 設原生值。**換相機/換機器要先用 `v4l2-ctl -d <by-id> --list-formats-ext` 確認原生模式再填**;串流/推論仍用小圖(軟體縮放)。 |
| **#2 LSTM 沒啟動** | 一般 grasp start 失敗會大聲警告並停在 pick 閘門等手動抓取;奶油刀 `prepare` 失敗會中止,不自動上抬。**放開**失敗仍用 `set_position([3072,3072,2048], stepped)` 真正張開回 home,再等 released 閘門。 |
| **#3 放開後手指回 home** | release 後 + 回 ready 時,手指用 `/set_position`(stepped,漸進)回 HOME_POS `[3072,3072,2048]` 張開,不再凍結在半途。**不是用 `o` jog** — `o` 會 state_lock+com_lock 卡死且沒有分段,回不準;set_position 走無死鎖的 write_goals。此修不需重啟 AGX 就生效。 |
| **AGX 重啟會回 home** | `lstm_gripper_service.py` 啟動時會 `fingers.home()`,重啟服務=手指回 home 張開(需重新部署+重啟才生效,見第 2 步)。 |
| **#4 VLA 亂跳用 YOLO** | 爪相機已看到物件(conf 夠、畫面新)但 VLA 想跳 > `vla_auto.vla_sanity_mm`(預設 40mm)時,**不信模型**,直接交給 YOLO 爪相機伺服收斂。設 0 關閉。 |
| **奶油刀 fixed-xy** | `vla_auto.objects.butter_knife.use_recorded_xy: true` → normal auto 的刀不做 x,y 接近/伺服，直接使用 `fixed_fallback.objects.butter_knife.pick_x_mm/pick_y_mm` 再盲降。CLI `--xy X Y` 仍可做單次 normal-auto override。 |
| **奶油刀 prepare + Z=170 同步抓/抬 + trace** | Normal auto：Z=128 先 `/lstm/prepare`，依 `contact_dwell_s` 等待後，以 `grasp_prepare_min_close_ticks: 20`（至少 2 指）確認「有意開始閉合」；這只解除暫停，不算抓取成功。log 應顯示 `thresholds activation=20 ticks/2 fingers, final=60 ticks/2 fingers`，接著離開 `grasp_goal_wait`，以 `vla_auto.travel_speed_percent` 做單一 Z=128→425 move，實測 Z>=170 同步 `/lstm/activate`，手臂不停。最後仍由 `grasp_min_close_ticks: 60`（至少 2 指）證明真抓取。固定 fallback 則使用 `fixed_fallback.travel_speed_percent` 並在相同連續上抬時送 `fixed_grasp_position`，完全不呼叫 LSTM。 |
| **LSTM 啟動不再先張手(2026-07-18)** | Normal auto release 從**當下夾住的位置**取 baseline，不先 `home()`；固定 fallback 更不呼叫 LSTM/home，直接走固定目標。連續 lift 使用各自的 `travel_speed_percent`，不使用 CLI `--speed 30`。 |
| **CLI 不關 web(看得到畫面)** | server 開著時直接跑 `auto_run_cli.py` 會**自動 attach** 到 server:相機/手臂還是 server 在管、網頁保持開著能看爪相機畫面,你照樣在終端機下指令、按閘門(UI 按鈕也能按)。要強制自己開相機用 `--local`(需先關 server)。 |
| **真抓取免確認、無逾時** | 抓取沒有 8 秒 fallback;STOP 是唯一取消。motor feedback 與 goal 都要相對 action start 至少 `grasp_min_close_ticks: 60`、達 `grasp_min_close_fingers: 2`,再用 `grasp_settle_ticks: 12` / `grasp_settle_hold_s: 1.2` 判定收斂。`motor_pos=[0,0,0]` 永遠不算成功。board 不用此判定:Z=128 依 `contact_dwell_s`（目前 1 秒）接觸等待後依 `auto_lift_after_dwell: true` 自動上抬。 |
| **梯形 2/3 tracking + guarded Enter** | `vla_auto.grasp_min_tracking_fingers: 2`：至少 2 指在 `finger_settle_tracking_ticks: 30` 內即可自動完成，但真閉合仍需 goal+motor 至少 2 指各 60 ticks，且全部 motor/goal 都要穩定。CLI Enter 與 UI 的 **Complete Grasp** 都只在此 real-close 成立後可接受當下 grip。 |
| **AGX 馬達埠重編號(夾爪不動的根因)** | 現象:`/lstm/status` 的 `motor_pos` 是 `[0,0,0]` 但 `tactile` 正常。原因:FTDI 馬達控制器 `ttyUSB` 重編號(如 1→2),跑動中的服務握著**失效的檔案 handle**。**修法=重啟 AGX 服務**(用 by-id 重開現在的節點,啟動時 `fingers.home()` 驗證):見「常見問題速查」最後一列。已加自動重開(dead read → `_reopen_motor`)。 |
| **放開也免確認、但必須真張開** | Normal auto 會先等 goal+motor 相對 action start 移動 `release_min_excursion_ticks: 60` 再判斷自動收斂；board 因 `PCB_release` 主要動 finger 1，物件 override 是 `release_min_excursion_fingers: 1`，其他預設 2。若判定卡住，board／奶油刀第一次 Enter 停 LSTM 並量測收斂到 `manual_release_position: [2872,3072,2048]`，第二次 Enter 才 stepped HOME 並量測驗證；梯形仍用單階段 **Complete Release + Open**。HOME 成功才讓手臂從 release Z 上抬。固定 fallback 直接走 `[3072,3072,2048]`，全部都沒有 8 秒成功捷徑。 |
| **tactile 異常大值／單位數** | AGX 只發布完整換行 frame，且三值都必須在預設 `0..4095`；`47764` 與截斷的 `1/14/334` 不再送入 LSTM。若感測器韌體量程真的不同，設定 AGX 環境變數 `GRIPPER_TACTILE_MIN`／`GRIPPER_TACTILE_MAX` 後重啟服務。 |
| **UI「Ready」鈕連帶手指回 home(2026-07-15)** | 按 Ready(`arm_ready`)→ 手臂回 ready **並**把手指 `set_position([3072,3072,2048], stepped)` 張開回 home。若正在跑 LSTM 動作會 409 略過(先停 run)。 |
| **單顆手指馬達過載鎖死(reads OK 但不動)** | 現象:`current_pos` 某顆卡在非 home 值(如 finger2=2300),其餘正常,server 一直寫 goal 也不動,log 無錯。這是 Dynamixel **單顆過載 torque-off 鎖存**(抓取夾太緊觸發),**不是** ttyUSB 掉線(那會 `[0,0,0]`)。**免整個重啟的三種復原**(2026-07-15,`/motor/reboot` reboot+torque+home 清鎖存):(1) **自動**:`_background` 讀 reg 70 硬體錯誤,閒置時(非抓取中)自動 reboot 該顆清鎖存(不 home、不亂動);(2) **UI**:Gripper 面板「Reboot Fingers」鈕;(3) **CLI**:`auto_run_cli.py --reboot-fingers`(直連 AGX,server 開關都能用)。仍不動/紅燈=真硬體壞。**別對卡死的手指硬送 goal**,可能傷馬達。根因:LSTM 抓取夾到高 tactile 把手指頂在物件上 stall → 過載;要根治可設馬達 current limit(addr 38)限扭力(會影響夾持力,demo 後再調)。 |

---

## 常見問題速查

| 症狀 | 原因/處理 |
|---|---|
| 相機起不來 | server 還開著佔用 → kill 確切 PID |
| CUDA unknown error(休眠後) | `sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm` |
| 爪相機**常常**斷 | 多半是**要求了非原生模式**(已修:capture 1920×1080@15)。若換相機又開始斷,先 `v4l2-ctl -d /dev/v4l/by-id/<claw> --list-formats-ext` 看原生模式,再改 `cameras.claw.profiles.*.capture_w/h/stream_fps`。狀態列的 `mode_warning` 會直接告訴你協商到的模式對不對。 |
| 爪相機**偶爾**斷一下 | 真的 USB 斷線(巢狀 hub),看門狗 ~4 秒自動復活;server 模式可用 Restart 鈕;demo 前建議直插不要串 hub。 |
| 手臂連不上 | `ping 192.168.1.232`;檢查 `systemctl --user status arm-route-autofix.timer` |
| AGX 手指 500 錯誤 | 串列埠被佔用(run_LSTM5 沒關)→ 關掉重啟服務 |
| **夾爪不動 / 抓取「像教導」而非 LSTM** | 先 `curl -s 192.168.1.100:5003/lstm/status`。若 `motor_pos=[0,0,0]` 但 `tactile` 正常 → **FTDI 馬達埠 ttyUSB 重編號**,服務握著失效 handle。**重啟 AGX 服務**(用 by-id 重開新節點,啟動會 `fingers.home()` 驗證):<br>`ssh agx 'kill <確切PID>; sleep 1; cd ~/Desktop/GP && nohup ~/.local/share/virtualenvs/GP-Qpf5NN6o/bin/python lstm_gripper_service.py > lstm_service.log 2>&1 & sleep 6; curl -s localhost:5003/lstm/status'`<br>確認回傳 `motor_pos` 是接近 `[3072,3072,2048]` 的真值,不是 0。用 `ssh agx 'ls -l /dev/serial/by-id/ \| grep FT6RW7NN'` 可看馬達現在對到哪個 ttyUSB。 |
| 抓取「像教導」但 `motor_pos` 正常 | log 有 `⚠️ AGX LSTM 沒有啟動 … 改用硬編 gripper` → LSTM 沒起來,走了硬編 `close()`(看起來就像教導)。查 `:5003` 是否 `/lstm/*` 服務(`/health` 應回 `api_version:lstm_v1`);別讓 client 連到舊的 `:5002` record_api。 |
| ARRIVED 一直不觸發 | UI/config 調大 `arrival_tol_mm`(預設 12mm);或 conf 光線問題調低 conf_floor |
| 伺服方向亂跑 | jacobian 正負號 → 用第 3 步 --observe 驗證後回報 |
| 換場地(rig 移動) | 側相機仿射全失效 → 跑 §3.7 重校(見英文手冊);爪相機設定點不受影響 |

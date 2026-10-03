# -*- coding: utf-8 -*-
import sys
import numpy as np
try:
    import numpy._core
except ModuleNotFoundError:
    from types import ModuleType
    # 建立偽裝的 _core 模組並塞進系統環境中
    _core = ModuleType('numpy._core')
    sys.modules['numpy._core'] = _core
    # 補齊 joblib 最常呼叫的 multiarray 核心對應
    import numpy.core.multiarray as _multiarray
    sys.modules['numpy._core.multiarray'] = _multiarray
    _core.multiarray = _multiarray
import threading
from dynamixel_sdk import *
from time import sleep, time
import serial
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import joblib
# ==========================================
# 1. PyTorch 模型定義 (與 training.py 完全一致)
# ==========================================
class GripperControllerLSTM(nn.Module):
    # 💡 input_size 修改為 15
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, output_size=3):
        super(GripperControllerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        output = self.fc(out)
        return output
# ==========================================
# 2. 硬體與連線參數設定
# ==========================================
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_TELEMETRY_START  = 124  
LEN_TELEMETRY  = 8           # 2 (PWM) + 2 (Current) + 4 (Velocity)
BAUDRATE_motor = 57600
DEVICENAME_motor = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RWC9P-if00-port0'
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_IDS = [1, 2, 3]

# 初始化馬達
portHandler   = PortHandler(DEVICENAME_motor)
packetHandler = PacketHandler(2.0)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE_motor)




for i in DXL_IDS:
    packetHandler.reboot(portHandler, i)
sleep(1)

group_read = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)

group_telemetry = GroupSyncRead(portHandler, packetHandler, ADDR_TELEMETRY_START, LEN_TELEMETRY)

for dxl_id in DXL_IDS:
    group_read.addParam(dxl_id)    
    group_telemetry.addParam(dxl_id) # 註冊感測讀取 ID

for i in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
# 初始化觸覺感測器
BAUDRATE_sensor   = 115200
DEVICENAME_sensor = '/dev/ttyUSB0' 
try:
    ser = serial.Serial(DEVICENAME_sensor, BAUDRATE_sensor, timeout=1)
    sleep(2)
except Exception as e:
    print(f" 無法連接感測器: {e}")
    ser = None
# ==========================================
# 3. 全域變數與背景執行緒
# ==========================================
com_lock   = threading.Lock()
state_lock = threading.Lock()
running = True
# 馬達安全極限 (防止過度撞擊或絞死)
MOTOR_LIMITS = [
    (1872, 4272),  # 指爪 1 範圍
    (1872, 4272),  # 指爪 2 範圍
    (848, 3248)    # 指爪 3 範圍
]
first_val, second_val, third_val = 0, 0, 0 
latest_motor_pos = [3072, 3072, 2048]
# 💡 新增：全域馬達感測參數
latest_motor_pwm = [0.0, 0.0, 0.0]
latest_motor_cur = [0.0, 0.0, 0.0]
latest_motor_vel = [0.0, 0.0, 0.0]
# 讀取感測器與馬達的背景執行緒
def background_worker():
    global running, first_val, second_val, third_val
    global latest_motor_pos, latest_motor_pwm, latest_motor_cur, latest_motor_vel
    
    while running:
        # --- 1. 讀取觸覺感測器 ---
        if ser and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split()
                if len(parts) >= 3:
                     with state_lock:
                        first_val = int(parts[0])
                        second_val = int(parts[1])
                        third_val = int(parts[2])
            except (ValueError, IndexError):
                pass
        
        # --- 2. 讀取馬達當前位置與感測數據 ---
        with com_lock:
            pos_comm_result = group_read.txRxPacket()
            tel_comm_result = group_telemetry.txRxPacket()
            
            cur_pos = []
            cur_pwm = []
            cur_cur = []
            cur_vel = []
            
            for dxl_id in DXL_IDS:
                # 讀取位置
                p = group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                if p > 0x7FFFFFFF:
                    p -= 0x100000000
                cur_pos.append(p)
                
                # 讀取感測數據並轉換
                raw_pwm = group_telemetry.getData(dxl_id, 124, 2)
                raw_cur = group_telemetry.getData(dxl_id, 126, 2)
                raw_vel = group_telemetry.getData(dxl_id, 128, 4)
                
                if raw_pwm > 32767: raw_pwm -= 65536
                if raw_cur > 32767: raw_cur -= 65536
                if raw_vel > 2147483647: raw_vel -= 4294967296
                
                pwm_val = round(raw_pwm * 0.113, 2)
                cur_val = round(raw_cur * 1.0, 1)
                vel_val = round(raw_vel * 0.229, 2)
                
                cur_pwm.append(pwm_val)
                cur_cur.append(cur_val)
                cur_vel.append(vel_val)
                
        with state_lock:
            latest_motor_pos = list(cur_pos)
            latest_motor_pwm = list(cur_pwm)
            latest_motor_cur = list(cur_cur)
            latest_motor_vel = list(cur_vel)
        
        sleep(0.01) # 100Hz 高頻更新
bg_thread = threading.Thread(target=background_worker, daemon=True)
bg_thread.start()
# ==========================================
# 4. 載入模型與標準化 Scalers
# ==========================================
# 💡 選擇模型模式，防止 Grasp 與 Release 混用
ACTION_TYPE = input("請選擇運行模式 (1: grasp, 2: release): ").strip()
ACTION_TYPE = 'grasp' if ACTION_TYPE == '1' else 'release'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🧠 使用的計算設備: {device} | 模式: {ACTION_TYPE}")
try:
    scaler_x = joblib.load(f'scaler_x_trapezoid_{ACTION_TYPE}.pkl')
    scaler_y = joblib.load(f'scaler_y_trapezoid_{ACTION_TYPE}.pkl')
    print(f"✅ 成功載入標準化 Scaler ({ACTION_TYPE})")
except Exception as e:
    print(f"❌ 無法載入 Scaler 檔案: {e}")
    quit()
# 💡 初始化模型時帶入 input_size=15
model = GripperControllerLSTM(input_size=15, hidden_size=64, num_layers=2, output_size=3).to(device)
try:
    model.load_state_dict(torch.load(f'LSTM_trapezoid_{ACTION_TYPE}.pth', map_location=device))
    model.eval()
    print(f"✅ 成功載入 LSTM 模型權重 ({ACTION_TYPE})")
except Exception as e:
    print(f"❌ 無法載入模型權重: {e}")
    quit()
# 💡 設定對應動作的起始歸位位置
if ACTION_TYPE == 'grasp':
    origin_pos = [3072, 3072, 2048] # 與 grasp 資料集起始值相同
else:
    current_positions = []
    for dxl_id in DXL_IDS:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, dxl_id, ADDR_PRESENT_POSITION
        )
        
        # 檢查通訊是否成功
        if dxl_comm_result != COMM_SUCCESS:
            print(f"❌ 馬達 {dxl_id} 讀取失敗: {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            # 處理 32 位元有號整數轉換（補數處理）
            if dxl_present_position > 0x7FFFFFFF:
                dxl_present_position -= 0x100000000
        
        current_positions.append(dxl_present_position)
        origin_pos = [current_positions[0], current_positions[1], current_positions[2]]
# ==========================================
# 5. 主控制流程
# ==========================================
def main():
    global running
    
    # 步驟 B: 安全確認提示
    print("\n==============================================")
    print(f"提示：自動控制即將開始 [{ACTION_TYPE}]，請確認測試物體已放置妥當。")
    print("==============================================")
    input("請按 [Enter] 鍵以啟動 LSTM 測試動作...")
    
    # 步驟 A: 將指爪移到起始位置歸位
    print("\n正在將指爪移回起始點...")
    with com_lock:
        for idx, dxl_id in enumerate(DXL_IDS):
            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, origin_pos[idx])
            sleep(0.5)
    sleep(1.0)
    print("歸位完成。")
    
    TOTAL_STEPS = 60        
    # 注意：因為中間加入了 input() 手動輸入會卡住等待，此間隔僅作為無輸入或快速跳過時的緩衝
    SLEEP_INTERVAL = 0.4      
    
    # 維持原本設計：讓序列長度隨著循環步數從 1 開始不斷增長，不進行滑動裁剪
    seq_history = []
    
    # 獲取初始位置與壓力值
    with state_lock:
        prev_val = [first_val, second_val, third_val]
        # 建立一個手動控制的絕對位置追蹤器，初始值等於歸位位置
        manual_target_pos = list(origin_pos)
        
    print("\n🚀 測試與推理控制啟動！")
    print("💡 提示：在每個 Step，你可以手動輸入馬達位置增量（例如 '50' 代表三軸同時前進 50，'-30' 代表後退）")
    print("💡 或是輸入三個數字（例如 '50 -20 0'）分別控制三軸。直接按 [Enter] 則不手動干預（馬達停在原地）。")
    print("--------------------------------------------------------------------------------")
    
    for step in range(1, TOTAL_STEPS + 1):
        # 1. 取得最新絕對狀態
        with state_lock:
            curr_pos = list(latest_motor_pos)
            curr_pwm = list(latest_motor_pwm)
            curr_cur = list(latest_motor_cur)
            curr_vel = list(latest_motor_vel)
            curr_val = [first_val, second_val, third_val]
            
        # 2. 計算差值 (t 時刻的特徵)
        val_diff = [curr_val[i] - prev_val[i] for i in range(3)]
        
        # 3. 組合特徵：按順序拼接成 15 維向量
        feature_vector = (
            val_diff + 
            [curr_pos[0], curr_pwm[0], curr_cur[0], curr_vel[0]] +
            [curr_pos[1], curr_pwm[1], curr_cur[1], curr_vel[1]] +
            [curr_pos[2], curr_pwm[2], curr_cur[2], curr_vel[2]]
        )
        
        # 4. 標準化特徵並加入歷史緩衝區（長度隨 step 自然增長：1, 2, 3 ... 直到 60）
        feature_scaled = scaler_x.transform([feature_vector])[0]
        seq_history.append(feature_scaled)
        
        # 5. 準備 PyTorch 輸入
        inputs_tensor = torch.from_numpy(np.array([seq_history], dtype=np.float32)).to(device)
        lengths_tensor = torch.tensor([len(seq_history)], dtype=torch.long)
        
        # 6. 模型推理
        with torch.no_grad():
            output_tensor = model(inputs_tensor, lengths_tensor)
            # 取得當前長度序列的最後一個時間步輸出
            pred_scaled = output_tensor[0, -1, :].cpu().numpy()
            
        # 7. 反標準化預測結果
        pred_pos = scaler_y.inverse_transform([pred_scaled])[0]
        
        # 8. 手動介入控制（強制改變馬達位置，藉此改變下一時刻的輸入特徵）
        user_input = input(f"Step [{step:03d}/{TOTAL_STEPS}] 控制增量 (Enter/值): ").strip()
        
        if user_input:
            try:
                parts = user_input.split()
                if len(parts) == 1:
                    # 如果只輸入一個數字，三軸同步增減
                    inc = int(parts[0])
                    increments = [inc, inc, inc]
                elif len(parts) == 3:
                    # 如果輸入三個數字，分別對應三軸
                    increments = [int(parts[0]), int(parts[1]), int(parts[2])]
                else:
                    increments = [0, 0, 0]
                    print("⚠️ 輸入格式錯誤，本次不前進。")
                
                # 更新手動目標位置
                for i in range(3):
                    manual_target_pos[i] += increments[i]
            except ValueError:
                print("⚠️ 請輸入有效的整數！")
        else:
            # 如果使用者直接按 Enter：馬達停在當前位置，單純觀察模型在該狀態下的預測輸出
            pass

        # 9. 計算最終目標絕對位置並限制在安全範圍內
        target_pos = []
        for i in range(3):
            clipped_target = max(MOTOR_LIMITS[i][0], min(manual_target_pos[i], MOTOR_LIMITS[i][1]))
            target_pos.append(clipped_target)
            manual_target_pos[i] = clipped_target # 確保追蹤器不會超出物理極限
            
        # 10. 驅動馬達運動（此時馬達完全聽從你的手動命令增量）
        with com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, target_pos[idx])
                
        # 11. 顯示當前運行數據與 AI 的預測值
        print(f" -> 實際下達目標: {target_pos}")
        print(f" -> 觸覺壓力狀態: {curr_val} (差值: {val_diff})")
        print(f" -> 🧠 AI 預測位置: {[round(float(x), 1) for x in pred_pos]}")
        print("-" * 50)
        
        # 12. 更新歷史數據
        prev_val = list(curr_val)
        
        sleep(0.05) # 給予硬體與通訊短暫緩衝
        
    print("\n⏹️ 測試序列已完成！")
    running = False
    print("👋 程式結束。")
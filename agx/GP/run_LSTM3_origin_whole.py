# -*- coding: utf-8 -*-
import sys
import numpy as np
try:
    import numpy._core
except ModuleNotFoundError:
    from types import ModuleType
    _core = ModuleType('numpy._core')
    sys.modules['numpy._core'] = _core
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
import os

class GripperControllerLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=3):
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
    
ADDR_SHUTDOWN = 63  # 新增此行
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_TELEMETRY_START  = 124  
LEN_TELEMETRY  = 8 
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84          
BAUDRATE_motor = 57600
DEVICENAME_motor =  os.getenv(
    "GRIPPER_MOTOR_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7NN-if00-port0",
)

#DEVICENAME_motor = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RWC9P-if00-port0'
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_IDS = [1, 2, 3]

portHandler   = PortHandler(DEVICENAME_motor)
packetHandler = PacketHandler(2.0)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE_motor)
#----關閉過載保護---------------
for dxl_id in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    sleep(0.05)

    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_SHUTDOWN, 21)
    sleep(0.05)
#------------------------------
for i in DXL_IDS:
    packetHandler.reboot(portHandler, i)
sleep(1)

group_read = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)

group_telemetry = GroupSyncRead(portHandler, packetHandler, ADDR_TELEMETRY_START, LEN_TELEMETRY)

for dxl_id in DXL_IDS:
    group_read.addParam(dxl_id)    
    group_telemetry.addParam(dxl_id) # 註冊感測讀取 ID

for i in DXL_IDS:
    # 1. 調高 P 增益
    packetHandler.write2ByteTxRx(portHandler, i, ADDR_POSITION_P_GAIN, 200)
    
    # 2. 啟用 I 增益
    packetHandler.write2ByteTxRx(portHandler, i, ADDR_POSITION_I_GAIN, 10)

for i in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

BAUDRATE_sensor   = 115200
DEVICENAME_sensor = '/dev/ttyUSB0' 
try:
    ser = serial.Serial(DEVICENAME_sensor, BAUDRATE_sensor, timeout=1)
    sleep(2)
except Exception as e:
    print(f" 無法連接感測器: {e}")
    ser = None

com_lock   = threading.Lock()
state_lock = threading.Lock()
running = True
MOTOR_LIMITS = [
    (1872, 4272),  
    (1872, 4272),  
    (848, 3248)    
]

first_val, second_val, third_val = 0, 0, 0 
latest_motor_pos = [3030, 3030, 2030]

def background_worker():
    global running, first_val, second_val, third_val
    global latest_motor_pos
    
    while running:
        # --- 觸覺感測器讀取：清空緩衝區並只解析最新一筆完整資料 ---
        if ser and ser.in_waiting > 0:
            try:
                # 一次性讀取當前緩衝區中的所有資料以清空積壓
                raw_data = ser.read(ser.in_waiting)
                # 以換行符號分割，並從後往前尋找最新的一筆完整資料
                lines = raw_data.decode('utf-8', errors='ignore').split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                val1 = int(parts[0])
                                val2 = int(parts[1])
                                val3 = int(parts[2])
                                with state_lock:
                                    first_val = val1
                                    second_val = val2
                                    third_val = val3
                                break  # 成功取得最新一筆，結束搜尋
                            except ValueError:
                                continue  # 若數值解析失敗，繼續往前尋找
            except Exception:
                pass
        
        with com_lock:
            pos_comm_result = group_read.txRxPacket()
            tel_comm_result = group_telemetry.txRxPacket()
            
            cur_pos = []
            
            
            for dxl_id in DXL_IDS:
                p = group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                if p > 0x7FFFFFFF:
                    p -= 0x100000000
                cur_pos.append(p)
                         
        with state_lock:
            latest_motor_pos = list(cur_pos)
        
        sleep(0.01) 
bg_thread = threading.Thread(target=background_worker, daemon=True)
bg_thread.start()

ACTION_CHOICE = input("請選擇運行模式 (1: grasp, 2: release, 3: homing): ").strip()
if ACTION_CHOICE == '3':
    ACTION_TYPE = 'homing'
    origin_pos = [3030, 3030, 2030]
    print(f"模式: {ACTION_TYPE} | 歸位目標: {origin_pos}")
else:
    ACTION_TYPE = 'grasp' if ACTION_CHOICE == '1' else 'release'
    
    OBJ_CHOICE = input("請選擇物件 (1: tweezers, 2: trapezoid): ").strip()
    OBJECT_TYPE = 'tweezers' if OBJ_CHOICE == '1' else 'trapezoid'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的計算設備: {device} | 物件: {OBJECT_TYPE} | 模式: {ACTION_TYPE}")
    try:
        scaler_x = joblib.load(f'scaler_x_{OBJECT_TYPE}_{ACTION_TYPE}.pkl')
        scaler_y = joblib.load(f'scaler_y_{OBJECT_TYPE}_{ACTION_TYPE}.pkl')
        print(f"成功載入標準化 Scaler ({OBJECT_TYPE} - {ACTION_TYPE})")
    except Exception as e:
        print(f"無法載入 Scaler 檔案: {e}")
        quit()


    model = GripperControllerLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=3).to(device)
    try:
        model.load_state_dict(torch.load(f'LSTM_{OBJECT_TYPE}_{ACTION_TYPE}.pth', map_location=device))
        model.eval()
        print(f"成功載入 LSTM 模型權重 ({OBJECT_TYPE} - {ACTION_TYPE})")
    except Exception as e:
        print(f"無法載入模型權重: {e}")
        quit()

    if ACTION_TYPE == 'grasp':
        origin_pos = [3030, 3030, 2030] 
    else:
        
        current_positions = []
        with com_lock:
            for dxl_id in DXL_IDS:
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                    portHandler, dxl_id, ADDR_PRESENT_POSITION
                )
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"❌ 馬達 {dxl_id} 讀取失敗: {packetHandler.getTxRxResult(dxl_comm_result)}")
                else:
                    if dxl_present_position > 0x7FFFFFFF:
                        dxl_present_position -= 0x100000000
                
                current_positions.append(dxl_present_position)
            
        # 💡 修正原本在 for 迴圈內缩排錯誤導致 list index out of range 的 bug
        origin_pos = [current_positions[0], current_positions[1], current_positions[2]]



def main():
    global running
    
    if ACTION_TYPE == 'homing':
        print("\n==============================================")
        print(f"提示：歸位動作即將開始，指爪將移回 {origin_pos}")
        print("==============================================")
        input("請按 [Enter] 鍵以啟動歸位動作...")
        print("\n正在將指爪移回歸位點...")
        with com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, origin_pos[idx])
                sleep(1)
        sleep(1.0)
        print("歸位完成。")
        running = False
        print("👋 程式結束。")
        return

    print("\n==============================================")
    print(f"提示：自動控制即將開始 [{ACTION_TYPE}]，請確認測試物體已放置妥當。")
    print("==============================================")
    input("請按 [Enter] 鍵以啟動 LSTM 自主動作...")
    print("\n正在將指爪移回起始點...")
    with com_lock:
        for idx, dxl_id in enumerate(DXL_IDS):
            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, origin_pos[idx])
            sleep(1)
    sleep(1.0)
    print("歸位完成。")
    TOTAL_STEPS = 100       
    SLEEP_INTERVAL = 0.05
    
    seq_history = []

    # # 獲取初始壓力值，用於計算第一個 step 的差值 (curr_val - prev_val)
    # with state_lock:
    #     prev_val = [first_val, second_val, third_val]
        
    print("\n自動控制啟動！")
    
    for step in range(1, TOTAL_STEPS + 1):
        with state_lock:
            curr_pos = list(latest_motor_pos)
            curr_val = [first_val, second_val, third_val]
            
        # 💡 符合 training2 格式：使用上一次讀到的數據與最新讀到的值做差值處理 (val_diff
        feature_vector = (curr_val + [curr_pos[0]] +[curr_pos[1]] +[curr_pos[2]])
        
        feature_scaled = scaler_x.transform([feature_vector])[0]
        seq_history.append(feature_scaled)
        
        inputs_tensor = torch.from_numpy(np.array([seq_history], dtype=np.float32)).to(device)
        lengths_tensor = torch.tensor([len(seq_history)], dtype=torch.long)
        
        with torch.no_grad():
            output_tensor = model(inputs_tensor, lengths_tensor)
            pred_scaled = output_tensor[0, -1, :].cpu().numpy()
            
        pred_pos = scaler_y.inverse_transform([pred_scaled])[0]
        
        target_pos = []
        for i in range(3):
            raw_target = int(pred_pos[i])
            clipped_target = max(MOTOR_LIMITS[i][0], min(raw_target, MOTOR_LIMITS[i][1]))
            target_pos.append(clipped_target)
            
        with com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, target_pos[idx])
                
        print(f"Step [{step:03d}/{TOTAL_STEPS}] | "
              f"觸覺壓力: {curr_val} | "
              f"當前位置: {curr_pos} | "
              f"模型預測pos: {[round(float(x), 1) for x in target_pos]}")
        
        # # 更新上一步的壓力值，用於下一步差值計算
        # prev_val = list(curr_val)
        
        sleep(SLEEP_INTERVAL)
        
    print("\n自動控制序列已完成！")
    
    running = False
    print("👋 程式結束。")
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        running = False
        print("\n偵測到中斷，正在關閉系統...")

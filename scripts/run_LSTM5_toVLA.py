# -*- coding: utf-8 -*-
import os
GP_ROOT = os.environ.get("GP_ROOT", os.path.dirname(os.path.abspath(__file__)))
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
import keyboard

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
    
ADDR_SHUTDOWN = 63  
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_TELEMETRY_START  = 124  
ADDR_TELEMETRY_LEN  = 8 
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84  
ADDR_POSITION_D_GAIN = 80     
BAUDRATE_motor = 57600
DEVICENAME_motor =  os.getenv(
    "GRIPPER_MOTOR_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7NN-if00-port0",
)

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
group_telemetry = GroupSyncRead(portHandler, packetHandler, ADDR_TELEMETRY_START, ADDR_TELEMETRY_LEN)

for dxl_id in DXL_IDS:
    group_read.addParam(dxl_id)    
    group_telemetry.addParam(dxl_id) 

for i in DXL_IDS:
    packetHandler.write2ByteTxRx(portHandler, i, ADDR_POSITION_P_GAIN, 300)
    packetHandler.write2ByteTxRx(portHandler, i, ADDR_POSITION_I_GAIN, 40)
    packetHandler.write2ByteTxRx(portHandler, i, ADDR_POSITION_D_GAIN, 20)

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
        if ser and ser.in_waiting > 0:
            try:
                raw_data = ser.read(ser.in_waiting)
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
                                break  
                            except ValueError:
                                continue  
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

def main():
    global running
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"當前使用設備: {device}")
    
    while True:
        print("\ncontrol start!")
        
        print("==============================================")
        choice = input("(1-5): ").strip()
        
        if choice == '1':
            ACTION_TYPE = 'trapezoid_grasp'
        elif choice == '2':
            ACTION_TYPE = 'trapezoid_release'
        elif choice == '3': 
            ACTION_TYPE = 'knife_release'   
        elif choice == '4':
            ACTION_TYPE = 'PCB_release'
        elif choice == '5':
            ACTION_TYPE = 'home'
        
        origin_pos = [3072, 3072, 2048]
        print(f"模式: {ACTION_TYPE} | 歸位目標: {origin_pos}")
        print("\n正在將指爪移回歸位點...")
        with com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, origin_pos[idx])
                sleep(1)
        sleep(1.0)
        print("歸位完成。")
        
    
        print(f"\n載入模型權重與 Scaler... 動作: {ACTION_TYPE}")
        try:
            if ACTION_TYPE == 'trapezoid_grasp':
                scaler_x = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/trapezoid_grasp/hidden64_ly3/LSTM_scaler_x_trapezoid_grasp_hidden64_ly3.pkl')
                scaler_y = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/trapezoid_grasp/hidden64_ly3/LSTM_scaler_y_trapezoid_grasp_hidden64_ly3.pkl')
                model_path = f"{GP_ROOT}/training_pack/model_pack/LSTM/trapezoid_grasp/hidden64_ly3/LSTM_trapezoid_grasp_hidden64_ly3.pth"
                MOVE_TYPE = 'grasp'
                hidden_size = 64
                num_layers = 3
            elif ACTION_TYPE == 'trapezoid_release':
                scaler_x = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/trapezoid_release/hidden64_ly3/LSTM_scaler_x_trapezoid_release_hidden64_ly3.pkl')
                scaler_y = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/trapezoid_release/hidden64_ly3/LSTM_scaler_y_trapezoid_release_hidden64_ly3.pkl')
                model_path = f"{GP_ROOT}/training_pack/model_pack/LSTM/trapezoid_release/hidden64_ly3/LSTM_trapezoid_release_hidden64_ly3.pth"
                MOVE_TYPE = 'release'
                hidden_size = 64
                num_layers = 3
            elif ACTION_TYPE == 'knife_release':
                scaler_x = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/knife_release/hidden64_ly2/LSTM_scaler_x_knife_release_hidden64_ly2.pkl')
                scaler_y = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/knife_release/hidden64_ly2/LSTM_scaler_y_knife_release_hidden64_ly2.pkl')
                model_path = f"{GP_ROOT}/training_pack/model_pack/LSTM/knife_release/hidden64_ly2/LSTM_knife_release_hidden64_ly2.pth"
                MOVE_TYPE = 'release'
                hidden_size = 64
                num_layers = 2
            elif ACTION_TYPE == 'PCB_release':
                scaler_x = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/PCB_release/hidden64_ly3/LSTM_scaler_x_PCB_release_hidden64_ly3.pkl')
                scaler_y = joblib.load(f'{GP_ROOT}/training_pack/model_pack/LSTM/PCB_release/hidden64_ly3/LSTM_scaler_y_PCB_release_hidden64_ly3.pkl')
                model_path = f"{GP_ROOT}/training_pack/model_pack/LSTM/PCB_release/hidden64_ly3/LSTM_PCB_release_hidden64_ly3.pth"
                MOVE_TYPE = 'release'
                hidden_size = 64
                num_layers = 3
            else:
                continue
            print("成功載入標準化 Scaler")
            
        except Exception as e:
            print(f"無法載入 Scaler 檔案: {e}")
            continue
            
        model = GripperControllerLSTM(input_size=6, hidden_size=hidden_size, num_layers=num_layers, output_size=3).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"成功載入模型權重: {model_path}")
        except Exception as e:
            print(f"無法載入模型權重: {e}")
            continue
        # --------------------------------
            
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
                        print(f"馬達 {dxl_id} 讀取失敗: {packetHandler.getTxRxResult(dxl_comm_result)}")
                        dxl_present_position = 3030 if dxl_id != 3 else 2030
                    else:
                        if dxl_present_position > 0x7FFFFFFF:
                            dxl_present_position -= 0x100000000
                    current_positions.append(dxl_present_position)
            origin_pos = [current_positions[0], current_positions[1], current_positions[2]]
            
        print(f"起始目標位置: {origin_pos}")
        print("正在將指爪移回起始點...")
        with com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, origin_pos[idx])
                sleep(1)
        sleep(1.0)
        print("移至起始點完成。")
        print("\n正在收集 1 秒的觸覺基準值以進行初始化平衡...")
        samples = []
        start_t = time()
        while time() - start_t < 1.0:
            with state_lock:
                samples.append((first_val, second_val, third_val))
            sleep(0.01)
        
        samples = np.array(samples)
        avg_1 = int(round(np.mean(samples[:, 0])))
        avg_2 = int(round(np.mean(samples[:, 1])))
        avg_3 = int(round(np.mean(samples[:, 2])))
        print(f"觸覺感測器基準平均值 (1秒內): Tactile_1={avg_1}, Tactile_2={avg_2}, Tactile_3={avg_3}")
        print("\n自動控制啟動！")
        SLEEP_INTERVAL = 0.05
        seq_history = []
        step = 0
        while True:
            if keyboard.is_pressed('x'):
                break 
            step = step + 1
            with state_lock:
                curr_pos = list(latest_motor_pos)
                curr_val = [first_val, second_val, third_val]
                
            curr_val_processed = [
                curr_val[0] - avg_1,
                curr_val[1] - avg_2,
                curr_val[2] - avg_3
            ]
            
            feature_vector = (curr_val_processed + [curr_pos[0]] + [curr_pos[1]] + [curr_pos[2]])
            
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
                    
            print(f"Step [{step:03d}] | "
                    f"觸覺壓力(原始): {curr_val} | "
                    f"處理後觸覺: {curr_val_processed} | "
                    f"當前位置: {curr_pos} | "
                    f"模型預測pos: {[round(float(x), 1) for x in target_pos]}")
            
            sleep(SLEEP_INTERVAL)
            
        print("\n自動控制序列已完成！")

        final_current_positions = []
        
        with com_lock:
            # 1. 讀取每個馬達的當前位置
            for dxl_id in DXL_IDS:
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                    portHandler, dxl_id, ADDR_PRESENT_POSITION
                )

                final_current_positions.append(dxl_present_position)

            # 2. 將目標位置設為剛剛讀取到的當前位置，使馬達停在原地
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, final_current_positions[idx])

        
            
    running = False
    print("程式結束。")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        running = False
        print("\n偵測到中斷，正在關閉系統...")
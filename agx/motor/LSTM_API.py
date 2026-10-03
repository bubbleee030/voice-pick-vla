# -*- coding: utf-8 -*-
import threading
from dynamixel_sdk import *
from time import sleep, time
import serial
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import joblib
import numpy as np
from flask import Flask, request, jsonify
# ==========================================
# 1. PyTorch 模型定義 (必須與 training.py 完全一致)
# ==========================================
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
        # 壓縮序列
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        # 解開序列
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        # 映射至控制輸出
        output = self.fc(out)
        return output
# ==========================================
# 2. 硬體參數設定 (移植自您的 gripper_record_CSV_API_LSTM.py)
# ==========================================
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
BAUDRATE_motor = 57600
DEVICENAME_motor = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RWC9P-if00-port0'
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_IDS = [1, 2, 3]
portHandler = PortHandler(DEVICENAME_motor)
packetHandler = PacketHandler(2.0)
if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE_motor):
    print("❌ 無法開啟馬達 Port 或設定 Baudrate")
    quit()
for i in DXL_IDS:
    packetHandler.reboot(portHandler, i)
sleep(1)
group_read = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)
for dxl_id in DXL_IDS:
    group_read.addParam(dxl_id)    
for i in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
BAUDRATE_sensor = 115200
DEVICENAME_sensor = '/dev/ttyUSB0' 
try:
    ser = serial.Serial(DEVICENAME_sensor, BAUDRATE_sensor, timeout=1)
    sleep(2)
except Exception as e:
    print(f"⚠️ 無法連接感測器: {e}")
    ser = None
# ==========================================
# 3. 全域變數與鎖設定
# ==========================================
com_lock = threading.Lock()
state_lock = threading.Lock()
origin_pos = [3072, 3072, 2048]
shared_pos = [3072, 3072, 2048]
running = True
gripper_speed = 20
# 感測器與馬達物理狀態
first_val, second_val, third_val = 0, 0, 0 
latest_motor_pos = list(origin_pos)
# 🤖 LSTM 自動控制狀態變數
lstm_status = {
    "state": "idle",           # idle, running, completed, error
    "steps_remaining": 0,
    "current_step": 0,
    "last_prediction": [0.0, 0.0, 0.0]
}
# 馬達安全限界設定 (相對於初始位置，防止過度擠壓或碰撞)
# origin_pos = [3072, 3072, 2048]
MOTOR_LIMITS = [
    (1872, 4272),  # 指爪 1 範圍
    (1872, 4272),  # 指爪 2 範圍
    (848, 3248)    # 指爪 3 範圍
]
# 讓夾爪回到初始位置
for idx, dxl_id in enumerate(DXL_IDS):
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
    sleep(0.5)
# ==========================================
# 4. 背景執行緒 (唯讀感測器與馬達狀態)
# ==========================================
def background_worker():
    global running, first_val, second_val, third_val, latest_motor_pos
    
    while running:
        # --- 1. 讀取感測器 ---
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
        
        # --- 2. 讀取馬達位置 ---
        with com_lock:
            group_read.txRxPacket()
            cur_pos = [group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4) for dxl_id in DXL_IDS]
        with state_lock:
            latest_motor_pos = list(cur_pos)
        
        sleep(0.01) # 100Hz 刷新速率
bg_thread = threading.Thread(target=background_worker, daemon=True)
bg_thread.start()
# ==========================================
# 5. LSTM 模型與標準化載入
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🧠 模型推理設備: {device}")
# 預載入標準化器
try:
    scaler_x = joblib.load('scaler_x_trapezoid.pkl')
    scaler_y = joblib.load('scaler_y_trapezoid.pkl')
    print("✅ 成功載入 Scaler X & Y")
except Exception as e:
    print(f"❌ 無法載入標準化 pkl 檔案: {e}")
    scaler_x, scaler_y = None, None
# 預載入模型
model = GripperControllerLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=3).to(device)
try:
    model.load_state_dict(torch.load('LSTM_trapezoid.pth', map_location=device))
    model.eval() # 設定為評估/推理模式
    print("✅ 成功載入 LSTM 模型權重")
except Exception as e:
    print(f"❌ 無法載入模型權重: {e}")
# ==========================================
# 6. 自動抓取演算法背景執行緒
# ==========================================
def lstm_auto_grasp_thread(total_steps, sleep_interval):
    global lstm_status, shared_pos
    
    if scaler_x is None or scaler_y is None:
        lstm_status["state"] = "error"
        print("❌ 未能成功載入標準化模組，取消自動抓取")
        return
    print("🤖 開始執行 LSTM 自動抓取序列...")
    lstm_status["state"] = "running"
    lstm_status["steps_remaining"] = total_steps
    lstm_status["current_step"] = 0
    
    # 初始化滾動歷史序列緩衝區 (List of normalized features)
    seq_history = []
    
    # 獲取初始位置與感測值
    with state_lock:
        prev_pos = list(latest_motor_pos)
        prev_val = [first_val, second_val, third_val]
    
    for step in range(total_steps):
        # 1. 取得當前感測器與馬達絕對狀態
        with state_lock:
            curr_pos = list(latest_motor_pos)
            curr_val = [first_val, second_val, third_val]
            
        # 2. 計算差值 (與前一個時間步的差)
        pos_diff = [curr_pos[i] - prev_pos[i] for i in range(3)]
        val_diff = [curr_val[i] - prev_val[i] for i in range(3)]
        
        # 組合當前特徵向量 [pos1_diff, pos2_diff, pos3_diff, val1_diff, val2_diff, val3_diff]
        feature_vector = pos_diff + val_diff
        
        # 3. 標準化特徵
        feature_scaled = scaler_x.transform([feature_vector])[0]
        seq_history.append(feature_scaled)
        
        # 4. 準備 PyTorch 輸入資料 (Batch_size = 1, Sequence_length = 當前長度 L, Features = 6)
        inputs_tensor = torch.tensor([seq_history], dtype=torch.float32).to(device)
        lengths_tensor = torch.tensor([len(seq_history)], dtype=torch.long)
        
        # 5. 進行預測
        with torch.no_grad():
            output_tensor = model(inputs_tensor, lengths_tensor)
            # 取得最後一個時間步的預測值
            pred_scaled = output_tensor[0, -1, :].cpu().numpy()
            
        # 6. 反標準化預測結果，得到下個時間步的預期位置增量 [pos1_diff, pos2_diff, pos3_diff]
        pred_diff = scaler_y.inverse_transform([pred_scaled])[0]
        
        # 7. 計算目標絕對位置，並套用馬達安全限界限制
        target_pos = []
        for i in range(3):
            # 新目標位置 = 當前絕對位置 + 預測的下一部增量
            raw_target = int(curr_pos[i] + pred_diff[i])
            # 套用邊界限制
            clipped_target = max(MOTOR_LIMITS[i][0], min(raw_target, MOTOR_LIMITS[i][1]))
            target_pos.append(clipped_target)
            
        # 8. 驅動馬達
        with com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, target_pos[idx])
                
        # 更新 shared_pos 使手動控制與自動控制同步
        shared_pos = list(target_pos)
        
        # 9. 更新歷史記錄以供下一步使用
        prev_pos = list(curr_pos)
        prev_val = list(curr_val)
        
        # 10. 更新 API 狀態
        lstm_status["current_step"] = step + 1
        lstm_status["steps_remaining"] = total_steps - (step + 1)
        lstm_status["last_prediction"] = [float(x) for x in pred_diff]
        
        # 依據 CSV 取樣頻率等待 (預設 0.16 秒接近您的 6Hz 採樣率)
        sleep(sleep_interval)
        
    lstm_status["state"] = "completed"
    print("⏹️ LSTM 自動抓取序列完成！")
# ==========================================
# 7. Flask API 路由定義
# ==========================================
app = Flask(__name__)
# --- 新增的 LSTM 自動抓取端點 ---
@app.route('/auto_grasp', methods=['POST'])
def auto_grasp():
    global lstm_status
    
    if lstm_status["state"] == "running":
        return jsonify({"status": "error", "message": "LSTM auto-grasp is already running"}), 400
        
    data = request.get_json() or {}
    steps = data.get('steps', 230)            # 預設跑 230 步 (根據 002.csv 長度)
    interval = data.get('interval', 0.16)     # 預設每步間隔 0.16 秒 (約 6Hz)
    
    # 啟動非阻塞背景執行緒
    t = threading.Thread(target=lstm_auto_grasp_thread, args=(steps, interval), daemon=True)
    t.start()
    
    return jsonify({
        "status": "started", 
        "message": "LSTM auto-grasp triggered successfully",
        "parameters": {"steps": steps, "interval": interval}
    })
# --- 手動與傳統控制 (保留您的原始控制，便於除錯) ---
@app.route('/command', methods=['POST'])
def handle_command():
    global shared_pos
    data = request.get_json()
    action = data.get('action')
    if not action:
        return jsonify({"status": "error", "message": "No action provided"}), 400
    with com_lock:
        if action == 'v':  #全開
            for i in range(3): shared_pos[i] += gripper_speed
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
        elif action == 'c': #全關
            for i in range(3): shared_pos[i] -= gripper_speed
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
        elif action == 'o': #瞬間張開
            for i in range(3): shared_pos[i] = origin_pos[i]
            for idx, dxl_id in enumerate(DXL_IDS):
                sleep(0.1)
                dxl_id_temp = (4 - dxl_id)
                idx_temp = (2 - idx)
                packetHandler.write4ByteTxRx(portHandler, dxl_id_temp, ADDR_GOAL_POSITION, shared_pos[idx_temp])
                
        # 1-6 各指獨立微調... (您原本寫的其餘分支)
        elif action == '1':
            shared_pos[0] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])
        elif action == '4':
            shared_pos[0] += gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])
        elif action == '2':
            shared_pos[1] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])
        elif action == '5':
            shared_pos[1] += gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])
        elif action == '3':
            shared_pos[2] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, shared_pos[2])
        elif action == '6':
            shared_pos[2] += gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, shared_pos[2])
    return jsonify({"status": "success", "action": action, "current_pos": shared_pos})
# --- 擴充的獲取狀態端點 ---
@app.route('/state', methods=['GET'])
def get_state():
    with state_lock:
        return jsonify({
            "status": "ok",
            "current_pos": latest_motor_pos,
            "tactile": [first_val, second_val, third_val],
            "lstm_auto_grasp": lstm_status  # 新增的自動化進度與模型狀態資訊
        })
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5002)
    except KeyboardInterrupt:
        running = False
        print("關閉中...")
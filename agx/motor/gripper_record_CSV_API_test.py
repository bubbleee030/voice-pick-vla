# -*- coding: utf-8 -*-
import threading
from dynamixel_sdk import *
from time import sleep, time
import serial
import csv
from flask import Flask, request, jsonify

# ==========================================
# 1. 硬體參數設定 (與你原本的一致)
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

BAUDRATE_sensor = 9600
DEVICENAME_sensor = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RWC9P-if00-port0' 
try:
    ser = serial.Serial(DEVICENAME_sensor, BAUDRATE_sensor, timeout=1)
    sleep(2)
except Exception as e:
    print(f"⚠️ 無法連接感測器: {e}")
    ser = None

# ==========================================
# 2. 全域變數與鎖設定
# ==========================================
com_lock = threading.Lock()
state_lock = threading.Lock()
origin_pos = [3072, 3072, 2048]
shared_pos = list(origin_pos)
running = True
gripper_speed = 15 # 每次按鍵移動的步長
script_start_unix = time()
latest_tactile_data = ""
latest_tactile_ts_unix = None
latest_motor_pos = list(origin_pos)

# 讓夾爪先回到初始位置
for idx, dxl_id in enumerate(DXL_IDS):
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])

# ==========================================
# 3. 背景記錄執行緒 (背景不斷寫入 CSV)
# ==========================================
def record_tactile_data(output_file):
    global running, latest_tactile_data, latest_tactile_ts_unix, latest_motor_pos
    print(f"開始錄製資料至 {output_file}...")
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_unix", "elapsed_s", "pos1", "pos2", "pos3", "tactile_data"])
        
        while running:
            now_unix = time()
            elapsed_s = now_unix - script_start_unix
            
            # 讀取感測器
            tactile_data = ""
            if ser and ser.in_waiting > 0:
                tactile_data = ser.readline().decode('utf-8', errors='ignore').strip()
                if tactile_data:
                    latest_tactile_data = tactile_data
                    latest_tactile_ts_unix = now_unix
            
            # 使用 Lock 保護馬達讀取動作
            with com_lock:
                group_read.txRxPacket()
                cur_pos = [group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4) for dxl_id in DXL_IDS]
            with state_lock:
                latest_motor_pos = list(cur_pos)
            
            writer.writerow([now_unix, elapsed_s] + cur_pos + [tactile_data])
            sleep(0.02) # 控制取樣率，避免過載

# 啟動背景錄製
bg_thread = threading.Thread(target=record_tactile_data, args=(f"tactile_data_{int(time())}.csv",), daemon=True)
bg_thread.start()

# ==========================================
# 4. Flask API 伺服器
# ==========================================
app = Flask(__name__)

@app.route('/command', methods=['POST'])
def handle_command():
    global shared_pos
    data = request.get_json()
    action = data.get('action')

    if not action:
        return jsonify({"status": "error", "message": "No action provided"}), 400

    # 使用 Lock 保護馬達寫入動作
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
            for i in range(3):
                shared_pos[i] = origin_pos[i]
            for idx, dxl_id in enumerate(DXL_IDS):
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])

        elif action == '1': #第一指關 
            shared_pos[0] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])

        elif action == '4': #第一指開
            shared_pos[0] += gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])

        elif action == '2': #第二指開
            shared_pos[1] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])

        elif action == '5': #第二指關
            shared_pos[1] += gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])

        elif action == '3': #第三指開
            shared_pos[2] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, shared_pos[2])

        elif action == '6': #第三指關
            shared_pos[2] += gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, shared_pos[2])
        
        elif action == '7':
            shared_pos[1] += gripper_speed
            shared_pos[0] -= gripper_speed
            packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])
            packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])

    return jsonify({"status": "success", "action": action, "current_pos": shared_pos})


@app.route('/state', methods=['GET'])
def get_state():
    now_unix = time()
    with state_lock:
        pos = list(latest_motor_pos)
        tactile = latest_tactile_data
        tactile_ts = latest_tactile_ts_unix
    return jsonify({
        "status": "ok",
        "server_time_unix": now_unix,
        "elapsed_s": now_unix - script_start_unix,
        "current_pos": pos,
        "tactile_data": tactile,
        "tactile_timestamp_unix": tactile_ts,
        "running": running,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "running": running})

@app.route('/stop', methods=['GET'])
def stop_system():
    global running
    running = False
    with com_lock:
        for i in DXL_IDS:
            packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    return jsonify({"status": "System stopped, motors disabled."})

if __name__ == '__main__':
    try:
        # 允許筆電連線
        app.run(host='0.0.0.0', port=5002)
    except KeyboardInterrupt:
        running = False
        print("關閉系統中...")
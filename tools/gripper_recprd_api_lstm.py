# -*- coding: utf-8 -*-
import threading
from dynamixel_sdk import *
from time import sleep, time
import serial
import csv
from flask import Flask, request, jsonify

# ==========================================
# 1. 硬體參數設定
# ==========================================
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
BAUDRATE_motor = 57600
# 請確認你的裝置路徑是否為 port0
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

BAUDRATE_sensor = 115200 # 已更新為 115200
DEVICENAME_sensor = '/dev/ttyUSB0' 
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
shared_pos = [3072, 3072, 2048]
running = True
is_recording = False
csv_file = None
csv_writer = None
gripper_speed = 20
script_start_unix = time() # 預先初始化避免背景執行緒崩潰
SENSOR_FRESHNESS_TIMEOUT_S = 0.5

# 感測器數值初始化
first_val, second_val, third_val = 0, 0, 0 
latest_tactile_ts_unix = None
latest_motor_pos = list(origin_pos)

# 讓夾爪回到初始位置

for idx, dxl_id in enumerate(DXL_IDS):
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
    sleep(0.5)
# ==========================================
# 3. 背景執行緒 (讀取與條件錄製)
# ==========================================
def background_worker():
    global running, is_recording, csv_file, csv_writer, script_start_unix
    global first_val, second_val, third_val, latest_tactile_ts_unix, latest_motor_pos
    
    while running:
        now_unix = time()
        elapsed_s = now_unix - script_start_unix
        
        # --- 1. 讀取感測器 ---
        if ser and ser.in_waiting > 0:
            # try:
            #     line = ser.readline().decode('utf-8', errors='ignore').strip()
            #     first_val = int(line.split(' ')[0])
            #     second_val = int(line.split(' ')[1])
            #     third_val = int(line.split(' ')[2])
            #     parts = line.split() # 解析空白分隔的數據[cite: 2]
            #     with state_lock:
            #         # first_val = int(line.split(' ')[0])
            #         # second_val = int(line.split(' ')[1])
            #         # third_val = int(line.split(' ')[2])
            #         # print(first_val)
            #         # print(second_val)
            #         # print(third_val)
            #         latest_tactile_ts_unix = now_unix
            # except (ValueError, IndexError):
            #     pass # 數據異常時跳過，保持舊值[cite: 2]
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(line)
                parts = line.split() # 解析空白分隔的數據[cite: 2]
                if len(parts) >= 3:
                    with state_lock:
                        first_val = int(parts[0])
                        second_val = int(parts[1])
                        third_val = int(parts[2])
                        latest_tactile_ts_unix = now_unix
            except (ValueError, IndexError):
                pass # 數據異常時跳過，保持舊值[cite: 2]
        
        # --- 2. 讀取馬達位置 ---
        with com_lock:
            group_read.txRxPacket()
            cur_pos = [group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4) for dxl_id in DXL_IDS]

        with state_lock:
            latest_motor_pos = list(cur_pos)
        
        # --- 3. 執行錄製 ---
        if is_recording and csv_writer:
            try:
                # 修正：將所有數值包進同一個 list[cite: 2]
                csv_writer.writerow([now_unix, elapsed_s] + cur_pos + [first_val, second_val, third_val])
                if csv_file:
                    csv_file.flush()
            except Exception as e:
                print(f"寫入失敗: {e}")
        
        sleep(0.01) # 約 100Hz 採樣率

bg_thread = threading.Thread(target=background_worker, daemon=True)
bg_thread.start()

# ==========================================
# 4. Flask API 伺服器
# ==========================================
app = Flask(__name__)

@app.route('/command', methods=['POST'])
def handle_command():
    global shared_pos, is_recording, csv_file, csv_writer, script_start_unix
    data = request.get_json()
    action = data.get('action')

    if not action:
        return jsonify({"status": "error", "message": "No action provided"}), 400

    # 錄影控制：按 s 開始，按 e 結束[cite: 2]
    if action == 's':
        if not is_recording:
            filename = f"tactile_data_{int(time())}.csv"
            csv_file = open(filename, "w", newline="")
            csv_writer = csv.writer(csv_file)
            # 更新 Header[cite: 2]
            csv_writer.writerow(["timestamp_unix", "elapsed_s", "pos1", "pos2", "pos3", "val1", "val2", "val3"])
            script_start_unix = time() # 重新計時，讓 elapsed_s 從 0 開始[cite: 2]
            is_recording = True
            print(f"🔴 開始錄影: {filename}")
        return jsonify({"status": "recording_started", "file": filename})

    elif action == 'e':
        if is_recording:
            is_recording = False
            sleep(0.1)
            if csv_file:
                csv_file.close()
                csv_file = None
                csv_writer = None
            print("⏹️ 錄影結束")
        return jsonify({"status": "recording_stopped"})

    # 馬達控制邏輯
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
                sleep(0.1)
                dxl_id_temp = dxl_id
                idx_temp = idx
                dxl_id_temp = (4 - dxl_id_temp)
                idx_temp = (2 - idx_temp)
                packetHandler.write4ByteTxRx(portHandler, dxl_id_temp, ADDR_GOAL_POSITION, shared_pos[idx_temp])

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
    with state_lock:
        now_unix = time()
        sensor_connected = bool(latest_tactile_ts_unix is not None and (now_unix - latest_tactile_ts_unix) <= SENSOR_FRESHNESS_TIMEOUT_S)
        return jsonify({
            "status": "ok",
            "api_version": "legacy",
            "server_time_unix": now_unix,
            "recording": is_recording,
            "current_pos": latest_motor_pos,
            "tactile": [first_val, second_val, third_val],
            "tactile_data": [first_val, second_val, third_val],
            "tactile_timestamp_unix": latest_tactile_ts_unix,
            "sensor_connected": sensor_connected,
            "sample_valid": sensor_connected,
            "elapsed_s": now_unix - script_start_unix
        })


@app.route('/set_position', methods=['POST'])
def set_position():
    global shared_pos
    data = request.get_json(silent=True) or {}
    positions = data.get('positions')
    if not isinstance(positions, list) or len(positions) != 3:
        return jsonify({"status": "error", "message": "positions must contain 3 values"}), 400
    with com_lock:
        shared_pos = [int(v) for v in positions]
        for idx, dxl_id in enumerate(DXL_IDS):
            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
    return jsonify({"status": "success", "current_pos": shared_pos})


@app.route('/health', methods=['GET'])
def health():
    payload = get_state().get_json()
    payload["ok"] = True
    return jsonify(payload)


@app.route('/stop', methods=['GET'])
def stop():
    with state_lock:
        return jsonify({"status": "stopped", "current_pos": latest_motor_pos})

if __name__ == '__main__':
    try:
        # 使用你指定的 Port 5002[cite: 2]
        app.run(host='0.0.0.0', port=5002)
    except KeyboardInterrupt:
        running = False
        if csv_file: csv_file.close()
        print("關閉中...")

# -*- coding: utf-8 -*-
from dynamixel_sdk import *
from time import sleep, time
import keyboard
import serial
import pandas as pd
import ctypes
import threading
import os


# === 設定參數 ===
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# 新增：連續感測數據的起始位址與長度 (124 ~ 1330
ADDR_TELEMETRY_START = 124  # 從 Present PWM 開始
LEN_TELEMETRY = 8           # 2 (PWM) + 2 (Current) + 4 (Velocity)

BAUDRATE = 57600
DEVICENAME =  os.getenv(
    "GRIPPER_MOTOR_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7NN-if00-port0",
)

TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0

DXL_IDS = [1, 2, 3]

SPEED = 35

base_folder = "saved_data/test"
""

# 1. 檢查並建立總資料夾 saved_data（如果不存在的話）
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

# 2. 在 saved_data 內部尋找尚未被使用的遞增數字資料夾
folder_index = 1
while os.path.exists(os.path.join(base_folder, str(folder_index))):
    folder_index += 1

# 3. 組合出最終的路徑（例如 saved_data/1）並建立它
save_folder = os.path.join(base_folder, str(folder_index))
os.makedirs(save_folder)

print(f"📁 程式啟動：已自動建立本次數據儲存路徑 -> 【 {save_folder} 】")

# === 初始化 Dynamixel 驅動 ===
portHandler   = PortHandler(DEVICENAME)
packetHandler = PacketHandler(2.0)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

# 重啟馬達
for i in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.reboot(portHandler, i)
sleep(1)

# 初始化同步讀寫器
group_write = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)
group_read  = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)

# 新增：初始化感測數據的同步讀取器 (讀取位址 124 寬度 8 bytes)
group_telemetry = GroupSyncRead(portHandler, packetHandler, ADDR_TELEMETRY_START, LEN_TELEMETRY)

# 註冊同步讀取的馬達 ID
for dxl_id in DXL_IDS:
    group_read.addParam(dxl_id)   
    group_telemetry.addParam(dxl_id)  # 新增：將馬達 ID 註冊進感測數據讀取器

# 開啟馬達扭力
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

# === 建立多執行緒 lock===
serial_lock = threading.Lock()

latest_tactile = [0, 0, 0]
tactile_lock = threading.Lock()

def bg_record_worker(stop_event, data_list, start_time):
    """在背景獨立運作，每 0.1 秒利用 SyncRead 抓取一次所有馬達的位置與感測數據"""
    print("\n 背景紀錄已啟動（安全鎖、位置與多重感測同步讀取模式）...")
    
    while not stop_event.is_set():
        current_record = {'Timestamp': round(time() - start_time, 2)}

        global latest_tactile
        
        # --- 1. 讀取觸覺感測器 (Non-blocking 方式) ---
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
                                with tactile_lock:
                                    latest_tactile = [val1, val2, val3]
                                break  # 成功取得最新一筆即時數據，結束搜尋
                            except ValueError:
                                continue  # 若數值解析失敗，繼續往前尋找舊的有效行
            except Exception:
                pass
        
        # 將最新的觸覺數據填入紀錄中
        with tactile_lock:
            current_record['Tactile_1'] = latest_tactile[0]
            current_record['Tactile_2'] = latest_tactile[1]
            current_record['Tactile_3'] = latest_tactile[2]
        
        # 使用 with 確保此時只有紀錄執行緒在使用序列埠
        with serial_lock:
            # 1. 讀取位置 (位址 132)
            pos_comm_result = group_read.txRxPacket()
            # 2. 讀取感測數據 (位址 124, 包含 PWM, Current, Velocity)
            tel_comm_result = group_telemetry.txRxPacket()
            
            for dxl_id in DXL_IDS:
                # --- A. 處理位置資料 ---
                dxl_present_position = None
                if pos_comm_result == COMM_SUCCESS and group_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION, 4):
                    dxl_present_position = group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                    if dxl_present_position > 0x7FFFFFFF:
                        dxl_present_position -= 0x100000000
                current_record[f'Motor_{dxl_id}_Pos'] = dxl_present_position
                
                # --- B. 處理 PWM, Current, Velocity 資料 ---
                pwm_val, cur_val, vel_val = None, None, None
                
                if tel_comm_result == COMM_SUCCESS and group_telemetry.isAvailable(dxl_id, ADDR_TELEMETRY_START, LEN_TELEMETRY):
                    # 從大 Buffer 中依據各自的相對位址與長度切出資料
                    raw_pwm = group_telemetry.getData(dxl_id, 124, 2)
                    raw_cur = group_telemetry.getData(dxl_id, 126, 2)
                    raw_vel = group_telemetry.getData(dxl_id, 128, 4)
                    
                    # 處理 16 位元有號整數 (二補數轉換)
                    if raw_pwm > 32767: raw_pwm -= 65536
                    if raw_cur > 32767: raw_cur -= 65536
                    # 處理 32 位元有號整數 (二補數轉換)
                    if raw_vel > 2147483647: raw_vel -= 4294967296
                    
                    # 轉換為真實物理單位
                    pwm_val = round(raw_pwm * 0.113, 2)      # 單位: %
                    cur_val = round(raw_cur * 1.0, 1)        # 單位: mA
                    vel_val = round(raw_vel * 0.229, 2)      # 單位: RPM
                
                # 存入紀錄字典motor
                current_record[f'Motor_{dxl_id}_PWM(%)'] = pwm_val
                current_record[f'Motor_{dxl_id}_Current(mA)'] = cur_val
                current_record[f'Motor_{dxl_id}_Velocity(RPM)'] = vel_val
        
        data_list.append(current_record)
        
        # 控制紀錄頻率為 0.1 秒一次
        sleep(0.1)


def keyboard_mode_with_record():
    while True:
        state = input("Choose the state(1:grasp 2:release):").strip()
        if state == '1':
            open_the_gripper()
            pos1 = 3072
            pos2 = 3072
            pos3 = 2048
            name = 'grasp'
            break
        elif state == '2':
            name = 'release'
            
            # 建立暫存讀取數值的列表
            current_positions = []
            
            # 逐一讀取 3 隻馬達的真實位置，記得加上 serial_lock 避免多執行緒衝突
            with serial_lock:
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
            
            # 將讀取到的當前位置分配給控制變數
            pos1 = current_positions[0]
            pos2 = current_positions[1]
            pos3 = current_positions[2]
            break
        else:
            print("input error!")
    
    # 建立資料容器與執行緒控制訊號
    recorded_data = []
    stop_recording_event = threading.Event()
    start_time = time()
    
    # 初始化並啟動背景紀錄執行緒
    record_thread = threading.Thread(
        target=bg_record_worker, 
        args=(stop_recording_event, recorded_data, start_time),
        daemon=True 
    )
    record_thread.start()
    
    print("\n[提示] 進入控制模式。")
    print("操作說明： 按住 'o' 張開、按住 'c' 閉合、按下 'x' 退出並自動存檔。")
    
    # 主控制迴圈 (維持極高的按鍵靈敏度)
    while True:
        if keyboard.is_pressed('o'):
            pos1 += SPEED
            pos2 += SPEED
            pos3 += SPEED
            
            # 寫入指令時上鎖，避免紀錄執行緒同時發送訊號導致硬體衝突
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 1, 116, pos1)
                packetHandler.write4ByteTxRx(portHandler, 2, 116, pos2)
                packetHandler.write4ByteTxRx(portHandler, 3, 116, pos3)
            sleep(0.02) # 微小延遲，留給底層硬體緩衝時間

        if keyboard.is_pressed('4'):
            pos1 += SPEED
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 1, 116, pos1)
            sleep(0.02) 

        if keyboard.is_pressed('5'):
            pos2 += SPEED
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 2, 116, pos2)
            sleep(0.02) 

        if keyboard.is_pressed('6'):
            pos3 += SPEED
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 3, 116, pos3)
            sleep(0.02) 
            
        if keyboard.is_pressed('c'):
            pos1 -= SPEED
            pos2 -= SPEED
            pos3 -= SPEED
            
            # 寫入指令時上鎖
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 1, 116, pos1)
                packetHandler.write4ByteTxRx(portHandler, 2, 116, pos2)
                packetHandler.write4ByteTxRx(portHandler, 3, 116, pos3)
            sleep(0.02)

        if keyboard.is_pressed('1'):
            pos1 -= SPEED
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 1, 116, pos1)
            sleep(0.02) 

        if keyboard.is_pressed('2'):
            pos2 -= SPEED
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 2, 116, pos2)
            sleep(0.02) 

        if keyboard.is_pressed('3'):
            pos3 -= SPEED
            with serial_lock:
                packetHandler.write4ByteTxRx(portHandler, 3, 116, pos3)
            
        if keyboard.is_pressed('x'):
            print("\n[系統] 正在結束控制，正在安全關閉背景紀錄...")
            # 通知紀錄執行緒停下，並等待其完全收尾
            stop_recording_event.set()
            record_thread.join()
            break
            
    # 離開控制迴圈後，將收集到的陣列轉為 CSV 存檔
    if recorded_data:
        df = pd.DataFrame(recorded_data)
        filename = os.path.join(save_folder, f"motor_telemetry_{name}.csv")
        cols = ['Timestamp', 'Tactile_1', 'Tactile_2', 'Tactile_3'] + [c for c in df.columns if c not in ['Timestamp', 'Tactile_1', 'Tactile_2', 'Tactile_3']]
        df = df[cols]
        df.to_csv(filename, index=False)
        print(f"【成功】本次控制過程的位置、馬達狀態與觸覺數據已儲存至：{filename}\n")
    else:
        print("【提示】未紀錄到任何數據。\n")


# === 原始功能：單純張開夾爪 ===3072,3072,2048
def open_the_gripper():
    original_pos = [3072,3072,2048]
    for idx, dxl_id in enumerate(DXL_IDS):
        packetHandler.write4ByteTxRx(portHandler, dxl_id%3 + 1, ADDR_GOAL_POSITION, original_pos[(idx+1)%3])
        sleep(0.5)


# === 主選單迴圈 ===
while True:
    mode = input("Choose the mode (1: Control & Record, 2: Open Gripper, x: Exit): ").strip()

    if mode == '1':
        keyboard_mode_with_record()  

    elif mode == '2':
        open_the_gripper()

    elif mode == 'x':
        print("關閉程式中...")
        break
    else:
        print("未知指令，請重新輸入。")

# === 關閉通訊與釋放馬達 ===
for i in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()
if ser:
    ser.close()
print("程式已安全結束。")
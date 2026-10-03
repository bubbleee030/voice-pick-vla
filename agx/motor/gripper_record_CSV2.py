# -*- coding: utf-8 -*-

import threading
from dynamixel_sdk import *
from time import sleep, time
import serial
import csv
import keyboard
import termios
import sys

# ... (前面的參數設定保持不變) ...
# === 設定參數 ===
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# === 馬達設定 ===
BAUDRATE_motor = 57600
DEVICENAME_motor = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RWC9P-if00-port0'

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_IDS = [1, 2, 3]

portHandler = PortHandler(DEVICENAME_motor)
packetHandler = PacketHandler(2.0)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE_motor)


for i in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.reboot(portHandler, i)
sleep(1)

group_write = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)
group_read = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)
present_positions = []

for dxl_id in DXL_IDS:
    group_read.addParam(dxl_id)    

for i in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# === 感測器設定 ===
BAUDRATE_sensor = 9600
DEVICENAME_sensor = '/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0'
ser = serial.Serial(DEVICENAME_sensor, BAUDRATE_sensor, timeout=1)
sleep(2)

# 建立一個 Lock 來確保同一時間只有一個執行緒在使用 Port
com_lock = threading.Lock()

# 使用列表來儲存共用的座標，方便在 thread 之間傳遞
shared_pos = [3072, 3072, 2048] 
running = True # 控制 thread 停止的旗標

def set_terminal_echo(enable): # 控制終端機顯示
    fd = sys.stdin.fileno()
    new_settings = termios.tcgetattr(fd)
    if enable:
        new_settings[3] |= termios.ECHO  # 開啟回顯
    else:
        new_settings[3] &= ~termios.ECHO # 關閉回顯
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

def control_gripper():
    global running
    gripper_speed = 10
    while running:
        # 使用 Lock 保護馬達寫入動作
        with com_lock:
            if keyboard.is_pressed('o'):
                for i in range(3): shared_pos[i] += gripper_speed
                for idx, dxl_id in enumerate(DXL_IDS):
                    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
            
            elif keyboard.is_pressed('c'):
                for i in range(3): shared_pos[i] -= gripper_speed
                for idx, dxl_id in enumerate(DXL_IDS):
                    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])

        if keyboard.is_pressed('x'):
            running = False
            break
        sleep(0.01) # 稍微暫停，避免過度占用 CPU

def record_tactile_data(output_file):
    global running

    for idx, dxl_id in enumerate(DXL_IDS):
        packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])

    sleep(1)

    set_terminal_echo(False)
    print("Recording started. Press 'o' to open, 'c' to close, 'x' to stop.")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "pos1", "pos2", "pos3", "tactile_data"])
        
        control_thread = threading.Thread(target=control_gripper)
        control_thread.start()
        
        start_time = time()
        try:
            while running:
                timestamp = time() - start_time
                
                # 讀取感測器
                tactile_data = ser.readline().decode().strip()
                
                # 使用 Lock 保護馬達讀取動作
                with com_lock:
                    group_read.txRxPacket()
                    cur_pos = [group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4) for dxl_id in DXL_IDS]
                
                writer.writerow([timestamp] + cur_pos + [tactile_data])
                print(f"Recorded: {timestamp:.2f}, {cur_pos}, {tactile_data}")

                if keyboard.is_pressed('x'):
                    running = False
                    break
        finally:
            running = False
            control_thread.join()
            # 恢復終端機回顯
            set_terminal_echo(True)
            # 善後處理
            for i in DXL_IDS:
                packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            ser.close()
            portHandler.closePort()
            print("Cleanup finished.")


# 啟動錄製
record_tactile_data("tactile_data.csv")
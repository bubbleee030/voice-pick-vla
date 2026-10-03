import threading
from dynamixel_sdk import *
from time import sleep, time
import serial
import csv
import keyboard
import sys
import termios
import tty

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

if not portHandler.openPort():
    print("❌ 無法開啟 Port")
    quit()

if not portHandler.setBaudRate(BAUDRATE_motor):
    print("❌ 無法設定 Baudrate")
    quit()

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
origin_pos = [3072, 3072, 2048]
shared_pos = [0,0,0]
for i in range(3):
    shared_pos[i] = origin_pos[i] 
running = True # 控制 thread 停止的旗標

def control_gripper():
    global running

    gripper_speed = 10
    while running:
        # 使用 Lock 保護馬達寫入動作
        with com_lock:
            if keyboard.is_pressed('v'):
                for i in range(3): shared_pos[i] += gripper_speed
                for idx, dxl_id in enumerate(DXL_IDS):
                    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
            
            elif keyboard.is_pressed('c'):
                for i in range(3): shared_pos[i] -= gripper_speed
                for idx, dxl_id in enumerate(DXL_IDS):
                    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
            
            elif keyboard.is_pressed('o'):
                for i in range(3):
                    shared_pos[i] = origin_pos[i]
                for idx, dxl_id in enumerate(DXL_IDS):
                    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])

            elif keyboard.is_pressed('1'):
                shared_pos[0] -= gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])

            elif keyboard.is_pressed('4'):
                shared_pos[0] += gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])

            elif keyboard.is_pressed('2'):
                shared_pos[1] -= gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])

            elif keyboard.is_pressed('5'):
                shared_pos[1] += gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])
    
            elif keyboard.is_pressed('3'):
                shared_pos[2] -= gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, shared_pos[2])

            elif keyboard.is_pressed('6'):
                shared_pos[2] += gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, shared_pos[2])
            
            elif keyboard.is_pressed('7'):
                shared_pos[1] += gripper_speed
                shared_pos[0] -= gripper_speed
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, shared_pos[0])
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, shared_pos[1])


        if keyboard.is_pressed('x'):
            running = False
            break
        sleep(0.01) # 稍微暫停，避免過度占用 CPU

def record_tactile_data(output_file):
    global running
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    for idx, dxl_id in enumerate(DXL_IDS):
        packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, shared_pos[idx])
    sleep(1) 
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "pos1", "pos2", "pos3", "tactile_data"])
        
        control_thread = threading.Thread(target=control_gripper)
        control_thread.start()
        
        start_time = time()
        try:
            tty.setcbreak(fd) 

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
            # 善後處理
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            for i in DXL_IDS:
                packetHandler.write1ByteTxRx(portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            ser.close()
            portHandler.closePort()
            print("Cleanup finished.")

# 啟動錄製
record_tactile_data("tactile_data.csv")
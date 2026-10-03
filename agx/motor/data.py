import serial
import time


PORT_NAME = '/dev/ttyUSB2' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(port=PORT_NAME, baudrate=BAUD_RATE, timeout=1)
    sleep(0.5)
    print(f"成功開啟 {PORT_NAME}，等待接收資料...")

    while True:
        if  ser and ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split()
            if len(parts) >= 3:
                    with state_lock:
                    first_val  = int(parts[0])
                    second_val = int(parts[1])
                    third_val  = int(parts[2])
            print(f"第一指資料:")

except KeyboardInterrupt:
    print("\n程式中斷")
except serial.SerialException as e:
    print(f"\n序列埠錯誤: {e}\n請確認線材、設備名稱，以及 dialout 權限是否已設定！")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
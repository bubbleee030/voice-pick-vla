import serial
import time

# 依照你的路線修改：
# 路線一 (CP2102) 通常是 '/dev/ttyUSB0'
# 路線二 (USB CDC) 通常是 '/dev/ttyACM0'
PORT_NAME = '/dev/ttyUSB2' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(port=PORT_NAME, baudrate=BAUD_RATE, timeout=1)
    print(f"成功開啟 {PORT_NAME}，等待接收資料...")

    while True:
        if ser.in_waiting > 0:
            raw_data = ser.readline()
            try:
                decoded = raw_data.decode('utf-8').strip()
                print(f"收到: {decoded}")
            except UnicodeDecodeError:
                print(f"收到(HEX): {raw_data.hex()}")
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n程式中斷")
except serial.SerialException as e:
    print(f"\n序列埠錯誤: {e}\n請確認線材、設備名稱，以及 dialout 權限是否已設定！")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
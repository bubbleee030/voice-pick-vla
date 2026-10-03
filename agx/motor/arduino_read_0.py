# -*- coding: utf-8 -*-
import serial
import socket
import threading
import time
from flask import Flask, jsonify, request

# --- 設定區 ---
SERIAL_PORT = '/dev/ttyUSB0'  
BAUD_RATE = 115200

app = Flask(__name__)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("AGX 數據中心已啟動...")

except Exception as e:
    print(f"無法連接 Arduino，錯誤訊息：{e}")
    ser = None


@app.route('/get_sensor', methods=['GET'])
def agx_get_sensor():
    if ser is None or not ser.is_open:
        return jsonify({"status": "error", "message": "Arduino 未連接"}), 500

    try:
        # 2. 清除累積的舊資料！
        # 這是非常關鍵的一步。如果你的感測器是用來做即時回饋（例如機械夾爪的觸覺數據），
        # 我們只想要「當下最新」的一筆資料，而不是 Arduino 幾秒前傳來、卡在排隊的舊數值。
        ser.reset_input_buffer() 

        # 3. 讀取最新的一行資料並解碼
        line = ser.readline().decode('utf-8').strip()

        first_val = int(line.split(' ')[0])
        second_val = int(line.split(' ')[1])
        third_val = int(line.split(' ')[2])
        # 4. 回傳給筆電
        return jsonify({
            "status": "success",
            "analog_value1": int(first_val),
            "analog_value2": int(second_val),
            "analog_value3": int(third_val),
        })

    except Exception as e:
        print(repr(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
# -*- coding: utf-8 -*-
import serial
import time

SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200

def read_tactile():
    print(f"Connecting to sensor on {SERIAL_PORT}...")
    
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2) 
    print("Connected. Start reading tactile data...")

    baseline = None
    start_time = time.time()
    try:
        while True:
            # Read and parse sensor line
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            values = line.split()
            
            if len(values) >= 3:
                try:
                    sensor1 = int(values[0])
                    sensor2 = int(values[1])
                    sensor3 = int(values[2])
                    
                    if baseline is None:
                        baseline = (sensor1, sensor2, sensor3)
                        
                    diff1 = sensor1 - baseline[0]
                    diff2 = sensor2 - baseline[1]
                    diff3 = sensor3 - baseline[2]
                    
                    elapsed_time = time.time() - start_time
                    yield elapsed_time, (diff1, diff2, diff3)
                except ValueError:
                    pass
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Stopping...")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == '__main__':
    try:
        for elapsed_time, diffs in read_tactile():
            print(f"Time: {elapsed_time:.2f}s | Tactile diffs: {diffs[0]}, {diffs[1]}, {diffs[2]}")
    except KeyboardInterrupt:
        pass


# -*- coding: utf-8 -*-
import threading
import sys

def run_agx_server():
    # Local imports to avoid importing packages/modules not needed on Laptop/Headless environments
    from motor import control
    from chart import run_server
    
    # 將 motor 中的 control() 放到後台執行緒執行，進行非阻塞馬達控制與鍵盤監聽
    control_thread = threading.Thread(target=control, daemon=True)
    control_thread.start()
    
    # 在主執行緒中開啟 TCP API 伺服器傳輸即時數據 (AGX IP為 192.168.1.100)
    run_server()

def run_laptop_client():
    from chart import chart_show, remote_data_generator
    
    # 預設 AGX IP 為 192.168.1.100，本機 (筆電) IP 為 192.168.1.50
    ip = "192.168.1.100"

        
    # 連接到伺服器並顯示即時圖表
    chart_show(remote_data_generator(ip))

def main():
    print("=" * 55)
    print(" Gripper LSTM Control & Visualization Mode Selector")
    print("=" * 55)
    print(" 1. AGX Server (Runs motor control & streams sensor data)")
    print(" 2. Laptop Client (Connects to AGX and displays charts)")
  
    print("=" * 55)
    
    try:
        choice = input("Select run mode (1/2): ").strip()
        if choice == '1':
            run_agx_server()
        elif choice == '2':
            run_laptop_client()

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == '__main__':
    main()

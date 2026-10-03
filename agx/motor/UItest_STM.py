import customtkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import threading
import time
import serial
import sys

# --- 序列埠讀取類別 ---
class ArduinoReader:
    def __init__(self, data_queue, port='/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0', baudrate=9600):
        self.data_queue = data_queue
        self.port = port
        self.baudrate = baudrate
        self.running = True

    def receive_data(self):
        """從 Arduino 讀取單一數值"""
        print(f"嘗試連接至 {self.port}...")
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) 
            print("連線成功！正在監控單一數據...")
            
            while self.running:
                if ser.in_waiting > 0:
                    try:
                        # 讀取一行數據並解析
                        raw_data = ser.readline().decode('utf-8').strip()
                        
                        if raw_data:
                            # 即使 Arduino 傳多個值，我們也只取第一個
                            # 支援 "123" 或 "123,456,789" 的格式
                            first_val = float(raw_data.split(' ')[0])
                            
                            if self.data_queue.full():
                                self.data_queue.get_nowait()
                            self.data_queue.put(first_val)
                            
                    except (ValueError, UnicodeDecodeError):
                        continue
                
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Serial 錯誤: {e}")

# --- 繪圖核心類別 ---
class StatusPlot:
    def __init__(self, parent, data_queue, on_display):
        self.on_display = on_display
        self.data_queue = data_queue
        
        # 僅儲存單一數據的歷史紀錄
        self.data_history = [0 for _ in range(self.on_display)]
        self.x_axis = [i for i in range(self.on_display)]

        # 建立 4 個子圖
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 9))
        self.lines = []

        # 初始化所有圖表
        for i in range(4):
            if i < 3:
                self.axes[i].get_xaxis().set_visible(False)
            
            # 每個圖表先畫上一條紅線
            line, = self.axes[i].plot(self.x_axis, [0]*self.on_display, color="red")
            self.lines.append(line)
            
            # 預設 Y 軸範圍
            self.axes[i].set_ylim(100, 600) 
            self.axes[i].set_title(f"Plot {i+1} {'(Active)' if i==0 else '(Standby)'}", fontsize=10)

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)

    def update_plot(self):
        """只更新第一個圖表的數據"""
        while True:
            if not self.data_queue.empty():
                new_val = self.data_queue.get()
                
                # 更新數據列表
                self.data_history.append(new_val)
                if len(self.data_history) > self.on_display:
                    self.data_history.pop(0)
                
                # 只針對第 0 條線（第一個圖表）更新 ydata
                self.lines[0].set_ydata(self.data_history)

                # 重新繪製畫布
                self.canvas.draw_idle()
            
            time.sleep(0.01)

# --- 主程式 ---
def main():
    root = customtkinter.CTk()
    root.title("Arduino Single Channel Monitor")
    root.geometry("800x800")
    customtkinter.set_appearance_mode('light')

    # 設定 Queue
    data_queue = queue.Queue(maxsize=50)

    # 1. 建立圖表 (顯示最近 100 筆數據)
    plot_mgr = StatusPlot(root, data_queue, on_display=100)

    # 2. 建立 Arduino 讀取器 (請確認你的 Port)
    arduino = ArduinoReader(data_queue, port='/dev/ttyUSB2', baudrate=115200)

    # 3. 啟動執行緒
    t_read = threading.Thread(target=arduino.receive_data, daemon=True)
    t_read.start()
    
    t_plot = threading.Thread(target=plot_mgr.update_plot, daemon=True)
    t_plot.start()

    root.protocol("WM_DELETE_WINDOW", sys.exit)
    root.mainloop()

if __name__ == "__main__":
    main()
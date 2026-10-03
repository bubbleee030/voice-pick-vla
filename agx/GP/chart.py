# -*- coding: utf-8 -*-
import sys
import time
import collections

def local_data_generator():
    """Generates real-time sensor and motor data from local hardware."""
    from tactile import read_tactile
    from motor import read_motor_positions
    for elapsed_time, diffs in read_tactile():
        try:
            m_pos = read_motor_positions()
        except Exception:
            m_pos = [3072, 3072, 2048]
        yield elapsed_time, diffs, m_pos

def remote_data_generator(ip="192.168.1.100", port=5002):
    """Polls the AGX Flask API server and yields data received over HTTP."""
    import urllib.request
    import json
    
    url = f"http://{ip}:{port}/api/data"
    print(f"Connecting to AGX Flask API at {url}...")
    
    last_time = None
    
    # Helper function to fetch data using urllib (built-in, no external dependencies)
    def fetch_data(api_url):
        try:
            with urllib.request.urlopen(api_url, timeout=1.0) as response:
                if response.status == 200:
                    return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            # Suppress excessive logging to keep terminal clean
            pass
        return None

    try:
        while True:
            data = fetch_data(url)
            if data:
                current_time = data.get("time")
                # Only yield if it's a new data point
                if current_time != last_time:
                    yield data["time"], data["tactile"], data["motor"]
                    last_time = current_time
            time.sleep(0.01)  # Poll at max ~100Hz
    except KeyboardInterrupt:
        print("\nClient plotting stopped.")

def run_server(port=5002):
    """Runs a Flask API server on AGX to expose sensor and motor positions."""
    import logging
    import threading
    from flask import Flask, jsonify
    from tactile import read_tactile
    from motor import read_motor_positions

    # Disable Flask default logging to avoid cluttering terminal
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app = Flask(__name__)

    # Thread-safe global store for the latest data point
    latest_data = {
        "time": 0.0,
        "tactile": [0, 0, 0],
        "motor": [3072, 3072, 2048]
    }
    data_lock = threading.Lock()

    def data_producer():
        nonlocal latest_data
        print("Sensor/Motor acquisition thread started.")
        try:
            for elapsed_time, diffs in read_tactile():
                try:
                    m_pos = read_motor_positions()
                except Exception:
                    m_pos = [3072, 3072, 2048]
                
                with data_lock:
                    latest_data = {
                        "time": elapsed_time,
                        "tactile": diffs,
                        "motor": m_pos
                    }
        except Exception as e:
            print(f"Error in data producer thread: {e}")

    # Start data producer thread in background
    producer_thread = threading.Thread(target=data_producer, daemon=True)
    producer_thread.start()

    @app.route('/api/data', methods=['GET'])
    def get_data():
        with data_lock:
            return jsonify(latest_data)

    print(f"AGX Flask API server starting on http://0.0.0.0:{port}...")
    try:
        app.run(host='192.168.1.100', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nFlask server shutting down.")

def chart_show(data_generator=None):
    import matplotlib.pyplot as plt
    if data_generator is None:
        data_generator = local_data_generator()
        
    # Configure Matplotlib styling for a premium aesthetic
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 觸覺感測器顏色
    color_sensor1 = "#B7471E"
    color_sensor2 = '#2EECB2'
    color_sensor3 = '#00D2FF'
    
    # 馬達位置顏色：橘色、綠色、藍色
    color_motor1 = "#FF9F43"
    color_motor2 = "#10AC84"
    color_motor3 = "#54A0FF"
    
    # 開啟互動模式
    plt.ion()
    
    # 建立暗色系視窗與兩個子圖 (上下排列)
    fig, (ax, ax_m) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    fig.patch.set_facecolor('#121212')  # 視窗暗色背景
    
    # 觸覺子圖樣式設定
    ax.set_facecolor('#1e1e1e')
    ax.grid(True, color='#2d2d2d', linestyle='--', linewidth=0.5)
    ax.set_title("Real-Time Tactile Sensor Detections", color='white', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel("Resistance Variation (ΔR)", color='#a0a0a0', fontsize=10, labelpad=5)
    ax.tick_params(colors='#a0a0a0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2d2d2d')
        
    # 馬達位置子圖樣式設定
    ax_m.set_facecolor('#1e1e1e')
    ax_m.grid(True, color='#2d2d2d', linestyle='--', linewidth=0.5)
    ax_m.set_title("Real-Time Motor Positions", color='white', fontsize=12, fontweight='bold', pad=10)
    ax_m.set_xlabel("Time (seconds)", color='#a0a0a0', fontsize=10, labelpad=5)
    ax_m.set_ylabel("Position Value", color='#a0a0a0', fontsize=10, labelpad=5)
    ax_m.tick_params(colors='#a0a0a0', labelsize=8)
    for spine in ax_m.spines.values():
        spine.set_color('#2d2d2d')
        
    # 設定滑動時間窗口容量
    max_points = 200
    time_history = collections.deque(maxlen=max_points)
    s1_history = collections.deque(maxlen=max_points)
    s2_history = collections.deque(maxlen=max_points)
    s3_history = collections.deque(maxlen=max_points)
    
    # 新增：馬達位置數據快取
    m1_history = collections.deque(maxlen=max_points)
    m2_history = collections.deque(maxlen=max_points)
    m3_history = collections.deque(maxlen=max_points)
    
    # 初始化觸覺線條
    line1, = ax.plot([], [], label='Sensor 1 (ΔR1)', color=color_sensor1, linewidth=2.0)
    line2, = ax.plot([], [], label='Sensor 2 (ΔR2)', color=color_sensor2, linewidth=2.0)
    line3, = ax.plot([], [], label='Sensor 3 (ΔR3)', color=color_sensor3, linewidth=2.0)
    
    # 初始化馬達線條
    line_m1, = ax_m.plot([], [], label='Motor 1 Pos', color=color_motor1, linewidth=2.0)
    line_m2, = ax_m.plot([], [], label='Motor 2 Pos', color=color_motor2, linewidth=2.0)
    line_m3, = ax_m.plot([], [], label='Motor 3 Pos', color=color_motor3, linewidth=2.0)
    
    # 設定觸覺圖例
    legend = ax.legend(loc='upper right', facecolor='#121212', edgecolor='#2d2d2d', fontsize=8)
    for text in legend.get_texts():
        text.set_color('white')
        
    # 設定馬達圖例
    legend_m = ax_m.legend(loc='upper right', facecolor='#121212', edgecolor='#2d2d2d', fontsize=8)
    for text in legend_m.get_texts():
        text.set_color('white')
        
    print("Plotting initialized. Press Ctrl+C in terminal to stop.")
    
    # We create a Queue for data points to plot from a background receiver thread
    import queue
    import threading
    data_queue = queue.Queue()
    
    # We define a producer thread that iterates over the generator and puts items in the queue
    def producer():
        try:
            for item in data_generator:
                data_queue.put(item)
        except Exception as e:
            print(f"Data acquisition thread error: {e}")
            
    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()
    
    # Show the plot window immediately (non-blocking)
    plt.show(block=False)
    
    try:
        # Loop while the figure window is open
        while plt.fignum_exists(fig.number):
            # Process all pending data in the queue
            updated = False
            while not data_queue.empty():
                try:
                    item = data_queue.get_nowait()
                    elapsed_time, diffs, m_pos = item
                    
                    # 更新時間與觸覺緩衝區
                    time_history.append(elapsed_time)
                    s1_history.append(diffs[0])
                    s2_history.append(diffs[1])
                    s3_history.append(diffs[2])
                    
                    # 更新馬達緩衝區
                    if m_pos is not None:
                        m1_history.append(m_pos[0])
                        m2_history.append(m_pos[1])
                        m3_history.append(m_pos[2])
                    else:
                        if m1_history:
                            m1_history.append(m1_history[-1])
                            m2_history.append(m2_history[-1])
                            m3_history.append(m3_history[-1])
                        else:
                            m1_history.append(3072)
                            m2_history.append(3072)
                            m3_history.append(2048)
                    updated = True
                except queue.Empty:
                    break
                    
            if updated:
                # 更新觸覺線條數據
                line1.set_data(time_history, s1_history)
                line2.set_data(time_history, s2_history)
                line3.set_data(time_history, s3_history)
                
                # 更新馬達線條數據
                line_m1.set_data(time_history, m1_history)
                line_m2.set_data(time_history, m2_history)
                line_m3.set_data(time_history, m3_history)
                
                # 設定滑動時間軸的 X 軸邊界（最近 10 秒）
                if len(time_history) > 1:
                    x_start = max(0.0, elapsed_time - 10.0)
                    ax.set_xlim(x_start, elapsed_time + 0.5)
                    ax_m.set_xlim(x_start, elapsed_time + 0.5)
                else:
                    ax.set_xlim(0.0, 5.0)
                    ax_m.set_xlim(0.0, 5.0)
                    
                # 觸覺子圖 Y 軸自動調整
                all_y = list(s1_history) + list(s2_history) + list(s3_history)
                if all_y:
                    min_y = min(all_y)
                    max_y = max(all_y)
                    y_span = max_y - min_y
                    margin = max(10.0, y_span * 0.15)
                    ax.set_ylim(min_y - margin, max_y + margin)
                    
                # 馬達子圖 Y 軸自動調整
                all_m = list(m1_history) + list(m2_history) + list(m3_history)
                if all_m:
                    min_m = min(all_m)
                    max_m = max(all_m)
                    m_span = max_m - min_m
                    margin_m = max(50.0, m_span * 0.15)  
                    ax_m.set_ylim(min_m - margin_m, max_m + margin_m)
                
                # Request redrawing
                fig.canvas.draw_idle()
            
            # This processes GUI events (draw, resize, keypress, close) and keeps window responsive
            plt.pause(0.01)
            
    except KeyboardInterrupt:
        print("\nPlotting session ended.")
    finally:
        # Keep window interactive at the end for inspection
        if plt.fignum_exists(fig.number):
            plt.ioff()
            print("Closing session, showing static view...")
            plt.show()

if __name__ == '__main__':
    chart_show()
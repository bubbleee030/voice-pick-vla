import os
import glob
import pandas as pd

# 定義原始資料夾與輸出資料夾
input_dir = './saved_data/knife/rawdata'
output_dir = './saved_data/knife/cut_average_data'

def cut_datasets_with_average(input_dir, output_dir):
    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立輸出資料夾：{output_dir}")
    
    # 獲取 input_dir 底下的所有子資料夾（例如 1, 2, 3...）
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    # 排序資料夾名稱（以數字大小優先排序）
    subdirs.sort(key=lambda x: int(x) if x.isdigit() else x)
    
    if not subdirs:
        # 如果沒有子資料夾，直接尋找 input_dir 下的 csv 檔案
        print(f"在 '{input_dir}' 中未找到子資料夾，將直接搜尋該目錄下的 CSV 檔案...")
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
        process_csv_files(csv_files, input_dir, output_dir)
        return
        
    print(f"開始處理資料，共找到 {len(subdirs)} 個資料夾...")
    
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, subdir)
        
        # 尋找該子資料夾底下的 CSV 檔案
        csv_files = glob.glob(os.path.join(subdir_path, '*.csv'))
        if csv_files:
            os.makedirs(output_subdir_path, exist_ok=True)
            process_csv_files(csv_files, subdir_path, output_subdir_path)

def process_csv_files(csv_files, src_dir, dest_dir):
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"正在處理: {os.path.basename(src_dir)}/{file_name} ...", end="")
        
        try:
            # 讀取 CSV 檔案
            df = pd.read_csv(file_path)
            
            # 尋找 timestamp 欄位 (忽略大小寫)
            timestamp_col = None
            for col in df.columns:
                if col.lower() == 'timestamp':
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                print(" [無 Timestamp 欄位，跳過]")
                continue
            
            # 找出所有包含 'Tactile' 的欄位
            tactile_cols = [col for col in df.columns if 'tactile' in col.lower()]
            if not tactile_cols:
                print(" [無 Tactile 欄位，跳過]")
                continue
            
            # 取得前五秒的數據
            df_first_5s = df[df[timestamp_col] < 5]
            
            if df_first_5s.empty:
                print(" [前五秒無數據，跳過]")
                continue
                
            # 計算前五秒的 Tactile 平均值並取整數
            averages = {}
            for col in tactile_cols:
                # 轉為 float，計算平均後四捨五入取整數
                avg_val = df_first_5s[col].astype(float).mean()
                averages[col] = int(round(avg_val))
            
            # 剃除 timestamp < 5 的資料 (保留 timestamp >= 5 的資料)
            df_cut = df[df[timestamp_col] >= 5].copy()
            
            # 依據平均值做 data preprocess (減去平均值)
            for col in tactile_cols:
                df_cut[col] = df_cut[col] - averages[col]
            
            # 儲存檔案
            output_file_path = os.path.join(dest_dir, file_name)
            df_cut.to_csv(output_file_path, index=False)
            
            avg_str = ", ".join([f"{col}平均: {averages[col]}" for col in tactile_cols])
            print(f" [成功，{avg_str}，剩餘 {len(df_cut)} 筆/原 {len(df)} 筆]")
            
        except Exception as e:
            print(f"\n[錯誤] 處理 {file_name} 時發生異常: {e}")

if __name__ == '__main__':
    cut_datasets_with_average(input_dir, output_dir)

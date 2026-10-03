import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
import joblib  # 用於儲存標準化模型
# ==========================================
# 1. PyTorch Dataset 數據集封裝
# ==========================================
class RobotActionDataset(Dataset):
    def __init__(self, x_list, y_list, lengths):
        self.x_list = x_list
        self.y_list = y_list
        self.lengths = lengths
    def __len__(self):
        return len(self.x_list)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_list[idx], dtype=torch.float32),
            torch.tensor(self.y_list[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.long)
        )
def collate_fn_padd(batch):
    """
    動態將一個批次（Batch）內長度不同的 CSV 補零對齊
    """
    # 依長度從大到小排序，優化 PyTorch pack 效能
    batch.sort(key=lambda x: x[2], reverse=True)
    x, y, lengths = zip(*batch)
    
    # 自動補零 (Padding) 
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.0)
    lengths = torch.stack(lengths)
    
    return x_padded, y_padded, lengths
# 💡 修改：加入 action_type 參數（可傳入 'grasp' 或 'release'）來過濾資料
def load_preprocessed_data(data_dir, action_type='grasp'):
    # 💡 尋找所有子資料夾下對應 action_type 的 CSV 檔案
    csv_files = glob.glob(os.path.join(data_dir, '*', f'*{action_type}.csv'))
    if not csv_files:
        raise FileNotFoundError(f"在 '{data_dir}' 底下找不到任何前處理後的 {action_type} CSV 檔案。")
        
    raw_x, raw_y = [], []
    lengths = []
    all_features, all_targets = [], []
    
    # 💡 定義新的輸入欄位（15 個特徵）與預測目標（3 個馬達位置）
    feature_cols = [
        'Tactile_1', 'Tactile_2', 'Tactile_3',
        'Motor_1_Pos', 'Motor_1_PWM(%)', 'Motor_1_Current(mA)', 'Motor_1_Velocity(RPM)',
        'Motor_2_Pos', 'Motor_2_PWM(%)', 'Motor_2_Current(mA)', 'Motor_2_Velocity(RPM)',
        'Motor_3_Pos', 'Motor_3_PWM(%)', 'Motor_3_Current(mA)', 'Motor_3_Velocity(RPM)'
    ]
    target_cols = ['Motor_1_Pos', 'Motor_2_Pos', 'Motor_3_Pos']
    
    # 讀取資料
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if len(df) < 5: 
            continue
            
        # 💡 使用新特徵欄位提取資料
        x_data_full = df[feature_cols].values
        y_data_full = df[target_cols].values
        
        x_data = x_data_full[:-1, :]
        y_data = y_data_full[1:, :]
        raw_x.append(x_data)
        raw_y.append(y_data)
        lengths.append(len(df) - 1)
        
        all_features.append(x_data)
        all_targets.append(y_data)
        
    # 模型內部訓練需要浮點數標準化 (避免數值落差影響 LSTM 收斂)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(np.vstack(all_features))
    scaler_y.fit(np.vstack(all_targets))
    
    scaled_x = [scaler_x.transform(x) for x in raw_x]
    scaled_y = [scaler_y.transform(y) for y in raw_y]
    
    print(f"成功載入 {len(scaled_x)} 個 [{action_type}] 動作生命週期序列。")
    return scaled_x, scaled_y, lengths, scaler_x, scaler_y
# ==========================================
# 2. PyTorch 現成 LSTM 控制器架構
# ==========================================
class GripperControllerLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, output_size=3):
        super(GripperControllerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        # 壓縮序列（自動忽略補零區域）
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        # 解開序列
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        # 映射至 3 個爪子的控制輸出
        output = self.fc(out)
        return output
# ==========================================
# 3. 訓練主程式
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"當前使用設備: {device}")
    
    # 超參數
    BATCH_SIZE = 4
    EPOCHS = 2000
    LEARNING_RATE = 0.001
    
    # 💡 1. 在此處選擇您要訓練的模型動作類型：'grasp' 或 'release'
    ACTION_TYPE = 'release' 
    
    # 💡 2. 指向新的前處理資料夾路徑
    datapath = './saved_data/trapezoid/preprocessed_data'
    
    # 💡 3. 讀取時過濾對應的動作資料
    x_data, y_data, lengths, scaler_x, scaler_y = load_preprocessed_data(datapath, action_type=ACTION_TYPE)
    
    # 建立 Dataloader
    dataset = RobotActionDataset(x_data, y_data, lengths)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padd)
    
    # 💡 4. 初始化模型時帶入 input_size=15
    model = GripperControllerLSTM(input_size=15, hidden_size=64, num_layers=2, output_size=3).to(device)
    
    criterion = nn.MSELoss(reduction='none') 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    print(f"開始訓練現成 LSTM [{ACTION_TYPE}] 模型...")
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        total_valid_steps = 0
        
        for batch_x, batch_y, batch_lens in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x, batch_lens)
            
            # 計算 Loss 並手動建立 Mask (遮罩)，不讓補零的區域干擾權重學習
            loss_raw = criterion(predictions, batch_y)
            mask = torch.zeros(batch_y.shape[:2], dtype=torch.float32).to(device)
            for i, l in enumerate(batch_lens):
                mask[i, :l] = 1.0
                
            loss_masked = loss_raw * mask.unsqueeze(-1)
            loss = loss_masked.sum() / mask.sum()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * mask.sum().item()
            total_valid_steps += mask.sum().item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss / total_valid_steps:.6f}")
            
    # 💡 5. 儲存時自動區分動作檔名，避免 grasp 和 release 互蓋
    torch.save(model.state_dict(), f'LSTM_trapezoid_{ACTION_TYPE}.pth')
    joblib.dump(scaler_x, f'scaler_x_trapezoid_{ACTION_TYPE}.pkl')
    joblib.dump(scaler_y, f'scaler_y_trapezoid_{ACTION_TYPE}.pkl')
    print(f"[{ACTION_TYPE}] 模型與標準化 Scaler 已成功儲存！")
if __name__ == '__main__':
    main()
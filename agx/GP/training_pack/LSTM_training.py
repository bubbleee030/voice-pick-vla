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
    batch.sort(key=lambda x: x[2], reverse=True)
    x, y, lengths = zip(*batch)
    
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.0)
    lengths = torch.stack(lengths)
    
    return x_padded, y_padded, lengths

def load_preprocessed_data(data_dir, action_type='release'):
    csv_files = glob.glob(os.path.join(data_dir, '*', f'*{action_type}.csv'))
    if not csv_files:
        raise FileNotFoundError(f"在 '{data_dir}' 底下找不到任何前處理後的 {action_type} CSV 檔案。")
        
    raw_x, raw_y = [], []
    lengths = []
    all_features, all_targets = [], []
    
    feature_cols = ['Tactile_1','Tactile_2','Tactile_3','Motor_1_Pos','Motor_2_Pos','Motor_3_Pos']
    target_cols = ['Motor_1_Pos','Motor_2_Pos','Motor_3_Pos']
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if len(df) < 5: 
            continue
            
        x_data_full = df[feature_cols].values
        y_data_full = df[target_cols].values
        
        x_data = x_data_full[:-1, :]
        y_data = y_data_full[1:, :]
        raw_x.append(x_data)
        raw_y.append(y_data)
        lengths.append(len(df) - 1)
        
        all_features.append(x_data)
        all_targets.append(y_data)
        
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
    # 🎯 這裡改為你期望的參數傳入形式，保留未來彈性
    def __init__(self, input_size, hidden_size, num_layers, output_size):
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
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        output = self.fc(out)
        return output

# ==========================================
# 3. 訓練主程式
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"當前使用設備: {device}")
    
    # 🎯 核心修改 1：自選物件名稱（修改此處即可更換物件）
    OBJECT_NAME = 'knife' 
    
    # 💡 選擇您要訓練的模型動作類型：'grasp' 或 'release'
    ACTION_TYPE = 'release' 
    
    # 超參數
    BATCH_SIZE = 4
    EPOCHS = 1200
    LEARNING_RATE = 0.001
    
    # 🎯 核心修改 2：多組參數巡覽 (hidden_size, num_layers)
    param_combinations = [
        (32, 2),
        (64, 2),
        (64, 3)
    ]
    
    # 🎯 核心修改 3：依物件動態決定資料夾路徑
    datapath = f'./saved_data/{OBJECT_NAME}/cut_average_data'
    
    # 讀取時過濾對應的動作資料（與參數無關，在迴圈外讀取一次即可）
    x_data, y_data, lengths, scaler_x, scaler_y = load_preprocessed_data(datapath, action_type=ACTION_TYPE)
    
    # 建立 Dataloader
    dataset = RobotActionDataset(x_data, y_data, lengths)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padd)
    
    # 🎯 核心修改 4：開始自動巡覽參數組態進行訓練
    for hidden_size, num_layers in param_combinations:
        param_dir_name = f"hidden{hidden_size}_ly{num_layers}"
        print("\n" + "="*50)
        print(f"🚀 開始訓練組態：物件={OBJECT_NAME}, 動作={ACTION_TYPE}, Hidden={hidden_size}, Layers={num_layers}")
        print("="*50)
        
        # 🎯 核心修改 5：動態建立儲存資料夾路徑 (.../LSTM/PCB/hidden32_ly2/)
        save_dir = f'./training_pack/model_pack/LSTM/{OBJECT_NAME}_{ACTION_TYPE}/{param_dir_name}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 🎯 核心修改 6：實例化模型，並確實傳入 4 個參數
        model = GripperControllerLSTM(
            input_size=6, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=3
        ).to(device)
        
        criterion = nn.MSELoss(reduction='none') 
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)

        model.train()
        
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            total_valid_steps = 0
            
            for batch_x, batch_y, batch_lens in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_x, batch_lens)
                
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
                
            avg_loss = epoch_loss / total_valid_steps
            scheduler.step(avg_loss)
            
            if epoch % 100 == 0 or epoch == 1: # 調整為每 100 epoch 印一次，簡化日誌畫面
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

        # 🎯 核心修改 7：組態專屬的模型與標準化 Scaler 檔名與路徑儲存
        model_name = f'LSTM_{OBJECT_NAME}_{ACTION_TYPE}_{param_dir_name}.pth'
        scaler_x_name = f'LSTM_scaler_x_{OBJECT_NAME}_{ACTION_TYPE}_{param_dir_name}.pkl'
        scaler_y_name = f'LSTM_scaler_y_{OBJECT_NAME}_{ACTION_TYPE}_{param_dir_name}.pkl'
        
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        joblib.dump(scaler_x, os.path.join(save_dir, scaler_x_name))
        joblib.dump(scaler_y, os.path.join(save_dir, scaler_y_name))
        
        print(f"✅ 組態 {param_dir_name} 訓練完成！")
        print(f"儲存路徑: {save_dir}")

    print("\n🎉 所有 LSTM 參數組合皆已訓練並儲存完畢！")

if __name__ == '__main__':
    main()
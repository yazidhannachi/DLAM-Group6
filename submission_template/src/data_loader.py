import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):

        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.data = torch.FloatTensor(df.iloc[:,1:].values)

        self.valid_windows = []
        
        for series_id, group in df.groupby('series_id'):
            group_indices = group.index.tolist()
            num_points = len(group_indices)
            
            max_start_offset = num_points - seq_len - pred_len
            
            for i in range(max_start_offset + 1):
                start_global_idx = group_indices[i]
                self.valid_windows.append((start_global_idx, series_id))


    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        start_idx, ts_id = self.valid_windows[idx]
        
        x_start = start_idx
        x_end = start_idx + self.seq_len
        y_end = x_end + self.pred_len
        
        data_in = self.data.iloc[x_start:x_end,:]
        y_target = self.data.iloc[x_end:y_end,-1]
        
        return data_in.unsqueeze(-1), y_target.unsqueeze(-1), series_id
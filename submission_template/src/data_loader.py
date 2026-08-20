import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):

        self.seq_len = seq_len
        self.label_len = int(self.seq_len/2)
        self.pred_len = pred_len

        df_ = df.reset_index(drop=True)
        self.data = torch.FloatTensor(df_.drop(["remainder__series_id", "remainder__timestamp"],axis=1).values)
        self.valid_windows = []
        
        for series_id, group in df_.groupby('remainder__series_id'):
            group_indices = group.index.tolist()
            num_points = len(group_indices)
            
            max_start_offset = num_points - seq_len - pred_len
            
            for i in range(max_start_offset + 1):
                start_global_idx = group_indices[i]
                self.valid_windows.append((start_global_idx, series_id))


    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        start_idx, series_id = self.valid_windows[idx]
        
        x_start = start_idx
        x_end = start_idx + self.seq_len
        y_end = x_end + self.pred_len
        
        x_enc = self.data[x_start:x_end,:]
        y_target = self.data[x_end:y_end,:]
        
        return x_enc, y_target, series_id
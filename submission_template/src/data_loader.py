import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = df.copy()

        self.original_index = df.index.to_numpy()

        model_cols = [c for c in df.columns if c not in ["series_id", "series_idx", "timestamp"]]
        self.target_idx = model_cols.index("target")
        self.data_x = torch.FloatTensor(df[model_cols].values)
        self.data_y = torch.FloatTensor(df[model_cols].values)

        self.valid_windows = []

        for series_idx, group in df.groupby("series_idx"):
            pos_idx = np.arange(len(df))[df["series_idx"].values == series_idx]
            num_points = len(pos_idx)

            max_start = num_points - (seq_len + pred_len)
            for i in range(max_start + 1):
                x_pos = pos_idx[i:i + seq_len]
                y_pos = pos_idx[i + seq_len:i + seq_len + pred_len]
                y_original_idx = self.original_index[y_pos]
                self.valid_windows.append((x_pos, y_pos, y_original_idx, series_idx))

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        x_pos, y_pos, y_original_idx, series_idx = self.valid_windows[idx]
        x_enc = self.data_x[x_pos]
        y_target = self.data_y[y_pos]
        return x_enc, y_target, series_idx, y_original_idx
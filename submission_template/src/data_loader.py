import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, stride=1, insample_fit=None, residual_hist=None, base_forecast=None, residual_target=None, as_feature=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

        self.insample_fit = insample_fit
        self.residual_hist = residual_hist
        self.base_forecast = base_forecast
        self.residual_target = residual_target
        self.as_feature = as_feature

        df = df.copy()

        self.original_index = df.index.to_numpy()

        self.model_cols = [c for c in df.columns if c not in ["series_id", "series_idx", "timestamp"]]
        self.target_idx = self.model_cols.index("target")
        self.data_x = torch.FloatTensor(df[self.model_cols].values)
        self.data_y = torch.FloatTensor(df[self.model_cols].values)

        self.valid_windows = []

        for series_idx, group in df.groupby("series_idx"):
            pos_idx = np.arange(len(df))[df["series_idx"].values == series_idx]
            num_points = len(pos_idx)

            max_start = num_points - (seq_len + pred_len)
            for i in range(0, max_start + 1, stride):
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

        if self.as_feature is not None:
            if self.as_feature==False:
                # replace target input channel with in-sample residual history
                res_hist = torch.from_numpy(self.residual_hist[idx]).float()
                x_enc[:, self.target_idx] = res_hist

                # replace target output channel with future residual target
                res_target = torch.from_numpy(self.residual_target[idx]).float()
                y_target[:, self.target_idx] = res_target

            elif self.as_feature==True:
                fit_hist = torch.from_numpy(self.insample_fit[idx]).float().unsqueeze(-1)
                x_enc = torch.cat([x_enc, fit_hist], dim=-1)

                # output target channel becomes future residual target
                res_target = torch.from_numpy(self.residual_target[idx]).float()
                y_target[:, self.target_idx] = res_target


        return x_enc, y_target, series_idx, y_original_idx, idx
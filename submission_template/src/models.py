import pickle 
import torch
import os
from datetime import datetime
import pandas as pd
import numpy as np
import lightning as L
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from external.xlstm_mixer.xlstm_mixer.models.xlstm_mixer import xLSTMMixer
from statsmodels.tsa.arima.model import ARIMA

from data_loader import CustomDataset
from utils import extract_initial_history

MODEL_REGISTRY = {
}

BASE_MODEL_REGISTRY = {
    "ARIMA": ARIMA
}

class ModelBuilder:
    def __init__(self, model_config):
        self.hybrid = model_config["hybrid"]
        self.model_name = model_config["model_name"]
        self.model_kwargs = model_config["model_kwargs"]
        if self.hybrid == True:
            self.base_model_name = model_config["base_model"]
            self.base_model_kwargs = model_config["base_model_kwargs"]
            
    def build(self):
        if self.hybrid == True:
            return HybridModel(BASE_MODEL_REGISTRY[self.base_model_name](**self.base_model_kwargs), MODEL_REGISTRY[self.model_name](**self.model_kwargs))
        return MODEL_REGISTRY[self.model_name](**self.model_kwargs)


class HybridModel:
    """Hybrid model from a classical base model and a residual model.
    The residual model is assumed to be a pytorch model.
    """
    def __init__(self, base_model, residual_model):
        self.base_model = base_model
        self.residual_model = residual_model

    def fit(self, X, y):
        self.base_model.fit(X, y)
        in_sample_y = self.base_model.predict(start=0, end=len(y)-1)
        in_sample_res = y - in_sample_y
        self.residual_model.fit(X, in_sample_res)
    
    def save(self, path):
        base_path = os.path.splittext(path)[0] + "_base.pkl"
        res_path = os.path.splittext(path)[0] + "_res.pth"
        self.base_model.save(base_path)
        torch.save(self.residual_model.state_dict(), res_path)
    
    def predict(self, X):
        y_pred_base = self.base_model.predict(X)
        y_pred_res = self.residual_model.predict(X)
        y_pred = y_pred_base + y_pred_res
        return y_pred

class FourierApprox:
    def __init__(self):
        pass

    def fit(self, X, y, cutoff=0):
        self.train_spectrum = np.fft.fft(y, norm='ortho')
        self.N = len(y)
        self.freqs = np.fft.fftfreq(self.N)
        self.cutoff = cutoff 
        
    def save(self, path):
        with open("path", w) as f:
            pickle.dump(self, f)

    def predict(self, X):
        t = np.arange(len(X)) #???
        y = np.zeros(len(t))
        count=0
        if np.abs(self.train_spectrum[0]) >= self.cutoff:
            count+=1
            y+= self.train_spectrum[0].real / np.sqrt(self.N)

        pos = np.where((self.freqs > 0) & (self.freqs < 0.5))[0]
        for k in pos:
            if np.abs(X[k]) >= self.cutoff:
                count += 1
                y += 2*np.abs(X[k])/np.sqrt(N)*np.cos(2*np.pi*self.freqs[k]*t+np.angle(X[k]))

        if self.N%2==0:
            k=self.N//2
            if np.abs(X[k]) >= cutoff:
                count+=1
                y += X[k].real/np.sqrt(N)*np.cos(np.pi*t)
        return y

BASE_MODEL_REGISTRY["fourier"] = FourierApprox

class xLSTMMixerWrapper(xLSTMMixer):

    def __init__(self, seq_len, pred_len, enc_in, num_series=96, embedding_dim=8, kwargs={}):
        self.embedding_dim = embedding_dim
        self.num_series = num_series
        enc_in_new = enc_in + self.embedding_dim
        super().__init__(pred_len, seq_len, enc_in_new, **kwargs)
        #self.model = xLSTMMixer(pred_len, seq_len, enc_in,**kwargs)  
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.series_embedding = torch.nn.Embedding(num_embeddings=num_series, embedding_dim=embedding_dim)
        self.emb_dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, series_idx=None, mask=None):
        series_idx = series_idx.detach().clone().to(dtype=torch.long, device=self.device)
        emb = self.series_embedding(series_idx)
        emb = self.emb_dropout(emb)
        emb = emb.unsqueeze(1).repeat(1, x_enc.shape[1], 1)
        x_enc_embedded = torch.cat([x_enc, emb], dim=-1)
        return super().forward(x_enc_embedded, x_mark_enc, x_dec, x_mark_dec, mask)
        

    def predict(self, x_enc, series_idx=None):
        if x_enc.dim() == 2:
            x_enc = x_enc.unsqueeze(0)
        predictions = self.forward(x_enc, 0, 0, 0, series_idx=series_idx)
        return predictions


    def predict_autoregressive(self, history_df, df_val):

        self.to(self.device)
        ignore_cols = ["series_id", "series_idx", "timestamp"]
        feature_cols = [c for c in history_df.columns if c not in ignore_cols]
    
        self.target_idx = feature_cols.index("target")

        pred_df = df_val.copy()
        pred_df['target'] = np.nan

        progress_bar = tqdm(list(df_val["series_id"].unique()), desc="Series ID", leave=True)

        for series_id, group in df_val.groupby('series_id'):

            series_idx = history_df.loc[history_df["series_id"] == series_id, "series_idx"].iloc[0]
            series_idx_tensor = torch.tensor([series_idx], dtype=torch.long, device=self.device)
            series_history = history_df.loc[history_df["series_id"] == series_id, feature_cols]

            history_window = series_history.tail(self.seq_len)
            current_window = torch.tensor(history_window.values, dtype=torch.float32).unsqueeze(0).to(self.device)
            series_future = pred_df.loc[pred_df["series_id"] == series_id, feature_cols]
            total_forecast_horizon = len(series_future)

            group_indices = group.index.tolist()

            all_forecasts = []
            steps_generated = 0

            with torch.no_grad():
                while steps_generated < total_forecast_horizon:

                    y_pred = self.predict(current_window, series_idx=series_idx_tensor) 
                    y_pred_target = y_pred[:, :, self.target_idx]
                    y_pred_target_np = y_pred_target.cpu().numpy().flatten()
                    all_forecasts.extend(y_pred_target_np)
                    steps_generated += self.pred_len

                    step_start = steps_generated - self.pred_len
                    step_end = steps_generated
                    
                    if step_end > total_forecast_horizon:
                        future_feature_rows = series_future.iloc[step_start:total_forecast_horizon].values
                        padding_needed = self.pred_len - len(future_feature_rows)
                        pad_rows = np.repeat(future_feature_rows[-1:], padding_needed, axis=0)
                        future_feature_rows = np.vstack([future_feature_rows, pad_rows])
                    else:
                        future_feature_rows = series_future.iloc[step_start:step_end].values

                    new_rows = torch.tensor(future_feature_rows, dtype=torch.float32).to(self.device)
                    new_rows[:, self.target_idx] = torch.tensor(y_pred_target_np, dtype=torch.float32).to(self.device)
                    
                    next_window = current_window.squeeze(0).clone()
                    next_window = torch.cat([next_window[self.pred_len:], new_rows], dim=0)
                    current_window = next_window.unsqueeze(0)

            pred_df.loc[group_indices, 'target'] = all_forecasts[:total_forecast_horizon]
            progress_bar.update(1)
        progress_bar.close()
        return pred_df

    def predict_autoregressive_tensor(self, x_enc, y, series_idx, target_idx, rollout_steps):
        context = x_enc
        prediction_list = []
        for i in range(rollout_steps):
            predictions = self.predict(context, series_idx=series_idx)
            prediction_list.append(predictions)
            idx_start = i*self.pred_len
            idx_end = (i+1)*self.pred_len
            y_slice = y[:, idx_start:idx_end, :].clone()
            y_slice[:,:,target_idx] = predictions[:,:,target_idx]
            new_context = torch.cat([context, y_slice], dim=1)
            context = new_context[:,self.pred_len:,:]
        combined_preds = torch.cat(prediction_list, dim=1)
        combined_preds = combined_preds[:,:y.shape[1],:]
        return combined_preds


            

MODEL_REGISTRY["xLSTMMixer"] = xLSTMMixerWrapper
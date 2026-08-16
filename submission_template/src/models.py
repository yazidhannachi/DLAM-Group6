import pickle 
import torch
import os
from datetime import datetime
import pandas as pd
import numpy as np
import lightning as L
from tqdm import tqdm
from torch.utils.data import DataLoader
from external.xlstm_mixer.xlstm_mixer.models.xlstm_mixer import xLSTMMixer
from statsmodels.tsa.arima.model import ARIMA

from data_loader import CustomDataset
from utils import extract_initial_history

MODEL_REGISTRY = {
}

BASE_MODEL_REGISTRY = {
    "ARIMA": ARIMA
}

CRITERION_REGISTRY = {
    "MAE": torch.nn.L1Loss,
    "MSE": torch.nn.MSELoss
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

class xLSTMMixerWrapper:
    def __init__(self, seq_len, pred_len, enc_in, criterion, lr=1e-3, kwargs={}):
        self.model = xLSTMMixer(pred_len, seq_len, enc_in,**kwargs)  
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = CRITERION_REGISTRY[criterion]()
        
    def fit(self, df_train, df_val, save_path, batch_size=128, max_epochs=100, patience=5):
        os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
        dataset_train = CustomDataset(df_train, self.seq_len, self.pred_len)
        dataset_val = CustomDataset(df_val, self.seq_len, self.pred_len)
        #self.initial_history = extract_initial_history(df_train, seq_len=self.seq_len)
        train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size, shuffle=True)

        

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_loss = float('inf')
        best_vloss = float('inf')
        counter = 0
        for epoch in range(max_epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=True)
            for i, batch in enumerate(train_loader):
                x_enc, y, _ = batch
                x_enc = x_enc.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(x_enc, 0, 0, 0).squeeze()
                loss = self.criterion(predictions, y)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
                progress_bar.update(1)
                if i >10:
                    break

            progress_bar.close()
            running_vloss = 0.0
        
            self.model.eval()
            val_bar = tqdm(val_loader, desc=f"Validation", leave=False)
            with torch.no_grad():
                for vdata in val_loader:
                    vx, vy, _ = vdata
                    vx = vx.to(self.device)
                    vy = vy.to(self.device)
                    vpred= self.model(vx,0,0,0).squeeze()
                    vloss = self.criterion(vpred, vy)
                    running_vloss += vloss.item()
                    val_bar.update(1)
                    val_bar.set_postfix(batch_loss=f"{vloss.item():.4f}")

            avg_vloss = running_vloss / len(val_loader)
            print(f"Epoch {epoch+1}/{max_epochs} - Avg. train. Loss: {running_loss / len(train_loader):.5f} - Avg. val. Loss: {running_vloss / len(val_loader):.5f}")
    
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join(save_path, "checkpoints", f'model_{timestamp}_{epoch}')
                torch.save(self.model.state_dict(), model_path)
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

            


    def predict(history_df, df_val):
        
        pred_df = df_val.copy()
        pred_df['prediction'] = pd.NA
        
        for series_id, group in df_val.groupby('series_id'):
            
            series_history = history_df.loc[history_df["series_id"] == series_id,:] 
            series_future = pred_df.loc[pred_df["series_id"] == series_id,:]

            group_indices = group.index.tolist()

            all_forecasts = []
            steps_generated = 0
            with torch.no_grad():
                while steps_generated < total_forecast_horizon:

                    y_pred = self.model.forecast(current_window) 

                    y_pred_np = y_pred.cpu().numpy().flatten()
                    all_forecasts.extend(y_pred_np)
                    steps_generated += self.pred_len

                    if steps_generated >= total_forecast_horizon:
                        break
                        
                    next_window = current_window.squeeze(0).clone()
                    new_rows = next_window[-1].repeat(self.pred_len, 1) 
                    new_rows[:, self.target_idx] = torch.tensor(y_pred_np, dtype=torch.float32).to(self.device)
                    next_window = torch.cat([next_window[self.pred_len:], new_rows], dim=0)
                    current_window = next_window.unsqueeze(0)

            pred_df.iloc[group_indices, 'prediction'] = all_forecasts

        return pred_df

MODEL_REGISTRY["xLSTMMixer"] = xLSTMMixerWrapper
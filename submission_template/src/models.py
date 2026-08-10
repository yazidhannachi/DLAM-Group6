import pickle 
import torch
import lightning as L
from torch.utils.data import DataLoader
#from external.xLSTM-Mixer.models.xlstm_mixer import xLSTMMixer
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
        hybrid = model_config["hybrid"]
        model_name = model_config["model"]
        model_kwargs = model_config["model_kwargs"]
        if hybrid == True:
            base_model_name = model_config["base_model"]
            base_model_kwargs = model_config["base_model_kwargs"]
            
    def build(self):
        if hybrid == True:
            return HybridModel(BASE_MODEL_REGISTRY[base_model_name](**base_model_kwargs), MODEL_REGISTRY[model_name](**model_kwargs))
        return MODEL_REGISTRY[model_name](**model_kwargs)


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
    def __init__(self, seq_len, pred_lenP, criterion, lr=1e-3, kwargs={}):
        self.model = xLSTMMixer(pred_len, seq_len,**kwargs)  
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = CRITERION_REGISTRY[criterion]()
        
    def fit(self, df, batch_size=32, max_epochs=100, patience=5):
        dataset = CustomDataset(df, self.seq_len, self.pred_len)
        self.initial_history = df.grou
        loader = DataLoader(dataset, batch_size, shuffle=True)
        
        self.model.train()
        best_loss = float('inf')
        counter = 0
        for epoch in range(max_epochs):
            running_loss = 0.0
            for batch in loader:
                x, y, _ = batch
                
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(x_features)
                loss = self.criterion(predictions, y_residual)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if running_loss/len(loader) > best_loss:
                    best_loss = running_loss/len(loader)
                    patience = 0
                else:
                    patience += 1

                if counter >= patience:
                    break

            print(f"Epoch {epoch+1}/{max_epochs} - Loss: {running_loss / len(loader):.5f}")


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

BASE_MODEL_REGISTRY["xLSTM-Mixer"] = xLSTMMixerWrapper
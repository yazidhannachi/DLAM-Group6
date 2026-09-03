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
}

class ModelBuilder:
    def __init__(self, model_config):
        self.hybrid = model_config["hybrid"]
        self.model_name = model_config["model_name"]
        self.model_kwargs = model_config["model_kwargs"]
        '''if self.hybrid == True:
            self.base_model_name = model_config["base_model"]
            self.base_model_kwargs = model_config["base_model_kwargs"]'''
            
    def build(self):
        '''if self.hybrid == True:
            return HybridModel(BASE_MODEL_REGISTRY[self.base_model_name](**self.base_model_kwargs), MODEL_REGISTRY[self.model_name](**self.model_kwargs))'''
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

class ARIMAWrapper:
    def __init__(self, order=(1,0,1), seasonal_order=(1,0,1,24), trend='n', method='innovations_mle'):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.method = method
        self.fitted_model = None

    def _fit_one(self, y):
        """Fit a single ARIMA model on y. Used by both fit() and precompute_offline() so
        train-time and validation-time residuals always come from the exact same model spec
        (order, seasonal_order, trend). Fits in float64 - statsmodels converges poorly (or not
        at all) in float32 - and falls back to the 'statespace' method if the configured one
        fails to converge."""
        y = np.asarray(y, dtype=np.float64)
        model = ARIMA(
            y, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend,
            enforce_stationarity=True, enforce_invertibility=True,
        )
        try:
            fitted = model.fit(method=self.method)
        except Exception:
            fitted = model.fit(method="statespace")
        return fitted

    def fit(self, y):
        self.train_target = np.asarray(y, dtype=np.float64)
        self.fitted_model = self._fit_one(self.train_target)
        return self

    def predict_in_sample(self):
        if self.fitted_model is None:
            raise RuntimeError("ARIMAWrapper must be fit before calling predict_in_sample().")
        return np.asarray(self.fitted_model.predict(start=0, end=len(self.train_target)-1),dtype=np.float32)

    def predict_future(self, steps):
        if self.fitted_model is None:
            raise RuntimeError("ARIMAWrapper must be fit before calling predict_future().")
        return np.asarray(self.fitted_model.forecast(steps=steps),dtype=np.float32)

    def precompute_offline(self,dataset):
        n = len(dataset)
        seq_len = dataset.seq_len
        pred_len = dataset.pred_len
        target_idx = dataset.target_idx

        insample_fit = np.zeros((n, seq_len), dtype=np.float32)
        residual_hist = np.zeros((n, seq_len), dtype=np.float32)
        base_forecast = np.zeros((n, pred_len), dtype=np.float32)
        residual_target = np.zeros((n, pred_len), dtype=np.float32)

        progress_bar = tqdm(total=n, desc="Precomputed windows", leave=True)

        for idx in range(n):
            x_pos, y_pos, _, _ = dataset.valid_windows[idx]

            x_target = dataset.data_x[x_pos, target_idx].numpy()
            y_target = dataset.data_y[y_pos, target_idx].numpy()

            fitted = self._fit_one(x_target)
            # in-sample predictions
            x_fit = np.asarray(fitted.predict(start=0, end=len(x_target)-1), dtype=np.float32)

            # future forecast for prediction horizon
            y_base = np.asarray(fitted.forecast(steps=pred_len), dtype=np.float32)

            insample_fit[idx] = x_fit
            residual_hist[idx] = x_target - x_fit
            base_forecast[idx] = y_base
            residual_target[idx] = y_target - y_base
            progress_bar.update(1)
        progress_bar.close()
        return insample_fit, residual_hist, base_forecast, residual_target

BASE_MODEL_REGISTRY["ARIMA"] = ARIMAWrapper


class FourierApprox:
    def __init__(self, cutoff=0):
        self.cutoff = cutoff

    def fit(self, y):
        self.N = len(y)
        #y_ = y.values
        self.train_spectrum = np.fft.fft(y, norm='ortho')
        self.freqs = np.fft.fftfreq(self.N)

    def predict_in_sample(self):
        t = np.arange(self.N)
        return self.reconstruct(t)

    def predict_future(self, H):
        t = np.arange(self.N, self.N + H)
        return self.reconstruct(t)

    def reconstruct(self, t):
        y = np.zeros(len(t))

        if np.abs(self.train_spectrum[0]) >= self.cutoff:
            y += self.train_spectrum[0].real / np.sqrt(self.N)

        pos = np.where((self.freqs > 0) & (self.freqs < 0.5))[0]
        for k in pos:
            if np.abs(self.train_spectrum[k]) >= self.cutoff:
                y += (2*np.abs(self.train_spectrum[k])/np.sqrt(self.N)*np.cos(2 * np.pi * self.freqs[k]*t + np.angle(self.train_spectrum[k])))

        if self.N % 2 == 0:
            k = self.N // 2
            if np.abs(self.train_spectrum[k]) >= self.cutoff:
                y += self.train_spectrum[k].real / np.sqrt(self.N) * np.cos(np.pi * t)

        return y
    
    def precompute_offline(self,dataset):
       
        n = len(dataset)
        seq_len = dataset.seq_len
        pred_len = dataset.pred_len
        target_idx = dataset.target_idx

        insample_fit = np.zeros((n, seq_len), dtype=np.float32)
        residual_hist = np.zeros((n, seq_len), dtype=np.float32)
        base_forecast = np.zeros((n, pred_len), dtype=np.float32)
        residual_target = np.zeros((n, pred_len), dtype=np.float32)

        progress_bar = tqdm(total=n, desc="Precomputed windows", leave=True)

        for idx in range(n):
            x_pos, y_pos, _, _ = dataset.valid_windows[idx]

            x_target = dataset.data_x[x_pos, target_idx].numpy()
            y_target = dataset.data_y[y_pos, target_idx].numpy()

            fitted = self.fit(x_target)

            # in-sample predictions
            x_fit = self.predict_in_sample()

            # future forecast for prediction horizon
            y_base = self.predict_future(pred_len)

            insample_fit[idx] = x_fit
            residual_hist[idx] = x_target - x_fit
            base_forecast[idx] = y_base
            residual_target[idx] = y_target - y_base
            progress_bar.update(1)
        progress_bar.close()
        return insample_fit, residual_hist, base_forecast, residual_target

BASE_MODEL_REGISTRY["fourier"] = FourierApprox

class xLSTMMixerWrapper(xLSTMMixer):

    def __init__(self, seq_len, pred_len, enc_in, num_series=96, embedding_dim=8, kwargs={}):
        self.embedding_dim = embedding_dim
        self.num_series = num_series
        enc_in_new = enc_in + self.embedding_dim
        # xLSTMMixer's own positional signature is (pred_len, seq_len, enc_in, ...) - the
        # opposite order from this wrapper's (seq_len, pred_len, enc_in, ...). This call is
        # intentionally reordered to match the parent, not swapped by mistake. Assert against
        # the parent's own stored attributes (set via BaseModel.__init__) right away, before
        # the reassignment below would otherwise mask a real argument-order regression.
        super().__init__(pred_len, seq_len, enc_in_new, **kwargs)
        assert self.seq_len == seq_len and self.pred_len == pred_len, (
            f"xLSTMMixer base got seq_len={self.seq_len}, pred_len={self.pred_len} but wrapper "
            f"expected seq_len={seq_len}, pred_len={pred_len} - argument order mismatch in super().__init__() call"
        )
        #self.model = xLSTMMixer(pred_len, seq_len, enc_in,**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.series_embedding = torch.nn.Embedding(num_embeddings=num_series, embedding_dim=embedding_dim)
        self.emb_dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, series_idx=None, mask=None):
        series_idx = series_idx.detach().clone().to(dtype=torch.long, device=self.device)
        assert series_idx.max().item() < self.num_series, (
            f"series_idx contains index {series_idx.max().item()} but the series embedding "
            f"only has num_series={self.num_series} entries - LabelEncoder produced more "
            f"distinct series than num_series was set to."
        )
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

    def predict_autoregressive_tensor(self, x_enc, y, series_idx, target_idx, rollout_steps,
                                       base_model=None, as_feature=None, insample_fit=None):
        """base_model/as_feature/insample_fit are only used for the hybrid case: they let this
        (used during autoregressive TRAINING) refit the base model at every rollout step, the
        same way predict_autoregressive_hybrid does during validation/inference. Without this,
        every step past the first reused CustomDataset's single static fit from the original
        window, which misaligns train-time and validation-time behavior for the hybrid prior.
        """
        prediction_list = []

        def pad_to_pred_len(y_slice):
            # Mirror predict_autoregressive's end-of-horizon padding (repeat the last
            # available future row) instead of silently building a short window.
            if y_slice.shape[1] < self.pred_len:
                padding_needed = self.pred_len - y_slice.shape[1]
                pad_rows = y_slice[:, -1:, :].repeat(1, padding_needed, 1)
                y_slice = torch.cat([y_slice, pad_rows], dim=1)
            return y_slice

        if base_model is None:
            context = x_enc
            for i in range(rollout_steps):
                predictions = self.predict(context, series_idx=series_idx)
                prediction_list.append(predictions)
                idx_start = i*self.pred_len
                idx_end = (i+1)*self.pred_len
                y_slice = pad_to_pred_len(y[:, idx_start:idx_end, :].clone())
                y_slice[:,:,target_idx] = predictions[:,:,target_idx]
                new_context = torch.cat([context, y_slice], dim=1)
                context = new_context[:,self.pred_len:,:]
        else:
            # Keep `context` as the RAW window at all times (no appended prior-fit column,
            # target channel holds genuine absolute values) and let _prepare_hybrid_window
            # build the augmented model input fresh at every step, exactly like
            # predict_autoregressive_hybrid does.
            if as_feature:
                context = x_enc[:, :, :-1].clone()
            else:
                assert insample_fit is not None, (
                    "insample_fit is required to reconstruct raw target history when "
                    "as_feature=False, since CustomDataset replaces the target channel with "
                    "the in-sample residual for the first window."
                )
                context = x_enc.clone()
                context[:, :, target_idx] = context[:, :, target_idx] + insample_fit

            for i in range(rollout_steps):
                batch_prepared, batch_y_base = [], []
                for b in range(context.shape[0]):
                    prepared_window, y_base = self._prepare_hybrid_window(
                        context[b:b+1], base_model, as_feature, target_idx
                    )
                    batch_prepared.append(prepared_window)
                    batch_y_base.append(y_base)

                prepared_batch = torch.cat(batch_prepared, dim=0)
                y_base_batch = torch.tensor(
                    np.stack(batch_y_base), dtype=torch.float32, device=context.device
                )

                predictions = self.predict(prepared_batch, series_idx=series_idx)
                prediction_list.append(predictions)

                idx_start = i*self.pred_len
                idx_end = (i+1)*self.pred_len
                y_slice = pad_to_pred_len(y[:, idx_start:idx_end, :].clone())
                # Next window's raw history needs an absolute target value (base forecast +
                # predicted residual), matching predict_autoregressive_hybrid's
                # `y_pred_final_np = y_base + y_pred_res_np`, so the base model can be refit
                # on genuine target history at the next step.
                y_slice[:,:,target_idx] = y_base_batch + predictions[:,:,target_idx]

                new_context = torch.cat([context, y_slice], dim=1)
                context = new_context[:,self.pred_len:,:]

        combined_preds = torch.cat(prediction_list, dim=1)
        combined_preds = combined_preds[:,:y.shape[1],:]
        return combined_preds

    def _prepare_hybrid_window(self, current_window, base_model, as_feature, target_idx):

        raw_window = current_window.squeeze(0).detach().cpu().numpy().copy()
        target_hist = raw_window[:, target_idx]

        base_model.fit(target_hist)
        x_fit = base_model.predict_in_sample()
        y_base = base_model.predict_future(self.pred_len)

        if as_feature == False:
            raw_window[:, target_idx] = target_hist - x_fit
            prepared_window = torch.tensor(raw_window, dtype=torch.float32, device=self.device).unsqueeze(0)

        elif as_feature == True:
            fit_col = x_fit.reshape(-1, 1)
            raw_window_aug = np.concatenate([raw_window, fit_col], axis=1)
            prepared_window = torch.tensor(raw_window_aug, dtype=torch.float32, device=self.device).unsqueeze(0)


        return prepared_window, y_base

    def predict_autoregressive_hybrid(self, history_df, df_val, base_model, as_feature=False):
        self.to(self.device)

        ignore_cols = ["series_id", "series_idx", "timestamp"]
        feature_cols = [c for c in history_df.columns if c not in ignore_cols]
        target_idx = feature_cols.index("target")
        self.target_idx = target_idx

        for series_id, group in df_val.groupby("series_id"):
            print(series_id, len(group))
            break

        pred_df = df_val.copy()
        pred_df["target"] = np.nan

        progress_bar = tqdm(list(df_val["series_id"].unique()), desc="Series ID", leave=True)

        for series_id, group in df_val.groupby("series_id"):
            series_idx = history_df.loc[history_df["series_id"] == series_id, "series_idx"].iloc[0]
            series_idx_tensor = torch.tensor([series_idx], dtype=torch.long, device=self.device)

            # Keep RAW history here
            series_history = history_df.loc[history_df["series_id"] == series_id, feature_cols].copy()
            history_window = series_history.tail(self.seq_len)
            current_window_raw = torch.tensor(
                history_window.values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            series_future = pred_df.loc[pred_df["series_id"] == series_id, feature_cols]
            total_forecast_horizon = len(series_future)
            group_indices = group.index.tolist()

            all_forecasts = []
            steps_generated = 0

            with torch.no_grad():
                while steps_generated < total_forecast_horizon:
                    prepared_window, y_base = self._prepare_hybrid_window(
                        current_window_raw, base_model, as_feature, target_idx
                    )

                    y_pred = self.predict(prepared_window, series_idx=series_idx_tensor)

                    y_pred_target = y_pred[:, :, target_idx]
                    y_pred_res_np = y_pred_target.cpu().numpy().flatten()

                    y_pred_final_np = y_base + y_pred_res_np
                    all_forecasts.extend(y_pred_final_np.tolist())

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

                    new_rows = torch.tensor(future_feature_rows, dtype=torch.float32, device=self.device)

                    new_rows[:, target_idx] = torch.tensor(y_pred_final_np, dtype=torch.float32, device=self.device)

                    next_window = current_window_raw.squeeze(0).clone()
                    next_window = torch.cat([next_window[self.pred_len:], new_rows], dim=0)
                    current_window_raw = next_window.unsqueeze(0)

            pred_df.loc[group_indices, "target"] = all_forecasts[:total_forecast_horizon]
            progress_bar.update(1)

        progress_bar.close()
        return pred_df

MODEL_REGISTRY["xLSTMMixer"] = xLSTMMixerWrapper
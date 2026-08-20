import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtsmixer import TSMixerExt

from data_loader import CustomDataset


TARGET_IDX = 22
DYNAMIC_FEATURE_IDX = list(range(0, 19))
STATIC_FEATURE_IDX = [19, 20, 21]


CRITERION_REGISTRY = {
    "MAE": torch.nn.L1Loss,
    "MSE": torch.nn.MSELoss,
}

MODEL_REGISTRY = {}


class ModelBuilder:
    def __init__(self, model_config):
        self.model_name = model_config["model_name"]
        self.model_kwargs = model_config["model_kwargs"]

    def build(self):
        return MODEL_REGISTRY[self.model_name](**self.model_kwargs)


class TSMixerExtWrapper:
    def __init__(self, seq_len, pred_len, enc_in, criterion,
                 hidden_channels=64, static_channels=3, extra_channels=19,
                 lr=1e-3, kwargs={}):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.model = TSMixerExt(
            sequence_length=seq_len,
            prediction_length=pred_len,
            input_channels=1,
            extra_channels=extra_channels,
            hidden_channels=hidden_channels,
            static_channels=static_channels,
            output_channels=1,
            **kwargs,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = CRITERION_REGISTRY[criterion]()

    def _split_batch(self, x_enc, y_target):
        x_hist = x_enc[:, :, TARGET_IDX:TARGET_IDX + 1]
        x_extra_hist = x_enc[:, :, DYNAMIC_FEATURE_IDX]
        x_extra_future = y_target[:, :, DYNAMIC_FEATURE_IDX]
        x_static = x_enc[:, 0, STATIC_FEATURE_IDX]
        y_true = y_target[:, :, TARGET_IDX:TARGET_IDX + 1]
        return x_hist, x_extra_hist, x_extra_future, x_static, y_true

    def fit(self, df_train, df_val, save_path, batch_size=128, max_epochs=100, patience=5):
        os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

        dataset_train = CustomDataset(df_train, self.seq_len, self.pred_len)
        dataset_val = CustomDataset(df_val, self.seq_len, self.pred_len)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_vloss = float("inf")
        counter = 0

        for epoch in range(max_epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=True)
            for batch in progress_bar:
                x_enc, y_target, _ = batch
                x_enc = x_enc.to(self.device)
                y_target = y_target.to(self.device)
                
                x_hist, x_extra_hist, x_extra_future, x_static, y_true = self._split_batch(x_enc, y_target)
                self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(x_hist, x_extra_hist, x_extra_future, x_static)
                loss = self.criterion(predictions, y_true)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

            self.model.eval()
            running_vloss = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx_enc, vy_target, _ = vbatch
                    vx_enc = vx_enc.to(self.device)
                    vy_target = vy_target.to(self.device)

                    vx_hist, vx_extra_hist, vx_extra_future, vx_static, vy_true = self._split_batch(vx_enc, vy_target)
                    vpred = self.model(vx_hist, vx_extra_hist, vx_extra_future, vx_static)
                    vloss = self.criterion(vpred, vy_true)
                    running_vloss += vloss.item()

            avg_tloss = running_loss / len(train_loader)
            avg_vloss = running_vloss / len(val_loader)
            print(f"Epoch {epoch+1}/{max_epochs} - Avg train loss: {avg_tloss:.5f} - Avg val loss: {avg_vloss:.5f}")

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join(save_path, "checkpoints", f"model_{timestamp}_{epoch}.pt")
                torch.save(self.model.state_dict(), model_path)
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def save(self, path):
        torch.save(self.model.state_dict(), path)


MODEL_REGISTRY["TSMixerExt"] = TSMixerExtWrapper
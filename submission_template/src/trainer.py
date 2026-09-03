import os
import torch
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from data_loader import CustomDataset
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, base_model=None, as_feature=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.as_feature = as_feature # weather the output from the hybrid model should serve as additional feature or for training on residuals
        self.base_model = base_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_tracker = []
        self.vloss_tracker = []
        self.vloss_tf_tracker = []  # teacher-forcing validation loss, observation only

    def fit(self, save_path, train_loader, df_train, df_val, loss_on='target', loss_weighted=False, alpha=None, mode='single_forecast', ar_steps=None, min_epochs=30, max_epochs=100, patience=10, grad_acc=False, grad_acc_steps=None, max_norm=1.0):
        # mode single_forecast means we only use the direct output of xLSTM-Mixer (exactly pred_len steps) to compute training loss
        # mode "autoregressive" means we use xLSTM-Mixer predict pred_len steps, then autoregressively feed predictions back into model
        # to obtain next pred_len steps (just like in validation), repeating ar_steps times
        # if mode "autoregressive" then the data loader must be configured in such a way that
        # y contains at least ar_steps*pred_len time steps.
        os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        target_idx = train_loader.dataset.target_idx
        best_vloss = float('inf')
        counter = 0

        # Teacher-forcing validation set: same seq_len/pred_len as the model's own architecture
        # (a single direct forward pass, not an autoregressive rollout), built once up-front
        # since df_val doesn't change across epochs. This is a diagnostic metric only - it must
        # never feed early stopping or the scheduler, both of which stay driven by the existing
        # rollout validation loss below.
        dataset_val_tf = CustomDataset(
            df_val, self.model.seq_len, self.model.pred_len, stride=self.model.pred_len
        )
        val_loader_tf = DataLoader(dataset_val_tf, train_loader.batch_size, shuffle=False)

        for epoch in range(max_epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=True)
            if grad_acc:
                self.optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(progress_bar):
                x_enc, y, series_idx, _, idx = batch
                x_enc = x_enc.to(self.device)
                y = y.to(self.device)
                if grad_acc == False:
                    self.optimizer.zero_grad(set_to_none=True)
                if mode=='single_forecast':
                    predictions = self.model.predict(x_enc, series_idx)
                elif mode=='autoregressive':
                    insample_fit_batch = None
                    if self.base_model is not None and self.as_feature == False:
                        # CustomDataset replaces the target channel with the in-sample residual
                        # for hybrid+as_feature=False; the tensor rollout needs the genuine raw
                        # history back to refit the base model at each step.
                        insample_fit_batch = torch.from_numpy(
                            train_loader.dataset.insample_fit[idx.numpy()]
                        ).float().to(self.device)
                    predictions = self.model.predict_autoregressive_tensor(
                        x_enc, y, series_idx, target_idx, ar_steps,
                        base_model=self.base_model, as_feature=self.as_feature,
                        insample_fit=insample_fit_batch,
                    )

                if loss_on=='all':
                    loss = self.criterion(predictions[:,:,:y.shape[-1]], y)
                elif loss_on=='target':
                    if loss_weighted==True:
                        self.criterion.reduction = 'none'
                        loss = self.criterion(predictions[:,:,target_idx], y[:,:,target_idx])
                        weights = 1.0 + alpha * torch.abs(y[:,:,target_idx])
                        loss = (weights * loss).sum() / weights.sum()
                    else:
                        loss = self.criterion(predictions[:,:,target_idx], y[:,:,target_idx])

                running_loss += loss.item()

                if grad_acc == True:
                    loss = loss/grad_acc_steps
                    loss.backward()
                    if ((i + 1) % grad_acc_steps == 0) or (i + 1 == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    self.optimizer.step()

                progress_bar.set_postfix({"loss": f"{running_loss / (i+1):.4f}"})

            self.loss_tracker.append(running_loss/len(train_loader))

            self.model.eval()
            with torch.no_grad():
                if self.base_model is not None:
                    pred_df = self.model.predict_autoregressive_hybrid(
                        df_train,
                        df_val,
                        base_model=self.base_model,
                        as_feature=self.as_feature,
                    )
                else:
                    pred_df = self.model.predict_autoregressive(df_train, df_val)

                pred_eval = pred_df.groupby("series_id").tail(336)
                true_eval = df_val.groupby("series_id").tail(336)

                assert pred_eval.index.equals(true_eval.index)

                pred = torch.tensor(pred_eval["target"].values, dtype=torch.float32, device=self.device)
                true = torch.tensor(true_eval["target"].values, dtype=torch.float32, device=self.device)

                self.criterion.reduction='mean'
                vloss = self.criterion(pred, true).item()

                # Teacher-forcing validation loss - observation only, does not feed the
                # scheduler or early stopping.
                tf_loss_total = 0.0
                tf_batches = 0
                for tf_batch in val_loader_tf:
                    x_enc_v, y_v, series_idx_v, _, _ = tf_batch
                    x_enc_v = x_enc_v.to(self.device)
                    y_v = y_v.to(self.device)
                    preds_v = self.model.predict(x_enc_v, series_idx_v)
                    tf_loss_total += self.criterion(preds_v[:,:,target_idx], y_v[:,:,target_idx]).item()
                    tf_batches += 1
                vloss_tf = tf_loss_total / max(tf_batches, 1)

            self.vloss_tracker.append(vloss)
            self.vloss_tf_tracker.append(vloss_tf)
            print(f"Epoch {epoch+1}/{max_epochs} - Avg. train. Loss: {running_loss / len(train_loader):.5f} - val. Loss (target): {vloss:.5f} - val. Loss (teacher-forced): {vloss_tf:.5f}")

            if self.scheduler is not None:
                self.scheduler.step(vloss)

            print("Current lr:", self.optimizer.param_groups[0]["lr"])

            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
            mlflow.log_metric("val_loss", vloss, step=epoch)
            mlflow.log_metric("val_loss_tf", vloss_tf, step=epoch)
            mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)

            if vloss < best_vloss:
                best_vloss = vloss
                model_path = os.path.join(save_path, "checkpoints", f'model_{timestamp}_{epoch}')
                self.best_model_path = model_path
                torch.save(self.model.state_dict(), model_path)
                mlflow.log_artifact(self.best_model_path, artifact_path="checkpoints")
                mlflow.log_metric("best_val_loss", best_vloss, step=epoch)
                counter = 0
            else:
                counter += 1

            if (counter >= patience) and (epoch>min_epochs):
                break
        self.plot_losses(save_path)
        return self.best_model_path

    def plot_losses(self, save_path):
        plt.plot(self.loss_tracker, label="training loss")
        plt.plot(self.vloss_tracker, label="validation loss")
        plt.plot(self.vloss_tf_tracker, label="validation loss (teacher-forced)")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss.png"), format='PNG')

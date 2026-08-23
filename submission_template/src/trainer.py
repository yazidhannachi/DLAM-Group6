import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_tracker = []
        self.vloss_tracker = []

    def fit(self, save_path, train_loader, df_train, df_val, loss_on='target', mode='single_forecast', ar_steps=None, min_epochs=30, max_epochs=100, patience=10):
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

        for epoch in range(max_epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=True)
            for i, batch in enumerate(train_loader):
                x_enc, y, series_idx, _ = batch
                x_enc = x_enc.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                if mode=='single_forecast':
                    predictions = self.model.predict(x_enc, series_idx)
                elif mode=='autoregressive':
                    predictions = self.model.predict_autoregressive_tensor(x_enc, y, series_idx, target_idx, ar_steps)

                if loss_on=='all':
                    loss = self.criterion(predictions[:,:,:y.shape[-1]], y)
                elif loss_on=='target':
                    loss = self.criterion(predictions[:,:,target_idx], y[:,:,target_idx])
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 50 == 0 or i==(len(train_loader)-1):
                    step = i % 50 if (i == len(train_loader) and i % 50 != 0) else 50
                    progress_bar.set_postfix({"loss": f"{running_loss / (i+1):.4f}"})
                    progress_bar.update(step)

            self.loss_tracker.append(running_loss/len(train_loader))

            progress_bar.close()
            #running_vloss = 0.0
        
            self.model.eval()
            #val_bar = tqdm(val_loader, desc=f"Validation", leave=False)
            with torch.no_grad():
                '''for vdata in val_loader:
                    vx, vy, series_idx = vdata
                    vx = vx.to(self.device)
                    vy = vy.to(self.device)
                    vpred= self.model.predict(vx, series_idx=series_idx)
                    vloss = self.criterion(vpred[:,:,:vy.shape[-1]], vy)
                    running_vloss += vloss.item()
                    val_bar.update(1)
                    val_bar.set_postfix(batch_loss=f"{vloss.item():.4f}")'''
                pred_df = self.model.predict_autoregressive(df_train, df_val)
                assert pred_df.index.equals(df_val.index)
                pred = torch.tensor(pred_df["target"].values, dtype=torch.float32, device=self.device)
                true = torch.tensor(df_val["target"].values, dtype=torch.float32, device=self.device)
                vloss = self.criterion(pred, true).item()

            self.vloss_tracker.append(vloss)
            print(f"Epoch {epoch+1}/{max_epochs} - Avg. train. Loss: {running_loss / len(train_loader):.5f} - val. Loss (target): {vloss:.5f}")
            #val_bar.close()

            self.scheduler.step(vloss)
            print("Current lr:", self.optimizer.param_groups[0]["lr"])

            if vloss < best_vloss:
                best_vloss = vloss
                model_path = os.path.join(save_path, "checkpoints", f'model_{timestamp}_{epoch}')
                self.best_model_path = model_path
                torch.save(self.model.state_dict(), model_path)
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
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss.png"), format='PNG')
import hashlib
import os
import torch
import matplotlib.pyplot as plt
from torchmetrics.regression import MeanSquaredError,  MeanAbsoluteError, MeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError
import pandas as pd

def hash_config(config_path):
    '''
    Computes hash from config file.
    '''
    with open(config_path, 'rb') as f:
        hash_func = hashlib.new("shake128")
        hash_func.update(f.read())
        file_hash = hash_func.hexdigest(length=4)
    return file_hash

def extract_initial_history(train_df, seq_len):

    history_df = (
        train_df.groupby('remainder__series_id', group_keys=False)
        .apply(lambda x: x.tail(seq_len))
        .sort_values(by='remainder__series_id')
    )
    return history_df

def compute_metrics(y_pred, y_val):
    y_pred_ = torch.tensor(y_pred, dtype=torch.float32)
    y_val_ = torch.tensor(y_val, dtype=torch.float32)

    mse_metric = MeanSquaredError()
    rmse_metric = MeanSquaredError(squared=False)
    mae_metric = MeanAbsoluteError()
    mape_metric = MeanAbsolutePercentageError()
    smape_metric = SymmetricMeanAbsolutePercentageError()

    mse = mse_metric(y_pred_, y_val_).numpy()
    rmse = rmse_metric(y_pred_, y_val_).numpy()
    mae = mae_metric(y_pred_, y_val_).numpy()
    mape = mape_metric(y_pred_, y_val_).numpy()
    smape = smape_metric(y_pred_, y_val_).numpy()

    wape = torch.sum(torch.abs(y_val_ - y_pred_)) / torch.sum(torch.abs(y_val_))
    wape = wape.numpy()

    data = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "WAPE": wape, "sMAPE": smape}
    df = pd.DataFrame(data, index=[0])

    return df

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def plot_preds(save_dir, pred_df, val_df):
    fix, axs = plt.subplots(5,1,figsize=(3,10))
    unit_names = ["unit_000", "unit_005", "unit_030", "unit_075", "unit_090"]
    groups_pred = pred_df.groupby("series_id")
    groups_val = val_df.groupby("series_id")
    for i, unit in enumerate(unit_names):
        group_val = groups_val.get_group(unit)
        group_pred = groups_pred.get_group(unit)
        axs[i].plot(group_val["target"].values, label="val")
        axs[i].plot(group_pred["target"].values, label="pred")
        axs[i].legend()

    plt.savefig(os.path.join(save_dir, "pred_plots.png"), format="PNG")



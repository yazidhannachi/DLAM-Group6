import hashlib
import torch
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


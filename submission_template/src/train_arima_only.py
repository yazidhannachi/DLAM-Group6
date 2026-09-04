import yaml
import random
import argparse
import os
import torch
import pandas as pd
import numpy as np
import mlflow
from experiment_tracking import get_git_info, write_git_diff

from utils import hash_config, compute_metrics, flatten_dict, plot_preds
from models import ARIMAWrapper
from preprocess import Preprocessor
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('-t', action='store_true') # test mode
args = parser.parse_args()

config_path = args.config
test_mode = args.t

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

mlflow.set_experiment(config["experiment"])
with mlflow.start_run():

    seed=config.get("seed", 8)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    flat_config = flatten_dict(config)
    for k, v in flat_config.items():
        if isinstance(v, (str, int, float, bool)):
            mlflow.log_param(k, v)
        else:
            mlflow.log_param(k, str(v))

    mlflow.log_artifact(config_path, artifact_path="config")

    config_hash = hash_config(config_path)
    mlflow.log_param("config_hash", config_hash)

    git_info, repo = get_git_info(".")
    for k, v in git_info.items():
        mlflow.log_param(k, v)

    diff_path = write_git_diff(repo)
    if diff_path is not None:
        mlflow.log_artifact(diff_path, artifact_path="git")

    pp_config = config["preprocessing"]
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    if test_mode: 
        os.makedirs('experiments/testing/', exist_ok=True)
        save_dir = 'experiments/testing'
    else:
        hybrid = model_config["hybrid"]
        name_dict = {True: "_arima_only_", False: "_"}
        run_id = mlflow.active_run().info.run_id
        save_dir = os.path.join(
            data_config["save_path"],
            f"{model_config['model_name']}{name_dict[hybrid]}{config_hash}_{run_id}"
        )
        os.makedirs(save_dir, exist_ok=True)

    mlflow.log_param("save_dir", save_dir)

    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
        
    splits = {'train': 'train.csv', 'validation': 'validation_input.csv'}
    df_train = pd.read_csv("data/train_data.csv", index_col=0)
    df_val = pd.read_csv("data/val_data.csv", index_col=0)
    df_train = df_train.sort_values(['series_id', 'timestamp'])
    df_val = df_val.sort_values(['series_id', 'timestamp'])

    train_dfs = []
    val_dfs = []
    seq_len = data_config["seq_len"]

    for series_id, group in df_train.groupby('series_id'):
        n = len(group)

        train_dfs.append(group.iloc[:-336]) # 336 because that is how much we are supposed to predict on the actual validation set for each series

        val_start = max(0, n - 336)
        val_dfs.append(group.iloc[val_start:])

    df_train_local = pd.concat(train_dfs, axis=0)
    df_val_local = pd.concat(val_dfs, axis=0)

    series_encoder = LabelEncoder()
    df_train_local['series_idx'] = series_encoder.fit_transform(df_train_local['series_id'])
    df_val_local['series_idx'] = series_encoder.transform(df_val_local['series_id'])
    df_val['series_idx'] = series_encoder.transform(df_val['series_id'])

    X_train_local = df_train_local.drop(columns=["target"])
    y_train_local = df_train_local[['series_id', "target"]]
    X_val_local   = df_val_local.drop(columns=["target"])
    y_val_local   = df_val_local[['series_id', "target"]]

    preprocessor = Preprocessor(pp_config)
    X_train_local_clean, y_train_local_clean = preprocessor.fit_transform(X_train_local, y_train_local)
    X_val_local_clean, y_val_local_clean = preprocessor.transform(X_val_local, y_val_local)
    df_train_local_clean = pd.concat([X_train_local_clean, y_train_local_clean["target"]], axis=1)
    df_val_local_clean = pd.concat([X_val_local_clean, y_val_local_clean["target"]], axis=1)

    if data_config.get("add_trailing_stats", False):
        df_train_local_clean["target_roll_mean_24"] = (
            df_train_local_clean.groupby("series_id")["target"]
            .transform(lambda s: s.rolling(window=24, min_periods=1).mean())
        )
        df_val_local_clean["target_roll_mean_24"] = (
        df_val_local_clean.groupby("series_id")["target"]
        .transform(lambda s: s.rolling(window=24, min_periods=1).mean())
        )
        df_train_local_clean["target_roll_std_24"] = (
            df_train_local_clean.groupby("series_id")["target"]
            .transform(lambda s: s.rolling(window=24, min_periods=1).std(ddof=0))
        )
        df_val_local_clean["target_roll_std_24"] = (
        df_val_local_clean.groupby("series_id")["target"]
        .transform(lambda s: s.rolling(window=24, min_periods=1).std(ddof=0))
        )

    if data_config["input"] == "target_only":
        keep_cols = ["series_id", "series_idx", "timestamp", "target"]
    elif data_config["input"] == "partial":
        selected_features = data_config["selected_features"]
        keep_cols = ["series_id", "series_idx", "timestamp", "target"] + selected_features
    else:
        keep_cols = df_train_local_clean.columns
        
    df_train_local_clean = df_train_local_clean[keep_cols]
    df_val_local_clean = df_val_local_clean[keep_cols]

    model_cols = [c for c in df_train_local_clean.columns if c not in ["series_id", "series_idx", "timestamp"]]
    enc_in = len(model_cols)
    if model_config.get("prior_as_feature")==True:
        enc_in += 1 # we add prior as a feature, so enc_in must be increased

    model_config["model_kwargs"].update({"enc_in": enc_in})
    mlflow.log_param("model_kwargs.enc_in", enc_in)

    def predict_arima_only(history_df, df_val, feature_cols, base_model):
        pred_df = df_val.copy()
        pred_df["target"] = np.nan

        target_idx = feature_cols.index("target")
        exog_indices = base_model._get_exog_indices(feature_cols, target_idx)

        for series_id, group in df_val.groupby("series_id"):
            series_history = history_df.loc[history_df["series_id"] == series_id, feature_cols].copy()
            series_future = df_val.loc[df_val["series_id"] == series_id, feature_cols].copy()

            hist_np = series_history.values
            fut_np = series_future.values

            y_hist = hist_np[:, target_idx]

            if len(exog_indices) > 0:
                x_hist = hist_np[:, exog_indices]
                x_future = fut_np[:, exog_indices]
            else:
                x_hist = None
                x_future = None

            base_model.fit(y_hist, exog=x_hist)
            y_pred = base_model.predict_future(len(series_future), exog_future=x_future)

            pred_df.loc[group.index, "target"] = y_pred

        return pred_df

    base_model = ARIMAWrapper(**model_config["base_model_kwargs"])

    pred_df = predict_arima_only(
        df_train_local_clean,
        df_val_local_clean,
        model_cols,
        base_model
    )

    target_predictions = pred_df[["series_id", "target"]]
    target_predictions_rescaled = preprocessor.inverse_transform(target_predictions)

    metrics_df = compute_metrics(
        target_predictions_rescaled["target"].values,
        y_val_local["target"].values
    )
    
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))
    merged_df = pd.merge(
        y_val_local,
        target_predictions_rescaled[["target"]],
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=("_actual", "_pred")
    )
    merged_df.to_csv(os.path.join(save_dir, "raw_preds.csv"))

    mlflow.log_metric("val_MSE_rescaled", metrics_df["MSE"].values)
    mlflow.log_metric("val_MAE_rescaled", metrics_df["MAE"].values)

    #model.fit(df_train_local_clean, df_val_local_clean, save_dir)
    #model.save(os.path.join(save_dir, model_config["save_name"]))

    mlflow.log_artifact(os.path.join(save_dir, "metrics.csv"))
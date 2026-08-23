import yaml
import argparse
import os
import torch
import pandas as pd
import mlflow
import mlflow.pytorch
from experiment_tracking import get_git_info, write_git_diff

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils import hash_config, compute_metrics, flatten_dict, plot_preds
from preprocess import Preprocessor
from models import ModelBuilder
from data_loader import CustomDataset
from trainer import Trainer
from sklearn.preprocessing import LabelEncoder

CRITERION_REGISTRY = {
    "MAE": torch.nn.L1Loss,
    "MSE": torch.nn.MSELoss
}

OPTIMIZER_REGISTRY = {
    "Adam": torch.optim.Adam
}

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

    flat_config = flatten_dict(config)
    for k, v in flat_config.items():
        if isinstance(v, (str, int, float, bool)):
            mlflow.log_param(k, v)
        else:
            mlflow.log_param(k, str(v))


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
        name_dict = {True: "_hybrid_", False: "_"}
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

        val_start = max(0, n - 336 - seq_len)
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

    if training_config["mode"] == "single_forecast":
        dataset_train = CustomDataset(df_train_local_clean, data_config["seq_len"], data_config["pred_len"], stride=data_config["stride"])
    else:
        pred_len = training_config["ar_steps"]*data_config["pred_len"]
        dataset_train = CustomDataset(df_train_local_clean, data_config["seq_len"], pred_len, stride=data_config["stride"])

    train_loader = DataLoader(dataset_train, training_config["batch_size"], shuffle=True)

    model_cols = [c for c in df_train_local_clean.columns if c not in ["series_id", "series_idx", "timestamp"]]
    enc_in = len(model_cols)
    model_config["model_kwargs"].update({"enc_in": enc_in})
    mlflow.log_param("model_kwargs.enc_in", enc_in)
    model_builder = ModelBuilder(model_config)
    model = model_builder.build()

    criterion = CRITERION_REGISTRY[training_config["criterion"]]()
    optimizer = OPTIMIZER_REGISTRY[training_config["optimizer"]](model.parameters(), **training_config["optimizer_kwargs"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

    trainer = Trainer(model, optimizer, criterion, scheduler)
    best_model_path = trainer.fit(save_dir, train_loader, df_train_local_clean, df_val_local_clean, 
                                loss_on=training_config["loss_on"], mode=training_config["mode"], 
                                ar_steps=training_config["ar_steps"], min_epochs=training_config["min_epochs"], 
                                max_epochs=training_config["max_epochs"], patience=training_config["patience"])

    mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
    model.load_state_dict(torch.load(best_model_path))
    model.load_state_dict(torch.load("experiments/xLSTMMixer_90a5961e_ab874efe01114e18a05ce06be0214162/checkpoints/model_20260823_175843_6"))
    model.eval()

    pred_df = model.predict_autoregressive(df_train_local_clean, df_val_local_clean)
    target_predictions = pred_df[["series_id", "target"]]
    target_predictions_rescaled = preprocessor.inverse_transform(target_predictions)

    assert target_predictions_rescaled.index.equals(y_val_local.index)

    plot_preds(save_dir, target_predictions_rescaled, y_val_local)
    mlflow.log_artifact(os.path.join(save_dir, "pred_plots.png"))

    merged_df = pd.merge(
    y_val_local,
    target_predictions_rescaled[['target']],  # Select only the predicted values column
    left_index=True,
    right_index=True,
    how='inner',
    suffixes=('_actual', '_pred')
    )

    merged_df.to_csv(os.path.join(save_dir, "raw_preds.csv"))

    metrics_df = compute_metrics(target_predictions_rescaled["target"].values, y_val_local["target"].values)
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))

    #model.fit(df_train_local_clean, df_val_local_clean, save_dir)
    #model.save(os.path.join(save_dir, model_config["save_name"]))'''

    mlflow.log_artifact(os.path.join(save_dir, "config.yaml"))
    mlflow.log_artifact(os.path.join(save_dir, "loss.png"))
    mlflow.log_artifact(os.path.join(save_dir, "metrics.csv"))
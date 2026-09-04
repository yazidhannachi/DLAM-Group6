import yaml
import random
import argparse
import os
import torch
import pandas as pd
import numpy as np
import mlflow
import hashlib
import mlflow.pytorch
from experiment_tracking import get_git_info, write_git_diff

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from utils import hash_config, compute_metrics, flatten_dict, plot_preds
from preprocess import Preprocessor
from models import ModelBuilder, FourierApprox, ARIMAWrapper
from data_loader import CustomDataset
from trainer import Trainer
from sklearn.preprocessing import LabelEncoder

CRITERION_REGISTRY = {
    "MAE": torch.nn.L1Loss,
    "MSE": torch.nn.MSELoss,
    "Huber": torch.nn.HuberLoss
}

OPTIMIZER_REGISTRY = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW
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

    series_encoder = LabelEncoder()
    df_train['series_idx'] = series_encoder.fit_transform(df_train['series_id'])
    df_val['series_idx'] = series_encoder.transform(df_val['series_id'])

    X_train = df_train.drop(columns=["target"])
    y_train = df_train[['series_id', "target"]]
    X_val = df_val.copy()

    preprocessor = Preprocessor(pp_config)
    X_train_clean, y_train_clean = preprocessor.fit_transform(X_train, y_train)
    X_val_clean, _ = preprocessor.transform(X_val, y_train)
    df_train_clean = pd.concat([X_train_clean, y_train_clean["target"]], axis=1)
    df_val_clean = X_val_clean.copy()
    df_val_clean["target"] = np.nan

    if data_config.get("add_trailing_stats", False):
        df_train_clean["target_roll_mean_24"] = (
            df_train_clean.groupby("series_id")["target"]
            .transform(lambda s: s.rolling(window=24, min_periods=1).mean())
        )

        df_train_clean["target_roll_std_24"] = (
            df_train_clean.groupby("series_id")["target"]
            .transform(lambda s: s.rolling(window=24, min_periods=1).std(ddof=0))
        )

    if data_config["input"] == "target_only":
        keep_cols = ["series_id", "series_idx", "timestamp", "target"]
    elif data_config["input"] == "partial":
        selected_features = data_config["selected_features"]
        keep_cols = ["series_id", "series_idx", "timestamp", "target"] + selected_features
    else:
        keep_cols = df_train_clean.columns
        
    df_train_clean = df_train_clean[keep_cols]
    df_val_clean = df_val_clean[keep_cols]

    if training_config["mode"] == "single_forecast":
        dataset_train = CustomDataset(df_train_clean, data_config["seq_len"], data_config["pred_len"], stride=data_config["stride"])
    else:
        pred_len = training_config["ar_steps"]*data_config["pred_len"]
        dataset_train = CustomDataset(df_train_clean, data_config["seq_len"], pred_len, stride=data_config["stride"])

    if model_config.get("hybrid")==True:
        if model_config["prior"] == 'ARIMA':
            base_model = ARIMAWrapper(**model_config["base_model_kwargs"])
            feature_signature = "|".join(dataset_train.model_cols)
            feature_hash = hashlib.md5(feature_signature.encode()).hexdigest()[:10]

            prior_signature = str(model_config["base_model_kwargs"])
            prior_hash = hashlib.md5(prior_signature.encode()).hexdigest()[:10]

            arima_file_name = (
                f"data/arima_precomputed/"
                f"arima_{data_config['seq_len']}_{data_config['pred_len']}_{data_config['stride']}_{feature_hash}_{prior_hash}.npz"
            )
            if os.path.exists(arima_file_name):
                prec = np.load(arima_file_name)
                insample_fit = prec["insample_fit"]
                res_hist = prec["res_hist"]
                base_fc = prec["base_fc"]
                res_tgt = prec["res_tgt"]
            else:
                insample_fit, res_hist, base_fc, res_tgt = base_model.precompute_offline(dataset_train)
                np.savez(arima_file_name, insample_fit=insample_fit, res_hist=res_hist, base_fc=base_fc, res_tgt=res_tgt)
            
        elif model_config["prior"] == 'Fourier':
            base_model = FourierApprox(**model_config["base_model_kwargs"])
            insample_fit, res_hist, base_fc, res_tgt = base_model.precompute_offline(dataset_train)
        else:
            raise ValueError("Not a valid prior.")   
        
        dataset_train.insample_fit = insample_fit
        dataset_train.residual_hist = res_hist
        dataset_train.base_forecast = base_fc
        dataset_train.residual_target = res_tgt
        dataset_train.as_feature = model_config["prior_as_feature"]
    else:
        base_model=None

    g = torch.Generator()
    g.manual_seed(5274)  
    train_loader = DataLoader(dataset_train, training_config["batch_size"], shuffle=True, generator=g)

    model_cols = [c for c in df_train_clean.columns if c not in ["series_id", "series_idx", "timestamp"]]
    enc_in = len(model_cols)
    if model_config.get("prior_as_feature")==True:
        enc_in += 1 # we add prior as a feature, so enc_in must be increased

    model_config["model_kwargs"].update({"enc_in": enc_in})
    mlflow.log_param("model_kwargs.enc_in", enc_in)
    model_builder = ModelBuilder(model_config)
    model = model_builder.build()

    criterion = CRITERION_REGISTRY[training_config["criterion"]]()
    optimizer = OPTIMIZER_REGISTRY[training_config["optimizer"]](model.parameters(), **training_config["optimizer_kwargs"])
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
    scheduler = MultiStepLR(optimizer, training_config["milestones"], gamma=0.5)

    as_feature = model_config.get("prior_as_feature", None)
    trainer = Trainer(model, optimizer, criterion, scheduler, base_model=base_model, as_feature=as_feature)
    final_model_path = trainer.fit_full_data(save_dir, train_loader,
                                loss_on=training_config["loss_on"], loss_weighted=training_config.get("loss_weighted",False), 
                                mode=training_config["mode"], alpha=training_config.get("alpha", None),
                                ar_steps=training_config["ar_steps"], num_epochs=training_config["num_epochs"],
                                grad_acc=training_config.get("grad_acc", False), grad_acc_steps=training_config.get("grad_acc_steps", None))

    mlflow.log_artifact(final_model_path, artifact_path="checkpoints")
    model.load_state_dict(torch.load(final_model_path))
    model.eval()

    if model_config.get("hybrid", False):
        as_feature = model_config["prior_as_feature"] 

        if model_config["prior"] == "ARIMA":
            base_model = ARIMAWrapper(**model_config["base_model_kwargs"])
        elif model_config["prior"] == "Fourier":
            base_model = FourierApprox(**model_config["base_model_kwargs"])
        else:
            raise ValueError("Not a valid prior.")

        pred_df = model.predict_autoregressive_hybrid(
            df_train_clean,
            df_val_clean,
            base_model=base_model,
            as_feature=as_feature,
        )
    else:
        pred_df = model.predict_autoregressive(df_train_clean, df_val_clean)

    pred_df_eval = pred_df.groupby("series_id").tail(336).copy()

    target_predictions = pred_df_eval[["series_id", "target"]]
    target_predictions_rescaled = preprocessor.inverse_transform(target_predictions)

    pred_df_eval["prediction"] = target_predictions_rescaled["target"]
    pred_df_eval.to_csv(os.path.join(save_dir, "pred_df.csv"))
    pred_df_reduced = pred_df_eval.loc[:,["series_id", "timestamp", "prediction"]]
    pred_df_reduced.to_csv(os.path.join(save_dir, "pred_df_reduced.csv"))

    #mlflow.log_metric("val_MSE_rescaled", metrics_df["MSE"].values)
    #mlflow.log_metric("val_MAE_rescaled", metrics_df["MAE"].values)

    #model.fit(df_train_clean, df_val_clean, save_dir)
    #model.save(os.path.join(save_dir, model_config["save_name"]))

    #mlflow.log_artifact(os.path.join(save_dir, "loss.png"))
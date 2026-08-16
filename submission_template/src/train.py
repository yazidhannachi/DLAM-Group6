import yaml
import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import hash_config
from preprocess import Preprocessor
from models import ModelBuilder

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('-t', action='store_true') # test mode
args = parser.parse_args()

config_path = args.config
test_mode = args.t

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config_hash = hash_config(config_path)

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
    save_dir = os.path.join(data_config["save_path"], model_config["model_name"] + name_dict[hybrid] + config_hash)
    os.makedirs(save_dir, exist_ok=True)

preprocessor = Preprocessor(pp_config)
X_pipeline, y_pipeline = preprocessor.build()

splits = {'train': 'train.csv', 'validation': 'validation_input.csv'}
df_train = pd.read_csv("data/train_data.csv", index_col=0)
df_val = pd.read_csv("data/val_data.csv", index_col=0)
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1].to_frame()

total_len = len(df_train)
train_end = int(total_len * (1-training_config["val_size"]))
X_train_local = X_train.iloc[:train_end,:]
X_val_local   = X_train.iloc[train_end:,:]
y_train_local = y_train.iloc[:train_end,:]
y_val_local = y_train.iloc[train_end:,:]

X_train_local_clean = X_pipeline.fit_transform(X_train_local)
X_val_local_clean  = X_pipeline.transform(X_val_local)
y_train_local_clean = y_pipeline.fit_transform(y_train_local)
y_val_local_clean = y_pipeline.transform(y_val_local)

#X_test = df_val.copy()
#X_train_clean = X_pipeline.fit_transform(X_train)
#X_test_clean = X_pipeline.transform(X_test)
#y_train_clean = y_pipeline.fit_transform(y_train)

df_train_local_clean = pd.concat([X_train_local_clean, y_train_local_clean], axis=1)
df_val_local_clean = pd.concat([X_val_local_clean, y_val_local_clean], axis=1)
model_builder = ModelBuilder(model_config)
model = model_builder.build()

model.fit(df_train_local_clean, df_val_local_clean, save_dir)
model.save(os.path.join(save_dir, model_config["save_name"]))

import yaml
import argparse
import sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, MissingIndicator, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from utils import hash_config

SCALER_REGISTRY = {
    "standard": StandardScaler
}

IMPUTER_REGISTRY = {
    "iterative": IterativeImputer,
    "knn": KNNImputer,
    "missing": MissingIndicator,
    "simple": SimpleImputer
}

ESTIMATOR_REGISTRY = {
    "rf": RandomForestRegressor,
    "ridge": Ridge
}

class Preprocessor:
    def __init__(self, pp_config):
        self.scale_columns = pp_config["scale_columns"]
        self.simple_impute_columns = pp_config["simple_impute_columns"]
        self.other_impute_columns = pp_config["other_impute_columns"]
        self.missing_indicator_columns = pp_config["missing_indicator_columns"]
        self.target_col = pp_config.get("target_col", "target")

        self.simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
        self.indicator = IMPUTER_REGISTRY["missing"](**pp_config.get("indicator_kwargs", {}))

        imputer_kwargs = pp_config.get("imputer_kwargs", {}).copy()
        if pp_config.get("imputer") == "iterative":
            mice_estimator = ESTIMATOR_REGISTRY[pp_config["mice_estimator"]](
                **pp_config["mice_estimator_kwargs"]
            )
            imputer_kwargs.update({"estimator": mice_estimator})

        self.other_imputer = IMPUTER_REGISTRY[pp_config["imputer"]](**imputer_kwargs)

    def fit_transform(self, X, y):
        self.feature_preprocessors = {}
        self.target_preprocessors = {}

        processed_x_list = []
        processed_y_list = []

        for series_id, group_x in X.groupby("series_id"):
            group_y = y[y["series_id"] == series_id]

            si_scale_pl = Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', self.simple_imputer)
            ])
            o_scale_pl = Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', self.other_imputer)
            ])
            scale_pl = Pipeline([('scaler', StandardScaler())])
            indicator_pl = Pipeline([
                ('indicator', self.indicator),
                ('to_float', FunctionTransformer(lambda x: x.astype(float)))
            ])

            transformers = []
            if self.missing_indicator_columns:
                transformers.append(('indicators', indicator_pl, self.missing_indicator_columns))
            if self.simple_impute_columns:
                transformers.append(('simple_imputed', si_scale_pl, self.simple_impute_columns))
            if self.other_impute_columns:
                transformers.append(('other_imputed', o_scale_pl, self.other_impute_columns))
            if self.scale_columns:
                transformers.append(('scaled', scale_pl, self.scale_columns))

            feature_preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            ).set_output(transform="pandas")

            target_preprocessor = StandardScaler().set_output(transform="pandas")

            X_group = feature_preprocessor.fit_transform(group_x)
            X_group.index = group_x.index

            target_cols = [c for c in group_y.columns if c not in ['series_id', 'series_idx', 'timestamp']]
            y_group = target_preprocessor.fit_transform(group_y[target_cols])
            y_group = pd.DataFrame(y_group, index=group_y.index, columns=target_cols)
            y_group["series_id"] = group_y["series_id"]

            processed_x_list.append(X_group)
            processed_y_list.append(y_group)

            self.feature_preprocessors[series_id] = feature_preprocessor
            self.target_preprocessors[series_id] = target_preprocessor

        X_preprocessed = pd.concat(processed_x_list, axis=0).sort_index()
        y_preprocessed = pd.concat(processed_y_list, axis=0).sort_index()

        return X_preprocessed, y_preprocessed

    def transform(self, X, y):
        processed_x_list = []
        processed_y_list = []

        for series_id, group_x in X.groupby("series_id"):
            group_y = y[y["series_id"] == series_id]

            X_group = self.feature_preprocessors[series_id].transform(group_x)
            X_group.index = group_x.index

            target_cols = [c for c in group_y.columns if c not in ['series_id', 'series_idx', 'timestamp']]
            y_group = self.target_preprocessors[series_id].transform(group_y[target_cols])
            y_group = pd.DataFrame(y_group, index=group_y.index, columns=target_cols)
            y_group["series_id"] = group_y["series_id"]

            processed_x_list.append(X_group)
            processed_y_list.append(y_group)

        X_preprocessed = pd.concat(processed_x_list, axis=0).sort_index()
        y_preprocessed = pd.concat(processed_y_list, axis=0).sort_index()

        return X_preprocessed, y_preprocessed

    def inverse_transform(self, y):
        processed_y_list = []

        for series_id, group_y in y.groupby("series_id"):
            target_tf = self.target_preprocessors[series_id]
            target_vals = group_y[[col for col in group_y.columns if col != "series_id"]]

            y_unscaled = target_tf.inverse_transform(target_vals)
            y_unscaled = pd.DataFrame(y_unscaled, index=group_y.index, columns=target_vals.columns)
            y_unscaled["series_id"] = series_id

            processed_y_list.append(y_unscaled)

        return pd.concat(processed_y_list, axis=0).sort_index()
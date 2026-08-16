import yaml
import argparse

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, MissingIndicator, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
        self.scaler = SCALER_REGISTRY[pp_config["scaler"]](**pp_config["scaler_kwargs"]).set_output(transform="pandas")

        imputer_kwargs = pp_config["imputer_kwargs"]
        if pp_config["imputer"] == "iterative":
            mice_estimator = ESTIMATOR_REGISTRY[pp_config["mice_estimator"]](**pp_config["mice_estimator_kwargs"])
            imputer_kwargs.update({"estimator": mice_estimator})

        self.other_imputer = IMPUTER_REGISTRY[pp_config["imputer"]](**imputer_kwargs)
        self.simple_imputer = SimpleImputer(strategy="constant", fill_value=0)

        self.indicator = IMPUTER_REGISTRY["missing"](**pp_config["indicator_kwargs"])

        self.scale_columns = pp_config["scale_columns"]
        self.simple_impute_columns = pp_config["simple_impute_columns"]
        self.other_impute_columns = pp_config["other_impute_columns"]
        self.missing_indicator_columns = pp_config["missing_indicator_columns"]

    def build(self):
        si_scale_pl = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', self.simple_imputer)
            ])
        o_scale_pl = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', self.other_imputer)
            ])
        scale_pl = Pipeline([('scaler', StandardScaler())])
        
        feature_preprocessor = ColumnTransformer(
            transformers=[
                ('indicators', self.indicator, self.missing_indicator_columns),
                ('simple_imputed', si_scale_pl, self.simple_impute_columns),
                ('other_imputed', o_scale_pl, self.other_impute_columns),
                ('scaled', scale_pl, self.scale_columns)
                ],
                remainder='passthrough' 
            ).set_output(transform="pandas")
        target_preprocessor = StandardScaler().set_output(transform="pandas")

        return feature_preprocessor, target_preprocessor

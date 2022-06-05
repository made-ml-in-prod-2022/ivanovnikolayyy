from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml_project.params.feature_params import FeatureParams


def preprocess_categorical_features(
    categorical_df: pd.DataFrame,
) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            ),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def preprocess_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    numerical_pipeline = build_numerical_pipeline()
    return pd.DataFrame(numerical_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scale", MinMaxScaler()),
        ]
    )
    return numerical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_features_and_target(
    df: pd.DataFrame, params: FeatureParams
) -> Tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[params.target_col])
    target = df[params.target_col]

    return features, target

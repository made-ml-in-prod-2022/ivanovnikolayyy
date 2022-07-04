import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml_project.params.feature_params import FeatureParams


def make_categorical_pipeline() -> Pipeline:
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


def make_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scale", MinMaxScaler()),
        ]
    )
    return numerical_pipeline


def make_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                make_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                make_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer

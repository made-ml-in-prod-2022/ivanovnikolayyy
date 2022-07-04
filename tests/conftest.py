import os

import pytest
from sklearn.pipeline import Pipeline

from ml_project.classifiers import make_classifier
from ml_project.features import make_transformer
from ml_project.params import ClassifierParams, FeatureParams

DEFAULT_NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
DEFAULT_CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


@pytest.fixture()
def categorical_features():
    return DEFAULT_CATEGORICAL_FEATURES


@pytest.fixture()
def numerical_features():
    return DEFAULT_NUMERICAL_FEATURES


@pytest.fixture()
def dataset_path():
    return os.path.dirname(__file__)


@pytest.fixture()
def model(categorical_features, numerical_features):
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )
    transforms = make_transformer(feature_params)
    clf = make_classifier(ClassifierParams())
    model = Pipeline([("transforms", transforms), ("clf", clf)])

    return model

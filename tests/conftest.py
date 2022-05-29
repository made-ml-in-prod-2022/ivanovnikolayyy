import os
from typing import List

import pytest
from sklearn.pipeline import Pipeline

from ml_project.features import build_transformer
from ml_project.models import build_classifier
from ml_project.params import (DEFAULT_CATEGORICAL_FEATURES,
                               DEFAULT_NUMERICAL_FEATURES, ClassifierParams,
                               FeatureParams)


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def target_col() -> str:
    return "condition"


@pytest.fixture()
def categorical_features() -> List[str]:
    return DEFAULT_CATEGORICAL_FEATURES


@pytest.fixture
def numerical_features() -> List[str]:
    return DEFAULT_NUMERICAL_FEATURES


@pytest.fixture()
def model():
    transforms = build_transformer(FeatureParams())
    clf = build_classifier(ClassifierParams())
    model = Pipeline([("transforms", transforms), ("clf", clf)])

    return model

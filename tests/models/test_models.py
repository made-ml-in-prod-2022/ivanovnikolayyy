import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_project.data import read_data
from ml_project.features import extract_features_and_target
from ml_project.models import load_model, save_model
from ml_project.params import FeatureParams


@pytest.fixture
def features_and_target(
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    data = read_data(dataset_path)
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    features, target = extract_features_and_target(data, params)

    return features, target


def test_train_model(model, features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model.fit(features, target)

    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(model, tmpdir: Path):
    expected_output = os.path.join(tmpdir, "model.pkl")
    real_output = save_model(model, expected_output)

    assert real_output == expected_output
    assert os.path.exists

    loaded_model = load_model(real_output)
    assert isinstance(loaded_model, Pipeline)

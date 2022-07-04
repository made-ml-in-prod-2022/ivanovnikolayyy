import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_project.classifiers import load_model, save_model
from ml_project.data import read_data


@pytest.fixture
def data_and_targets(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data, targets = read_data(dataset_path)

    return data, targets


def test_train_model(model, data_and_targets: Tuple[pd.DataFrame, pd.Series]):
    data, targets = data_and_targets
    model.fit(data, targets)

    assert model.predict(data).shape[0] == targets.shape[0]


def test_serialize_model(model, tmpdir: Path):
    expected_output = os.path.join(tmpdir, "model.pkl")
    real_output = save_model(model, expected_output)

    assert real_output == expected_output
    assert os.path.exists

    loaded_model = load_model(real_output)
    assert isinstance(loaded_model, Pipeline)

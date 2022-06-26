from typing import List

import numpy as np
import pandas as pd
import pytest

from ml_project.features import (make_categorical_pipeline,
                                 make_numerical_pipeline)


@pytest.fixture()
def categorical_feature() -> str:
    return "categorical_feature"


@pytest.fixture()
def categorical_values() -> List[str]:
    return ["category0", "category1", "category2"]


@pytest.fixture()
def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
    return categorical_values + [np.nan]


@pytest.fixture
def fake_categorical_data(
    categorical_feature: str, categorical_values_with_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame({categorical_feature: categorical_values_with_nan})


def test_process_categorical_features(
    fake_categorical_data: pd.DataFrame,
):
    categorical_pipeline = make_categorical_pipeline()
    transformed = categorical_pipeline.fit_transform(fake_categorical_data)
    assert transformed.shape[1] == 3
    assert transformed.sum().sum() == 4

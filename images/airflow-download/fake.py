from typing import Tuple

import numpy as np
import pandas as pd

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
TARGET = "condition"


def build_fake_data(
    reference_data_path: str, size: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(reference_data_path)

    numerical_means = data[NUMERICAL_FEATURES].mean().values
    numerical_stds = data[NUMERICAL_FEATURES].std().values

    fake_numerical_data = np.zeros((size, len(NUMERICAL_FEATURES)))
    for i in range(size):
        fake_numerical_data[i] = np.random.normal(numerical_means, numerical_stds)

    fake_categorical_data = np.zeros((size, len(CATEGORICAL_FEATURES)))
    for i, column in enumerate(CATEGORICAL_FEATURES):
        fake_categorical_data[:, i] = data[column].sample(size, replace=True).values

    fake_data = np.concatenate((fake_numerical_data, fake_categorical_data), axis=1)
    fake_data = pd.DataFrame(
        fake_data, columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    fake_targets = pd.DataFrame(data[TARGET].sample(size, replace=True)).reset_index(
        drop=True
    )

    return fake_data, fake_targets

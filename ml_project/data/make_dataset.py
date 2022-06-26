import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.params import SplittingParams


def read_data(dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(os.path.join(dataset_path, "data.csv"))
    targets = pd.read_csv(os.path.join(dataset_path, "targets.csv")).values.flatten()

    return data, targets


def split_train_test_data(
    data: pd.DataFrame, targets: np.ndarray, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    data_train, data_test, targets_train, targets_test = train_test_split(
        data,
        targets,
        test_size=params.val_size,
        random_state=params.random_state,
        stratify=targets,
    )
    return data_train, data_test, targets_train, targets_test

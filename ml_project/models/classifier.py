import pickle
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml_project.params.classifier_params import ClassifierParams

SklearnClassifier = Union[RandomForestClassifier, LogisticRegression]


def build_classifier(train_params: ClassifierParams) -> SklearnClassifier:
    if train_params.model_type == "RandomForestClassifier":
        clf = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        clf = LogisticRegression(max_iter=train_params.max_iter)
    else:
        raise NotImplementedError()

    return clf


def evaluate_classifier(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def save_model(model: SklearnClassifier, save_path: str) -> str:
    with open(save_path, "wb") as output_stream:
        pickle.dump(model, output_stream)
    return save_path


def load_model(load_path: str) -> Optional[SklearnClassifier]:
    with open(load_path, "rb") as input_stream:
        model = pickle.load(input_stream)
    return model

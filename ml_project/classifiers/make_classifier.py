import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from ml_project.params.classifier_params import ClassifierParams


def make_classifier(train_params: ClassifierParams) -> ClassifierMixin:
    if train_params.model_type == "RandomForestClassifier":
        clf = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        clf = LogisticRegression(max_iter=train_params.max_iter)
    else:
        raise NotImplementedError()

    return clf


def evaluate_classifier(targets: np.ndarray, predicts: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(targets, predicts),
        "precision": precision_score(targets, predicts, average="weighted"),
        "recall": recall_score(targets, predicts, average="weighted"),
        "f1_score": f1_score(targets, predicts, average="weighted"),
    }
    return metrics


def save_model(model: Pipeline, save_path: str) -> str:
    with open(save_path, "wb") as output_stream:
        pickle.dump(model, output_stream)
    return save_path


def load_model(load_path: str) -> Pipeline:
    with open(load_path, "rb") as input_stream:
        model = pickle.load(input_stream)
    return model

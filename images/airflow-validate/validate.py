import json
import os
import pickle

import click
from sklearn.pipeline import Pipeline

from ml_project.classifiers import evaluate_classifier, load_model
from ml_project.data import read_data


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


@click.command("predict")
@click.option("--model-path")
@click.option("--test-dataset-dir")
@click.option("--output-dir")
def validate(model_path: str, test_dataset_dir: str, output_dir: str):
    model = load_model(os.path.join(model_path, "model.pkl"))
    data_test, targets_test = read_data(test_dataset_dir)

    predicts = model.predict(data_test)
    metrics = evaluate_classifier(targets_test, predicts)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_path:
        json.dump(metrics, metrics_path)


if __name__ == "__main__":
    validate()

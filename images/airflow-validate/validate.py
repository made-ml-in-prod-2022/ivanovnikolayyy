import os
import json

import click
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

TARGET = "condition"


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
def validate(input_dir: str, output_dir: str):
    model = load_object(os.path.join(input_dir, "model.pkl"))
    data = pd.read_csv(os.path.join(input_dir, "test.csv"))

    predicts = model.predict(data)
    metrics = classification_report(data[TARGET], predicts, output_dict=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    validate()

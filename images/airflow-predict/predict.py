import os

import click

from ml_project.classifiers import load_model
from ml_project.data import read_data


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data, targets = read_data(input_dir)
    model = load_model(os.path.join(model_dir, "model.pkl"))

    os.makedirs(output_dir)
    data["preds"] = model.predict(data)
    data.to_csv(os.path.join(output_dir, "preds.csv"), index=False)


if __name__ == "__main__":
    predict()

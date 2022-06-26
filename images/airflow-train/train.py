import os
from dataclasses import replace

import click

from ml_project import train_pipeline
from ml_project.params.pipeline_params import read_training_pipeline_params


@click.command("train")
@click.option("--train-dataset-dir")
@click.option("--output-dir")
def train(train_dataset_dir: str, output_dir: str):
    params = read_training_pipeline_params("train_config.yaml")
    params = replace(
        params,
        dataset_path=train_dataset_dir,
        output_model_path=os.path.join(output_dir, params.output_model_path),
        metric_path=os.path.join(output_dir, params.metric_path),
    )

    os.makedirs(output_dir, exist_ok=True)
    train_pipeline(params)


if __name__ == "__main__":
    train()

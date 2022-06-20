import os

import click
import pandas as pd

from ml_project.run import train_pipeline
from ml_project.params.pipeline_params import read_training_pipeline_params


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    params = read_training_pipeline_params("train_config.yaml")
    params.input_data_path = os.path.join(input_dir, "train.csv")
    params.output_model_path = os.path.join(output_dir, "model.pkl")
    params.metric_path = os.path.join(output_dir, "metrics.pkl")
    train_pipeline(params)

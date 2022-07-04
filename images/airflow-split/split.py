import os.path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir), "data.csv")
    train, test = train_test_split(data, test_size=0.2)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

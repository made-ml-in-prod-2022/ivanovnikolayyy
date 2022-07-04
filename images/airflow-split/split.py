import logging
import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):
    logging.info("reading data")
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    targets = pd.read_csv(os.path.join(input_dir, "targets.csv"))

    data_train, data_test, targets_train, targets_test = train_test_split(
        data, targets, test_size=0.1
    )

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    data_train.to_csv(os.path.join(output_dir, "train/data.csv"), index=False)
    data_test.to_csv(os.path.join(output_dir, "test/data.csv"), index=False)
    targets_train.to_csv(os.path.join(output_dir, "train/targets.csv"), index=False)
    targets_test.to_csv(os.path.join(output_dir, "test/targets.csv"), index=False)


if __name__ == "__main__":
    split()

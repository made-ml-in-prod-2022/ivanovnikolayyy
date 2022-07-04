import json
import os

import click
import pandas as pd


@click.command("eda")
@click.option("--input-dir")
@click.option("--output-dir")
def eda(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    targets = pd.read_csv(os.path.join(input_dir, "targets.csv"))

    description = data.describe()
    counts = targets["targets"].value_counts().to_dict()

    os.makedirs(output_dir, exist_ok=True)
    description.to_csv(os.path.join(output_dir, "description.csv"))
    with open(os.path.join(output_dir, "description.csv"), "w") as f:
        json.dump(counts, f)


if __name__ == "__main__":
    eda()

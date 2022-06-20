import os

import click
import numpy as np
import pandas as pd

from fake import build_fake_data


@click.command("download")
@click.option("--reference-data-path")
@click.option("--output-dir")
def download(reference_data_path: str, output_dir: str):
    data, targets = build_fake_data(reference_data_path)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "reference_data.csv"), index=False)
    targets.to_csv(os.path.join(output_dir, "targets.csv"), index=False)


if __name__ == '__main__':
    download()
